"""LLM access layer: model factory (primary + fallback), error classification, JSON-output helper.

Every node obtains its model through `get_llm(...)`, never by constructing a client itself. That gives
one seam for (a) switching to the fallback model when the graph's retry policy says so and
(b) substituting a scripted model in tests via `set_llm_factory`.
"""
import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional, TypeVar

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, ValidationError

from backend.config import get_settings

ErrorKind = Literal["rate_limit", "too_large", "malformed", "transient", "fatal"]
T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class LLMSpec:
    """Everything that distinguishes one model instance from another."""
    purpose: str
    temperature: float = 0.3
    max_tokens: int = 1024
    fallback: bool = False
    json_mode: bool = False


LLMFactory = Callable[[LLMSpec], BaseChatModel]
_factory: Optional[LLMFactory] = None


def set_llm_factory(factory: Optional[LLMFactory]) -> None:
    """Override how models are built (tests). Pass None to restore the Groq default."""
    global _factory
    _factory = factory


# What the API has told us about the models we use (process-wide, learned from error messages).
_learned_token_caps: dict[str, int] = {}
_unavailable_models: set[str] = set()


def reset_learned_limits() -> None:
    _learned_token_caps.clear()
    _unavailable_models.clear()


def fallback_available() -> bool:
    s = get_settings()
    return bool(s.GROQ_FALLBACK_MODEL) and s.GROQ_FALLBACK_MODEL != s.GROQ_MODEL and s.GROQ_FALLBACK_MODEL not in _unavailable_models


def _model_name(fallback: bool) -> str:
    s = get_settings()
    return s.GROQ_FALLBACK_MODEL if (fallback and fallback_available()) else s.GROQ_MODEL


def effective_max_tokens(requested: int, *, fallback: bool = False) -> int:
    """`requested`, clamped to the configured ceiling and to any limit learned for the model in use."""
    caps = [c for c in (get_settings().LLM_MAX_OUTPUT_TOKENS, _learned_token_caps.get(_model_name(fallback))) if c]
    return max(64, min([requested, *caps]))


_TOO_LARGE = re.compile(r"request too large for model `([^`]+)`.*?output tokens.*?limit (\d+), requested (\d+)", re.S | re.I)
_MODEL_MISSING = re.compile(r"model `([^`]+)` does not exist", re.I)


def learn_from_error(exc: BaseException) -> None:
    """Record limits the API just revealed, so the retry (and later calls) succeed instead of repeating the mistake."""
    text = str(exc)
    m = _TOO_LARGE.search(text)
    if m:
        # Leave headroom: the limit is per *minute* and other calls share it.
        _learned_token_caps[m.group(1)] = max(200, int(int(m.group(2)) * 0.8))
        return
    m = _MODEL_MISSING.search(text)
    if m and _status_code(exc) == 404:
        s = get_settings()
        if m.group(1) == s.GROQ_FALLBACK_MODEL and m.group(1) != s.GROQ_MODEL:
            _unavailable_models.add(m.group(1))


def _groq_factory(spec: LLMSpec) -> BaseChatModel:
    from langchain_groq import ChatGroq

    settings = get_settings()
    model = _model_name(spec.fallback)
    extra = {"reasoning_effort": "low"} if "gpt-oss" in model else {}
    llm = ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model=model,
        temperature=spec.temperature,
        max_tokens=spec.max_tokens,
        # The graph's retry/fallback edges own the retry policy; disable the SDK's hidden retries
        # so every failure is visible to (and traced by) the graph.
        max_retries=0,
        timeout=60,
        **extra,
    )
    if spec.json_mode:
        return llm.bind(response_format={"type": "json_object"})  # type: ignore[return-value]
    return llm


def get_llm(
    purpose: str,
    *,
    temperature: float = 0.3,
    max_tokens: int = 1024,
    fallback: bool = False,
    json_mode: bool = False,
) -> BaseChatModel:
    fallback = fallback and fallback_available()  # the spec reflects the model that will really be used
    spec = LLMSpec(purpose, temperature, effective_max_tokens(max_tokens, fallback=fallback), fallback, json_mode)
    return (_factory or _groq_factory)(spec)


# --------------------------------------------------------------------------- errors

class MalformedOutputError(ValueError):
    """The model answered, but not with usable / schema-valid output."""


def _status_code(exc: BaseException) -> Optional[int]:
    code = getattr(exc, "status_code", None)
    if code is None:
        code = getattr(getattr(exc, "response", None), "status_code", None)
    return code if isinstance(code, int) else None


def classify_error(exc: BaseException) -> ErrorKind:
    """Map an exception to the recovery strategy the graph should apply.

    too_large  -> retry at once: the output-token limit was just learned and max_tokens now fits
    rate_limit -> switch to the fallback model (its own quota bucket) or wait out Retry-After
    malformed  -> retry, then fall back (a different model often formats correctly)
    transient  -> retry after backoff (5xx / timeouts / connection resets)
    fatal      -> don't retry (bad credentials, invalid request, programming errors)
    """
    explicit = getattr(exc, "error_kind", None)
    if explicit in ("rate_limit", "too_large", "malformed", "transient", "fatal"):
        return explicit  # type: ignore[return-value]
    text = str(exc).lower()
    code = _status_code(exc)

    if isinstance(exc, (MalformedOutputError, json.JSONDecodeError, ValidationError)):
        return "malformed"
    if _TOO_LARGE.search(str(exc)):
        return "too_large"
    if code == 404 and (m := _MODEL_MISSING.search(str(exc))):
        s = get_settings()
        # A missing *fallback* model is recoverable (we carry on with the primary); a missing primary is not.
        return "transient" if (m.group(1) == s.GROQ_FALLBACK_MODEL and m.group(1) != s.GROQ_MODEL) else "fatal"
    if code == 429 or code == 413 or "rate limit" in text or "rate_limit" in text:
        return "rate_limit"
    if code == 400 and any(s in text for s in ("tool_use_failed", "failed to call a function", "json_validate_failed", "failed to generate json")):
        return "malformed"
    if code is not None and (code >= 500 or code in (408, 409)):
        return "transient"
    name = type(exc).__name__.lower()
    if any(s in name for s in ("timeout", "connection", "connecterror", "readerror", "remoteprotocol")):
        return "transient"
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return "transient"
    return "fatal"


def retry_after_seconds(exc: BaseException) -> Optional[float]:
    """Honor the API's Retry-After header when present."""
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if not headers:
        return None
    try:
        value = headers.get("retry-after")
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- JSON helper

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


# A backslash that doesn't start a legal JSON escape (typically LaTeX such as \\mathbf, \\eta, \\sum).
_BAD_ESCAPE = re.compile(r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})')


def _loads_lenient(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return json.loads(_BAD_ESCAPE.sub(r"\\\\", text))


def extract_json(text: str) -> Any:
    """Parse the first JSON object/array in `text`, tolerating code fences, surrounding prose and
    unescaped LaTeX backslashes."""
    cleaned = _FENCE.sub("", text.strip()).strip()
    try:
        return _loads_lenient(cleaned)
    except json.JSONDecodeError:
        pass
    start = min((i for i in (cleaned.find("{"), cleaned.find("[")) if i != -1), default=-1)
    if start == -1:
        raise MalformedOutputError("No JSON found in model output")
    try:
        obj, _ = json.JSONDecoder().raw_decode(_BAD_ESCAPE.sub(r"\\\\", cleaned[start:]))
        return obj
    except json.JSONDecodeError as e:
        raise MalformedOutputError(f"Invalid JSON in model output: {e}") from e


def _has_mangled_latex(value: Any) -> bool:
    """`\\f` / `\\b` decode to form-feed / backspace, which never occur in real text: they mean the model
    wrote LaTeX like \\frac or \\beta with a single backslash inside a JSON string."""
    if isinstance(value, str):
        return "\x0c" in value or "\x08" in value
    if isinstance(value, dict):
        return any(_has_mangled_latex(v) for v in value.values())
    if isinstance(value, list):
        return any(_has_mangled_latex(v) for v in value)
    return False


def message_text(message: Any) -> str:
    """Plain text of a chat message whether content is a string or a list of content blocks."""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b if isinstance(b, str) else b.get("text", "") for b in content if isinstance(b, (str, dict))
        )
    return str(content)


async def ainvoke_json(llm: BaseChatModel, messages: list, schema: type[T], config: Optional[dict] = None) -> T:
    """Call the model and validate its JSON answer against a pydantic schema.

    Anything short of a schema-valid object raises MalformedOutputError, which the graph's
    resilience edges turn into a retry / fallback-model attempt.
    """
    response = await llm.ainvoke(messages, config=config)
    data = extract_json(message_text(response))
    if _has_mangled_latex(data):
        raise MalformedOutputError("Model wrote LaTeX backslashes unescaped inside JSON strings")
    try:
        return schema.model_validate(data)
    except ValidationError as e:
        raise MalformedOutputError(f"Model output failed {schema.__name__} validation: {e.errors()[:2]}") from e


async def ainvoke_text(llm: BaseChatModel, messages: list, config: Optional[dict] = None) -> str:
    """Call the model and return its text, treating an empty answer as malformed (so it gets retried)."""
    text = message_text(await llm.ainvoke(messages, config=config)).strip()
    if not text:
        raise MalformedOutputError("Model returned an empty answer")
    return _FENCE.sub("", text).strip()
