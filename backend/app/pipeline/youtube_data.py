import json
import time
from groq import Groq
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    RequestBlocked
)
from youtube_transcript_api.formatters import TextFormatter, SRTFormatter
from youtube_transcript_api.proxies import WebshareProxyConfig, GenericProxyConfig
import yt_dlp
import os
import subprocess
from skimage.metrics import structural_similarity as ssim
import cv2
from backend.config import get_settings

import re

def extract_id(url: str) -> str:
    """Extract YouTube video ID from URL supporting all YouTube URL formats."""
    pattern = r'(?:v=|\/|be\/|shorts\/|embed\/|live\/)([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    return "None"


class TranscriptUnavailableError(Exception):
    """Raised with a clear, user-facing reason when a transcript truly cannot be retrieved."""
    pass


def _build_transcript_api() -> YouTubeTranscriptApi:
    """Build a YouTubeTranscriptApi client, using a proxy if one is configured.

    A proxy is the officially recommended fix for YouTube blocking transcript
    requests from cloud-hosted server IPs (see youtube-transcript-api README,
    "Working around IP bans"). It's entirely optional: with no proxy env vars
    set, this behaves exactly as before.
    """
    settings = get_settings()
    if settings.WEBSHARE_PROXY_USERNAME and settings.WEBSHARE_PROXY_PASSWORD:
        return YouTubeTranscriptApi(proxy_config=WebshareProxyConfig(
            proxy_username=settings.WEBSHARE_PROXY_USERNAME,
            proxy_password=settings.WEBSHARE_PROXY_PASSWORD,
        ))
    if settings.YT_HTTP_PROXY or settings.YT_HTTPS_PROXY:
        return YouTubeTranscriptApi(proxy_config=GenericProxyConfig(
            http_url=settings.YT_HTTP_PROXY or settings.YT_HTTPS_PROXY,
            https_url=settings.YT_HTTPS_PROXY or settings.YT_HTTP_PROXY,
        ))
    return YouTubeTranscriptApi()


def download_any_transcript(video_id: str, output_dir: str, output_format: str = "txt") -> dict | None:
    """Download transcript and save to output directory.

    Raises TranscriptUnavailableError with a clear, actionable reason if the
    transcript genuinely cannot be retrieved, so the caller can surface it to
    the user instead of a silent generic failure.
    """
    max_attempts = 3
    last_error = None

    for attempt in range(1, max_attempts + 1):
        try:
            ytt = _build_transcript_api()
            transcript_list = ytt.list(video_id)
            manual_transcripts = list(transcript_list._manually_created_transcripts.values())
            generated_transcripts = list(transcript_list._generated_transcripts.values())
            if manual_transcripts:
                transcript = manual_transcripts[0]
            elif generated_transcripts:
                transcript = generated_transcripts[0]
            else:
                transcript = next(iter(transcript_list))
            data = transcript.fetch()
            transcript_dict = [item.__dict__ for item in data]

            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.join(output_dir, f"{transcript.language_code}.{output_format}")

            if output_format == "srt":
                formatter = SRTFormatter()
            else:
                formatter = TextFormatter()
            formatted_output = formatter.format_transcript(data)

            with open(filename, "w", encoding="utf-8") as f:
                f.write(formatted_output)

            json_filename = os.path.join(output_dir, f"{transcript.language_code}.json")
            with open(json_filename, "w", encoding="utf-8") as f:
                json.dump(transcript_dict, f, indent=4, ensure_ascii=False)

            return {
                "language": transcript.language,
                "language_code": transcript.language_code,
                "is_generated": transcript.is_generated,
                "data": data
            }
        except TranscriptsDisabled:
            print(f"Captions are disabled on video: {video_id}")
            raise TranscriptUnavailableError(
                "Captions are disabled for this video. Try a lecture video that has captions/subtitles enabled."
            )
        except NoTranscriptFound:
            print(f"No transcripts found for video: {video_id}")
            raise TranscriptUnavailableError(
                "No transcript/captions could be found for this video."
            )
        except RequestBlocked as e:
            # Often transient rate-limiting rather than a permanent ban — retry with backoff.
            last_error = e
            print(f"[YT TRANSCRIPT] Request blocked by YouTube (attempt {attempt}/{max_attempts}): {e}")
            if attempt < max_attempts:
                time.sleep(2 * attempt)
                continue
            raise TranscriptUnavailableError(
                "YouTube blocked the transcript request from this server's IP. This is common on "
                "cloud-hosted servers. Configure a proxy (WEBSHARE_PROXY_USERNAME/WEBSHARE_PROXY_PASSWORD, "
                "or YT_HTTP_PROXY/YT_HTTPS_PROXY) or try again later."
            )
        except Exception as e:
            print(f"An error occurred: {e}")
            raise TranscriptUnavailableError(f"Failed to download transcript: {e}")

    # Should be unreachable, but keep a defensive fallback.
    raise TranscriptUnavailableError(f"Failed to download transcript: {last_error}")

def get_stream_url(youtube_url: str) -> str:
    """Get the stream URL for video download."""
    ydl_opts = {
        'format': 'bestvideo[height<=720][ext=mp4]/best[height<=720]',
        'quiet': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        return info['url']

def create_and_save_chunks(transcript_file: str, output_chunks_file: str, chunk_size: int = 50, overlap: int = 5):
    """Chunk the transcript for processing."""
    with open(transcript_file, "r", encoding="utf-8") as f:
        transcript = json.load(f)
    
    total_segments = len(transcript)
    if total_segments > 2000:
        chunk_size = 100
        overlap = 10

    chunks = []
    step = chunk_size - overlap
    for i in range(0, total_segments, step):
        window = transcript[i : i + chunk_size]
        if window:
            chunks.append({
                "chunk_id": len(chunks) + 1, 
                "start_index": i, 
                "end_index": min(i + chunk_size, total_segments), 
                "segments": window
            })
    with open(output_chunks_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

def process_saved_chunks_with_llm(chunks_file: str, output_file: str, model_name: str = None):
    """Process chunks with LLM to find visual cues."""
    settings = get_settings()
    client = Groq(api_key=settings.GROQ_API_KEY)
    
    if model_name is None:
        model_name = settings.GROQ_MODEL
        
    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    all_detections = []
    seen_starts = set()
    total_chunks = len(chunks)
    print(f"[GROQ LLM] Analyzing {total_chunks} transcript chunks using model '{model_name}'...")
    for idx, chunk_obj in enumerate(chunks):
        chunk_id = chunk_obj["chunk_id"]
        segments = chunk_obj["segments"]
        print(f" -> [GROQ LLM] Chunk {chunk_id}/{total_chunks} ({len(segments)} transcript segments)...")
        prompt = f"""
You are an expert at identifying visual reference points in bilingual Hindi/English/Hinglish lecture transcripts.

Analyze this raw transcript segment. Identify moments where the speaker points to, references, or discusses something visually displayed on screen (PPT slide, architecture diagram, code editor, formula on board, table, or graph).

Hindi/Hinglish triggers to look for:
- "यहाँ देखो" / "yahan dekho"
- "स्क्रीन पे" / "screen pe"
- "स्लाइड में" / "slide me"
- "इस डायग्राम में" / "is diagram mein"
- "बोर्ड पे" / "board pe"
- "ये जो कोड है" / "ye jo part hai"
- English triggers: "look at this", "as shown here", "in this graph", "notice this"

Ignore purely conversational/figurative statements ("देखते हैं आगे क्या होता है", "let's see").

Input:
{json.dumps(segments, ensure_ascii=False)}

Return strictly a valid JSON object with the key "visual_points":
{{
  "visual_points": [
    {{
      "start": 124.5,
      "cue_type": "diagram | slide | code | board",
      "trigger_text": "exact phrase spoken by the instructor",
      "confidence": 0.95
    }}
  ]
}}
"""
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            data = json.loads(response.choices[0].message.content)
            detections = data.get("visual_points", [])
            print(f"    └─ Found {len(detections)} visual cues in Chunk {chunk_id}")
            for item in detections:
                start_sec = round(float(item.get("start", 0.0)), 1)
                if not any(abs(start_sec - s) < 5.0 for s in seen_starts):
                    seen_starts.add(start_sec)
                    all_detections.append(item)
        except Exception as e:
            print(f"    └─ Error on chunk {chunk_id}: {e}")
        time.sleep(1.0)
    print(f"[GROQ LLM COMPLETE] Total unique visual timestamps detected: {len(all_detections)}")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_detections, f, indent=2, ensure_ascii=False)

def extract_frames(youtube_url: str, cue_data_path: str, output_dir: str):
    """Extract frames from YouTube video."""
    os.makedirs(output_dir, exist_ok=True)
    stream_url = get_stream_url(youtube_url)
    with open(cue_data_path, "r", encoding="utf-8") as f:
        cue_data = json.load(f)
    print(f"[FFMPEG EXTRACTION] Extracting candidate frames for {len(cue_data)} timestamps...")
    for idx, item in enumerate(cue_data, start=1):
        timestamp = item["start"]
        cue_type = item.get("cue_type", "visual")
        output_filename = f"frame_{idx:02d}_{cue_type}_{int(timestamp)}s.png"
        output_path = os.path.join(output_dir, output_filename)
        print(f" -> Extracting Frame {idx}/{len(cue_data)} @ {timestamp}s ({cue_type})...")
        ffmpeg_cmd = ["ffmpeg", "-y", "-ss", str(timestamp), "-i", stream_url, "-frames:v", "1", "-q:v", "2", output_path]
        subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def deduplicate_frames(input_dir: str, output_dir: str, threshold: float = 0.85):
    """Deduplicate frames based on SSIM."""
    os.makedirs(output_dir, exist_ok=True)
    image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if not image_files:
        print("[SSIM DEDUPLICATION] No candidate frames to process.")
        return
    print(f"[SSIM DEDUPLICATION] Filtering {len(image_files)} candidate frames...")
    last_saved_img = None
    kept_count = 0
    for filename in image_files:
        file_path = os.path.join(input_dir, filename)
        current_img = cv2.imread(file_path)
        if current_img is None:
            continue
        gray_current = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)
        gray_resized = cv2.resize(gray_current, (640, 360))
        if last_saved_img is None:
            cv2.imwrite(os.path.join(output_dir, filename), current_img)
            last_saved_img = gray_resized
            kept_count += 1
            continue
        score, _ = ssim(last_saved_img, gray_resized, full=True)
        if score < threshold:
            cv2.imwrite(os.path.join(output_dir, filename), current_img)
            last_saved_img = gray_resized
            kept_count += 1
    print(f"[SSIM COMPLETE] Kept {kept_count}/{len(image_files)} unique visual frames.")
