import os
import asyncio
from backend.config import get_settings
from backend.database import (
    update_session, upload_frame,
    get_completed_session_by_video_id, get_frame_url
)
from backend.app.pipeline.youtube_data import (
    extract_id,
    download_any_transcript,
    create_and_save_chunks,
    process_saved_chunks_with_llm,
    extract_frames,
    deduplicate_frames
)

settings = get_settings()


def _load_disk_cache(video_id: str, video_dir) -> dict | None:
    """
    Fall back to whatever was already extracted for this video_id on disk.
    Covers the case where the in-memory/DB session cache is unavailable
    (e.g. a server restart) but a previous run already saved the files.
    """
    unique_frames_dir = video_dir / "unique_frames"
    if not unique_frames_dir.exists():
        return None

    frame_files = sorted(
        f for f in os.listdir(unique_frames_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    )
    if not frame_files:
        return None

    transcript_text = None
    for txt_file in video_dir.glob("*.txt"):
        with open(txt_file, "r", encoding="utf-8") as f:
            transcript_text = f.read()
        break

    if not transcript_text:
        return None

    frame_urls = [
        get_frame_url(str(unique_frames_dir / fname), video_id)
        for fname in frame_files
    ]
    return {"transcript_text": transcript_text, "frame_urls": frame_urls}


async def process_video(youtube_url: str, session_id: str) -> dict:
    """Orchestrate the video processing pipeline, reusing already-extracted data per video_id."""
    try:
        print(f"\n==================================================")
        print(f"[PIPELINE START] Session ID: {session_id}")
        print(f"[PIPELINE START] YouTube URL: {youtube_url}")
        print(f"==================================================")

        # 1. Extract video ID
        update_session(session_id, {"status": "extracting_id"})
        video_id = extract_id(youtube_url)
        if video_id == "None":
            raise ValueError("Invalid YouTube URL")

        update_session(session_id, {"video_id": video_id})
        print(f"[STAGE 1/7] Extracted Video ID: {video_id}")

        video_dir = settings.STORAGE_DIR / video_id

        # Cache check #1: another session already fully processed this video_id.
        existing = get_completed_session_by_video_id(video_id)
        if existing and existing.get("transcript_text"):
            print(f"[PIPELINE CACHE HIT] Reusing saved data for video {video_id} (from session {existing.get('id')}) — skipping preprocessing.")
            frame_urls = existing.get("frame_urls", [])
            update_session(session_id, {
                "status": "completed",
                "transcript_text": existing.get("transcript_text", ""),
                "frame_urls": frame_urls,
                **({"notes_markdown": existing["notes_markdown"]} if existing.get("notes_markdown") else {}),
            })
            return {"status": "success", "frame_urls": frame_urls}

        # Cache check #2: no live session record, but the files are still on disk
        # (e.g. after a server restart) under this video's own folder.
        disk_cache = _load_disk_cache(video_id, video_dir)
        if disk_cache:
            print(f"[PIPELINE CACHE HIT] Found previously saved files on disk for video {video_id} — skipping preprocessing.")
            update_session(session_id, {"status": "completed", **disk_cache})
            return {"status": "success", "frame_urls": disk_cache["frame_urls"]}

        os.makedirs(video_dir, exist_ok=True)

        # 2. Download Transcript
        update_session(session_id, {"status": "downloading_transcript"})
        print(f"[STAGE 2/7] Downloading transcript for video {video_id}...")
        transcript_data = await asyncio.to_thread(
            download_any_transcript, video_id, str(video_dir)
        )
        if not transcript_data:
            raise ValueError("Failed to download transcript")

        lang_code = transcript_data["language_code"]
        transcript_json_path = video_dir / f"{lang_code}.json"
        print(f"[STAGE 2/7 COMPLETE] Downloaded transcript in language '{lang_code}' ({len(transcript_data.get('data', []))} segments).")

        # Read formatted text transcript into session record
        txt_filename = video_dir / f"{lang_code}.txt"
        if txt_filename.exists():
            with open(txt_filename, "r", encoding="utf-8") as f:
                full_transcript = f.read()
            update_session(session_id, {"transcript_text": full_transcript})

        # 3. Create Chunks
        update_session(session_id, {"status": "chunking_transcript"})
        chunks_file_path = video_dir / "chunks.json"
        print(f"[STAGE 3/7] Chunking transcript for LLM visual cue detection...")
        await asyncio.to_thread(
            create_and_save_chunks, str(transcript_json_path), str(chunks_file_path)
        )

        # 4. Process Chunks with LLM
        update_session(session_id, {"status": "processing_cues"})
        cues_file_path = video_dir / "cues.json"
        print(f"[STAGE 4/7] Invoking Groq LLM to detect visual cues...")
        await asyncio.to_thread(
            process_saved_chunks_with_llm, str(chunks_file_path), str(cues_file_path)
        )

        # 5. Extract Frames
        update_session(session_id, {"status": "extracting_frames"})
        frames_dir = video_dir / "extracted_frames"
        print(f"[STAGE 5/7] Running FFmpeg to sample candidate video frames...")
        await asyncio.to_thread(
            extract_frames, youtube_url, str(cues_file_path), str(frames_dir)
        )

        # 6. Deduplicate Frames
        update_session(session_id, {"status": "deduplicating_frames"})
        unique_frames_dir = video_dir / "unique_frames"
        print(f"[STAGE 6/7] Running SSIM visual deduplication on candidate frames...")
        await asyncio.to_thread(
            deduplicate_frames, str(frames_dir), str(unique_frames_dir)
        )

        # 7. Upload to Supabase / Local Storage, keyed by video_id so any future
        # session for the same video can reuse these files without re-extracting them.
        update_session(session_id, {"status": "uploading_frames"})
        print(f"[STAGE 7/7] Storing visual frames...")
        frame_urls = []
        if unique_frames_dir.exists():
            for filename in os.listdir(unique_frames_dir):
                if filename.endswith(('.png', '.jpg', '.jpeg')):
                    file_path = unique_frames_dir / filename
                    url = await asyncio.to_thread(upload_frame, str(file_path), video_id)
                    if url:
                        frame_urls.append(url)

        # 8. Complete
        update_session(session_id, {"status": "completed", "frame_urls": frame_urls})
        print(f"==================================================")
        print(f"[PIPELINE SUCCESS] Session {session_id} ready with {len(frame_urls)} visual frames!")
        print(f"==================================================\n")
        return {"status": "success", "frame_urls": frame_urls}

    except Exception as e:
        print(f"[PIPELINE ERROR] Error processing video: {e}")
        update_session(session_id, {"status": "error", "error_message": str(e)})
        return {"status": "error", "message": str(e)}
