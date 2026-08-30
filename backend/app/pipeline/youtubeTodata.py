import json
import time
from groq import Groq
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound
)
from youtube_transcript_api.formatters import TextFormatter, SRTFormatter
from dotenv import load_dotenv
import yt_dlp
import os
import subprocess
from skimage.metrics import structural_similarity as ssim
import cv2



def extract_id(url):
    """
    Extracts the video ID from a YouTube URL.

    Args:
        url (str): The YouTube URL.
    """
    if "youtu.be" in url:
        return url.split("/")[-1]
    elif "youtube.com" in url:
        query = url.split("?")[-1]
        params = query.split("&")
        for param in params:
            key, value = param.split("=")
            if key == "v":
                return value
    return "None"




def download_any_transcript(video_id: str, output_format: str = "txt") -> dict:
    """
    Downloads the first available transcript (manual or auto-generated)
    in whatever language is provided by the video.
    """
    try:

        ytt = YouTubeTranscriptApi()
        transcript_list = ytt.list(video_id)
        manual_transcripts = list(transcript_list._manually_created_transcripts.values())
        generated_transcripts = list(transcript_list._generated_transcripts.values())

        if manual_transcripts:
            transcript = manual_transcripts[0]
            print(f"Found manual transcript in: {transcript.language} ({transcript.language_code})")
        elif generated_transcripts:
            transcript = generated_transcripts[0]
            print(f"Found auto-generated transcript in: {transcript.language} ({transcript.language_code})")
        else:
            # Fallback iterator
            transcript = next(iter(transcript_list))
            print(f"Found transcript in: {transcript.language} ({transcript.language_code})")

        data = transcript.fetch()
        transcript_dict = [item.__dict__ for item in data]

        filename = f"backend/storage/hlGoQC332VM_hi/{transcript.language_code}.{output_format}"
        if output_format == "srt":
            formatter = SRTFormatter()
        else:
            formatter = TextFormatter()

        formatted_output = formatter.format_transcript(data)
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write(formatted_output)

        with open(f"{filename[:-4]}.json", "w", encoding="utf-8") as f:
            json.dump(transcript_dict, f, indent=4, ensure_ascii=False)
        
        print(f"Saved transcript successfully to {filename}")
        return {
            "language": transcript.language,
            "language_code": transcript.language_code,
            "is_generated": transcript.is_generated,
            "data": data
        }

    except TranscriptsDisabled:
        print(f"Captions are disabled on video: {video_id}")
    except NoTranscriptFound:
        print(f"No transcripts found for video: {video_id}")
    except Exception as e:
        print(f"An error occurred: {e}")


def get_stream_url(youtube_url: str) -> str:
    """Extracts the direct 720p/1080p stream URL without downloading."""
    ydl_opts = {
        'format': 'bestvideo[height<=720][ext=mp4]/best[height<=720]',
        'quiet': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        return info['url']



# VIDEO_ID = extract_id("https://www.youtube.com/watch?v=hlGoQC332VM")
# TARGET_LANGUAGE = "en"

# transcript_text = download_any_transcript(VIDEO_ID, output_format="txt")
# print("\n--- Transcribed Text ---")
# print(transcript_text["data"][:10])
# for i in transcript_text["data"][:10]:
#     print(f"{i.text} -- {i.start}")





def create_and_save_chunks(
    transcript_file: str,
    output_chunks_file: str,
    chunk_size: int = 50,
    overlap: int = 5,
):
    """Chunks the 4-5 hour transcript list, runs LLM extraction, and saves deduplicated timestamps."""

    with open(transcript_file, "r", encoding="utf-8") as f:
        transcript = json.load(f)

    total_segments = len(transcript)
    print(f"Loaded {total_segments} transcript segments.")
    chunks = []
    step = chunk_size - overlap

    for i in range(0, total_segments, step):
        window = transcript[i : i + chunk_size]
        if window:
            chunks.append(
                {
                    "chunk_id": len(chunks) + 1,
                    "start_index": i,
                    "end_index": min(i + chunk_size, total_segments),
                    "segments": window,
                }
            )

    with open(output_chunks_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(
        f"Saved {len(chunks)} chunks from {total_segments} segments to '{output_chunks_file}'."
    )

def process_saved_chunks_with_llm(
    chunks_file: str ,
    output_file: str ,
    model_name: str = "qwen/qwen3.8-27b",
):

    """Loads saved chunks from file, passes each to the LLM,

    and writes deduplicated visual timestamps to the output file.
    """

    load_dotenv(r"/Users/shubham/github_work/smartlearn_ai/.env")
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    total_chunks = len(chunks)
    print(f"Loaded {total_chunks} chunks to process with LLM.")

    all_detections = []
    seen_starts = set()

    for idx, chunk_obj in enumerate(chunks):
        chunk_id = chunk_obj["chunk_id"]
        segments = chunk_obj["segments"]

        print(f"Processing chunk {chunk_id}/{total_chunks}...")

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

            for item in detections:
                start_sec = round(float(item.get("start", 0.0)), 1)
                # Deduplicate points within a 5-second window across chunk overlaps
                if not any(abs(start_sec - s) < 5.0 for s in seen_starts):
                    seen_starts.add(start_sec)
                    all_detections.append(item)

        except Exception as e:
            print(f"Error on chunk {chunk_id}: {e}")

        # Small pause to comfortably clear free-tier RPM limits
        time.sleep(1.5)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_detections, f, indent=2, ensure_ascii=False)

    print(
        f"\nExtraction complete! Saved {len(all_detections)} visual cues to '{output_file}'."
    )
    
def extract_frames(youtube_url, cue_data_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    print("Resolving YouTube stream URL...")
    stream_url = get_stream_url(youtube_url)
    
    with open(cue_data_path, "r", encoding="utf-8") as f:
        cue_data = json.load(f)

    for idx, item in enumerate(cue_data, start=1):
        timestamp = item["start"]
        cue_type = item.get("cue_type", "visual")
        
        output_filename = f"frame_{idx:02d}_{cue_type}_{int(timestamp)}s.png"
        output_path = os.path.join(output_dir, output_filename)
        
        print(f"Extracting frame at {timestamp}s -> {output_filename}...")
        
        # -ss placed before -i performs fast stream seeking
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-ss", str(timestamp),
            "-i", stream_url,
            "-frames:v", "1",
            "-q:v", "2",
            output_path
        ]
        
        subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    print(f"\nCompleted! Frames saved to: {output_dir}")



def deduplicate_frames(input_dir, output_dir, threshold=0.85):
    """
    threshold: float between 0 and 1.0.
      - 0.85 to 0.90 is ideal for presentation slides / lecture boards.
      - Similarity >= threshold means it's considered a duplicate and skipped.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files sorted chronologically/alphabetically
    image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    if not image_files:
        print("No images found in the directory.")
        return

    last_saved_img = None
    kept_count = 0

    for filename in image_files:
        file_path = os.path.join(input_dir, filename)
        current_img = cv2.imread(file_path)
        
        if current_img is None:
            continue

        # Convert to grayscale and resize for fast, robust comparison
        gray_current = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)
        gray_resized = cv2.resize(gray_current, (640, 360))

        if last_saved_img is None:
            # Always keep the very first frame
            cv2.imwrite(os.path.join(output_dir, filename), current_img)
            last_saved_img = gray_resized
            kept_count += 1
            continue

        # Compute similarity against the last kept frame
        score, _ = ssim(last_saved_img, gray_resized, full=True)

        # If similarity is lower than threshold, it's a new visual
        if score < threshold:
            cv2.imwrite(os.path.join(output_dir, filename), current_img)
            last_saved_img = gray_resized
            kept_count += 1

    print(f"Deduplication complete: Reduced {len(image_files)} frames -> {kept_count} unique slides.")

deduplicate_frames(input_dir="extracted_frames", output_dir="unique_frames", threshold=0.40)