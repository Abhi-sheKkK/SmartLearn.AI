from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound
)
from youtube_transcript_api.formatters import TextFormatter, SRTFormatter

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

        # 4. Format and save to disk
        filename = f"{video_id}_{transcript.language_code}.{output_format}"
        
        if output_format == "srt":
            formatter = SRTFormatter()
        else:
            formatter = TextFormatter()

        formatted_output = formatter.format_transcript(data)
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write(formatted_output)

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

VIDEO_ID = extract_id("https://www.youtube.com/watch?v=X0btK9X0Xnk&list=PLKnIA16_Rmva0dRLWEHLznSHKbFD_RJfX")
TARGET_LANGUAGE = "en"

transcript_text = download_any_transcript(VIDEO_ID, output_format="txt")
# print("\n--- Transcribed Text ---")
print(transcript_text["data"][:10])
for i in transcript_text["data"][:10]:
    print(f"{i.text}  {i.start} - {i.start + i.duration}")


# video_id = extract_id("https://www.youtube.com/watch?v=X0btK9X0Xnk&list=PLKnIA16_Rmva0dRLWEHLznSHKbFD_RJfX")
# ytt_api = YouTubeTranscriptApi()
# print(ytt_api.fetch(video_id))[:100]