import re
from transformers import pipeline
from youtube_transcript_api import YouTubeTranscriptApi

def extract_video_id(youtube_url):
    """
    Extracts the video ID from various YouTube URL formats.
    """
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", youtube_url)
    if match:
        return match.group(1)
    return None

# Get user input for YouTube URL
youtube_video = input("Paste YouTube video URL: ")
video_id = extract_video_id(youtube_video)

if not video_id:
    print("Invalid YouTube URL. Please enter a valid link.")
    exit()

try:
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    result = ' '.join([entry['text'] for entry in transcript])

    summarizer = pipeline('summarization', model="facebook/bart-large-cnn")
    
    summarized_text = []
    num_iters = len(result) // 1000 + 1

    for i in range(num_iters):
        chunk = result[i * 1000:(i + 1) * 1000]
        if chunk.strip():  # Avoid passing empty text
            out = summarizer(chunk)[0]['summary_text']
            summarized_text.append(out)

    summarized_text_str = ' '.join(summarized_text)
    print("\nFinal Summarized Text:\n", summarized_text_str)

except Exception as e:
    print(f"Error occurred: {str(e)}")
