from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import cv2
import numpy as np
import os
import re
from transformers import pipeline
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
TEXT_OUTPUT_FOLDER = 'text_outputs'

# Ensure required directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(TEXT_OUTPUT_FOLDER, exist_ok=True)

def extract_video_id(youtube_url):
    """Extracts the video ID from various YouTube URL formats."""
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", youtube_url)
    return match.group(1) if match else None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('video')
    threshold = float(request.form.get('threshold', 20.0))
    
    if file and file.filename:
        video_filename = file.filename
        video_path = os.path.join(UPLOAD_FOLDER, video_filename)
        output_filename = f'summarized_{video_filename}'
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)

        file.save(video_path)
        try:
            summarize_video(video_path, output_path, threshold)
            return redirect(url_for('download', filename=output_filename))
        except Exception as e:
            return f"Error processing video: {str(e)}", 500
    
    return "No file uploaded", 400

@app.route('/outputs/<filename>')
def download(filename):
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=True)

@app.route('/summarize_youtube', methods=['POST'])
def summarize_youtube():
    youtube_url = request.form.get('youtube_url', '')
    min_length = int(request.form.get('min_length', 250))
    max_length = int(request.form.get('max_length', 300))

    video_id = extract_video_id(youtube_url)

    if not video_id:
        return "Invalid YouTube URL. Please enter a valid link.", 400

    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        full_text = ' '.join([entry['text'] for entry in transcript])

        summarizer = pipeline('summarization', model="facebook/bart-large-cnn", min_length=min_length, max_length=max_length)
        
        summarized_text = []
        chunk_size = 500  # Reduce chunk size to avoid memory issues
        num_iters = len(full_text) // chunk_size + 1

        for i in range(num_iters):
            chunk = full_text[i * chunk_size:(i + 1) * chunk_size]
            if chunk.strip():  # Avoid empty summaries
                summary = summarizer(chunk, truncation=True)[0]['summary_text']
                summarized_text.append(summary)

        result_text = ' '.join(summarized_text)
        output_filename = f'summary_{video_id}.txt'
        output_path = os.path.join(TEXT_OUTPUT_FOLDER, output_filename)

        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(result_text)

        return redirect(url_for('download_text', filename=output_filename))

    except (TranscriptsDisabled, NoTranscriptFound):
        return "This video has no available transcript. Try another video.", 400
    except Exception as e:
        return f"Error occurred: {str(e)}", 500

@app.route('/text_outputs/<filename>')
def download_text(filename):
    return send_from_directory(TEXT_OUTPUT_FOLDER, filename, as_attachment=True)

def summarize_video(input_path, output_path, threshold):
    """Summarizes a video by selecting unique frames based on pixel difference threshold."""
    video = cv2.VideoCapture(input_path)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    ret, frame1 = video.read()
    prev_frame = frame1

    frame_count = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frame_count += 1

        # Reduce processing by skipping frames
        if frame_count % 5 == 0:  # Process every 5th frame
            continue

        if np.sum(np.abs(frame - prev_frame)) / np.size(frame) > threshold:
            writer.write(frame)
            prev_frame = frame

    video.release()
    writer.release()

if __name__ == '__main__':
    app.run(debug=True)
