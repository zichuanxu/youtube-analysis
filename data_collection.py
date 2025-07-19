import csv
import googleapiclient.discovery
import googleapiclient.errors
from datetime import datetime
from urllib.parse import urlparse, parse_qs
import os
import re
import requests
import numpy as np
from PIL import Image
import io
import cv2
from config import API_KEY, CHANNEL_ID, URL_INPUT_FILE, OUTPUT_CSV_FILE, CASCADE_FILE_PATH

# --------------------------------------------------------------------------
# Helper Functions (New/Improved)
# --------------------------------------------------------------------------

def parse_iso8601_duration(duration_str):
    """Converts an ISO 8601 duration string (e.g., PT1M13S) to seconds."""
    # Fix for unexpected formats like durations not starting with PT (e.g., P0D).
    match = re.match(r'PT((?P<hours>\d+)H)?((?P<minutes>\d+)M)?((?P<seconds>\d+)S)?', duration_str)

    # Prevent errors if re.match returns None (i.e., the format doesn't match).
    if not match:
        # print(f"Warning: Unknown duration format: '{duration_str}'. Treating as 0 seconds.")
        return 0

    # If the format matches, continue with the previous process.
    parts = match.groupdict()
    hours = int(parts['hours']) if parts['hours'] else 0
    minutes = int(parts['minutes']) if parts['minutes'] else 0
    seconds = int(parts['seconds']) if parts['seconds'] else 0
    return hours * 3600 + minutes * 60 + seconds

def categorize_video_length(seconds):
    """Categorizes the video length."""
    if seconds <= 60:
        return "Shorts (<= 1 min)"
    elif seconds <= 300:
        return "Short (1-5 min)"
    elif seconds <= 1200:
        return "Medium (5-20 min)"
    else:
        return "Long (> 20 min)"

def analyze_thumbnail(image_url, face_cascade):
    """
    Analyzes the thumbnail image and returns its brightness, colorfulness,
    and whether a person is present.
    """
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()

        pil_image = Image.open(io.BytesIO(response.content))
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # 1. Brightness
        gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        brightness = round(gray_image.mean(), 2)

        # 2. Colorfulness
        (B, G, R) = cv2.split(cv_image.astype("float"))
        rg = np.absolute(R - G)
        yb = np.absolute(0.5 * (R + G) - B)
        (rbMean, rbStd) = (np.mean(rg), np.std(rg))
        (ybMean, ybStd) = (np.mean(yb), np.std(yb))
        stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
        meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))
        colorfulness = round(stdRoot + (0.3 * meanRoot), 2)

        # 3. Person Detection
        faces = face_cascade.detectMultiScale(
            gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        person_present = "Yes" if len(faces) > 0 else "No"

        return brightness, colorfulness, person_present

    except Exception as e:
        return 'N/A', 'N/A', 'N/A'

def load_face_cascade(file_path):
    """Loads or downloads the cascade classifier for face detection."""
    cascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    if not os.path.exists(file_path):
        print(f"'{file_path}' not found. Downloading from GitHub...")
        try:
            r = requests.get(cascade_url, allow_redirects=True)
            r.raise_for_status()
            with open(file_path, 'wb') as f:
                f.write(r.content)
            print("Download complete.")
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return None
    return cv2.CascadeClassifier(file_path)

# --------------------------------------------------------------------------
# Main Data Collection and Processing Functions
# --------------------------------------------------------------------------

def get_video_ids_from_channel(youtube, channel_id):
    """Gets all video IDs from the specified channel ID."""
    print(f"Fetching video list from Channel ID: {channel_id}...")
    try:
        channel_response = youtube.channels().list(part='contentDetails', id=channel_id).execute()
        if not channel_response.get("items"):
            print(f"Error: Channel ID '{channel_id}' not found.")
            return []
        uploads_playlist_id = channel_response['items'][0]['contentDetails']['relatedPlaylists']['uploads']

        video_ids = []
        next_page_token = None
        while True:
            playlist_response = youtube.playlistItems().list(
                part='contentDetails', playlistId=uploads_playlist_id, maxResults=50, pageToken=next_page_token
            ).execute()
            video_ids.extend([item['contentDetails']['videoId'] for item in playlist_response['items']])
            next_page_token = playlist_response.get('nextPageToken')
            if not next_page_token: break

        print(f"Found a total of {len(video_ids)} videos.")
        return video_ids
    except googleapiclient.errors.HttpError as e:
        print(f"An error occurred during an API request: {e}")
        return []

def get_video_details(youtube, video_ids, face_cascade):
    """Uses the YouTube Data API to get detailed information and thumbnail analysis results for a list of video IDs."""
    if not video_ids: return None

    video_details_list = []
    total_videos = len(video_ids)
    print(f"Fetching video details (Total: {total_videos})...")

    for i in range(0, total_videos, 50):
        chunk_ids = video_ids[i:i+50]
        print(f"  Processing: {i+1} - {min(i+50, total_videos)} / {total_videos}")
        request = youtube.videos().list(
            part="snippet,statistics,contentDetails", id=",".join(chunk_ids)
        )
        try:
            response = request.execute()
            for item in response.get("items", []):
                snippet = item.get("snippet", {})
                statistics = item.get("statistics", {})
                content_details = item.get("contentDetails", {})

                duration_sec = parse_iso8601_duration(content_details.get("duration", "PT0S"))
                length_category = categorize_video_length(duration_sec)

                thumbnail_url = snippet.get("thumbnails", {}).get("high", {}).get("url")
                brightness, colorfulness, person = analyze_thumbnail(thumbnail_url, face_cascade)

                tags_str = ", ".join(snippet.get("tags", []))
                published_at_utc = datetime.fromisoformat(snippet.get("publishedAt").replace("Z", "+00:00"))
                published_at_jst = published_at_utc.astimezone(datetime.now().astimezone().tzinfo)

                video_details_list.append([
                    item.get("id"),
                    snippet.get("title"),
                    snippet.get("channelTitle"),
                    published_at_jst.strftime('%Y-%m-%d %H:%M:%S'),
                    statistics.get("viewCount", 0),
                    statistics.get("likeCount", 0),
                    statistics.get("commentCount", 0),
                    duration_sec,
                    length_category,
                    brightness,
                    colorfulness,
                    person,
                    tags_str,
                    thumbnail_url
                ])
        except googleapiclient.errors.HttpError as e:
            print(f"An error occurred during an API request: {e}")
            return video_details_list if video_details_list else None
    return video_details_list

def save_to_csv(data, filename):
    """Saves the collected data to a CSV file."""
    header = [
        "Video ID", "Title", "Channel Name", "Published At (JST)",
        "View Count", "Like Count", "Comment Count",
        "Duration (sec)", "Video Length Category", "Thumbnail Brightness", "Thumbnail Colorfulness", "Person in Thumbnail",
        "Tags", "Thumbnail URL"
    ]
    try:
        with open(filename, 'w', newline='', encoding='utf-8-sig') as file:
            writer = csv.writer(file)
            writer.writerow(header)
            if data:
                writer.writerows(data)
        print(f"✅ Data successfully saved to '{filename}'.")
    except IOError as e:
        print(f"❌ An error occurred while writing to the file: {e}")

# --- (Utility Functions: Reused from previous version) ---
def extract_video_id(url):
    parsed_url = urlparse(url)
    if parsed_url.hostname in ['www.youtube.com', 'youtube.com']:
        if parsed_url.path == '/watch': return parse_qs(parsed_url.query).get('v', [None])[0]
    elif parsed_url.hostname == 'youtu.be': return parsed_url.path[1:]
    return None

def read_urls_from_file(filename):
    if not os.path.exists(filename):
        print(f"File '{filename}' not found. Creating a sample file.")
        with open(filename, 'w') as f: f.write("https://www.youtube.com/watch?v=M7lc1UVf-VE\n")
        return []
    with open(filename, 'r') as file:
        return [line.strip() for line in file if line.strip() and not line.startswith('#')]

# --- Program Execution ---
if __name__ == "__main__":
    if API_KEY == "YOUR_API_KEY":
        print("⚠️ Warning: API key is not set. Please replace 'YOUR_API_KEY' in the code with your actual key.")
    else:
        face_cascade = load_face_cascade(CASCADE_FILE_PATH)
        if not face_cascade:
            print("❌ Could not load the face detection model. Exiting the program.")
        else:
            try:
                youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=API_KEY)

                video_ids = []
                if CHANNEL_ID:
                    video_ids = get_video_ids_from_channel(youtube, CHANNEL_ID)
                else:
                    print(f"📄 Reading URLs from '{URL_INPUT_FILE}'...")
                    urls = read_urls_from_file(URL_INPUT_FILE)
                    if urls:
                        video_ids = [extract_video_id(url) for url in urls if extract_video_id(url)]
                        print(f"🔍 Extracted {len(video_ids)} valid video IDs.")

                if video_ids:
                    video_data = get_video_details(youtube, video_ids, face_cascade)
                    if video_data:
                        print(f"📊 Fetched and analyzed data for {len(video_data)} videos. Saving to CSV file...")
                        save_to_csv(video_data, OUTPUT_CSV_FILE)
                    else:
                        print("❌ Failed to fetch data.")
                else:
                    print("❌ No videos to process. Please check the Channel ID or the URL file.")

            except googleapiclient.errors.HttpError as e:
                print(f"An error occurred while connecting to the API: {e}")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
