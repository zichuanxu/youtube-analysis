import csv
import googleapiclient.discovery
import googleapiclient.errors
from datetime import datetime
from urllib.parse import urlparse, parse_qs
import os
import re
import requests
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import cv2
import pytesseract
from collections import Counter
import json
from config import API_KEY, CHANNEL_ID, URL_INPUT_FILE, OUTPUT_CSV_FILE, CASCADE_FILE_PATH

# --------------------------------------------------------------------------
# Helper Functions
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

def detect_text_in_image(image):
    """Detects if there's text in the thumbnail image using OCR."""
    try:
        # Use pytesseract to detect text
        text = pytesseract.image_to_string(image, config='--psm 8')
        return 1 if text.strip() else 0
    except:
        return 0

def detect_graphics_effects(cv_image):
    """Detects graphic effects like borders, gradients, or overlays."""
    try:
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # Detect edges (potential graphics/borders)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = np.sum(edges > 0) / edges.size

        # Check for high saturation areas (often indicates graphics)
        high_sat = np.sum(hsv[:,:,1] > 200) / hsv[:,:,1].size

        # Simple heuristic: if there are many edges or high saturation areas
        return 1 if (edge_ratio > 0.1 or high_sat > 0.3) else 0
    except:
        return 0

def extract_rgb_values(cv_image):
    """Extract average RGB values from the thumbnail."""
    try:
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        # Calculate mean RGB values
        r_mean = round(np.mean(rgb_image[:,:,0]), 2)
        g_mean = round(np.mean(rgb_image[:,:,1]), 2)
        b_mean = round(np.mean(rgb_image[:,:,2]), 2)

        return r_mean, g_mean, b_mean
    except:
        return 'N/A', 'N/A', 'N/A'

def detect_objects_basic(cv_image):
    """Basic object detection using color and shape analysis."""
    try:
        # Convert to HSV for better object detection
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        detected_objects = []

        # Detect common colors that might indicate objects
        # Red objects
        red_mask1 = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
        red_mask2 = cv2.inRange(hsv, (170, 50, 50), (180, 255, 255))
        red_mask = red_mask1 + red_mask2
        if np.sum(red_mask) > 1000:  # Threshold for significant red area
            detected_objects.append("red_object")

        # Blue objects
        blue_mask = cv2.inRange(hsv, (100, 50, 50), (130, 255, 255))
        if np.sum(blue_mask) > 1000:
            detected_objects.append("blue_object")

        # Green objects
        green_mask = cv2.inRange(hsv, (40, 50, 50), (80, 255, 255))
        if np.sum(green_mask) > 1000:
            detected_objects.append("green_object")

        # Yellow objects
        yellow_mask = cv2.inRange(hsv, (20, 50, 50), (40, 255, 255))
        if np.sum(yellow_mask) > 1000:
            detected_objects.append("yellow_object")

        return detected_objects
    except:
        return []

def analyze_thumbnail_advanced(image_url, face_cascade):
    """
    Enhanced thumbnail analysis including all research-based features.
    """
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()

        pil_image = Image.open(io.BytesIO(response.content))
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # Original features
        gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        brightness = round(gray_image.mean(), 2)

        # Colorfulness
        (B, G, R) = cv2.split(cv_image.astype("float"))
        rg = np.absolute(R - G)
        yb = np.absolute(0.5 * (R + G) - B)
        (rbMean, rbStd) = (np.mean(rg), np.std(rg))
        (ybMean, ybStd) = (np.mean(yb), np.std(yb))
        stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
        meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))
        colorfulness = round(stdRoot + (0.3 * meanRoot), 2)

        # Person detection
        faces = face_cascade.detectMultiScale(
            gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        person_present = 1 if len(faces) > 0 else 0

        # NEW FEATURES based on research

        # RGB values
        r_mean, g_mean, b_mean = extract_rgb_values(cv_image)

        # Text detection
        text_present = detect_text_in_image(pil_image)

        # Graphics effects detection
        graphics_present = detect_graphics_effects(cv_image)

        # Basic object detection
        detected_objects = detect_objects_basic(cv_image)

        # Create image tags summary
        image_tags = []
        if person_present:
            image_tags.append("person")
        if text_present:
            image_tags.append("text")
        if graphics_present:
            image_tags.append("graphics")
        image_tags.extend(detected_objects)

        # Get top 5 most confident tags (simplified for basic detection)
        category_tags = image_tags[:5] if len(image_tags) >= 5 else image_tags

        return {
            'brightness': brightness,
            'colorfulness': colorfulness,
            'person': person_present,
            'r_mean': r_mean,
            'g_mean': g_mean,
            'b_mean': b_mean,
            'text': text_present,
            'graphics': graphics_present,
            'image_tags': ', '.join(image_tags) if image_tags else 'none',
            'category_tags': ', '.join(category_tags) if category_tags else 'none'
        }

    except Exception as e:
        print(f"Error analyzing thumbnail {image_url}: {e}")
        return {
            'brightness': 'N/A',
            'colorfulness': 'N/A',
            'person': 0,
            'r_mean': 'N/A',
            'g_mean': 'N/A',
            'b_mean': 'N/A',
            'text': 0,
            'graphics': 0,
            'image_tags': 'N/A',
            'category_tags': 'N/A'
        }

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

def get_channel_statistics(youtube, channel_id):
    """Get channel-level statistics for brand/channel features."""
    try:
        channel_response = youtube.channels().list(
            part='statistics,snippet',
            id=channel_id
        ).execute()

        if channel_response.get("items"):
            channel_data = channel_response['items'][0]
            stats = channel_data.get('statistics', {})
            snippet = channel_data.get('snippet', {})

            return {
                'subscriber_count': int(stats.get('subscriberCount', 0)),
                'total_video_count': int(stats.get('videoCount', 0)),
                'total_view_count': int(stats.get('viewCount', 0)),
                'channel_created': snippet.get('publishedAt', ''),
                'channel_title': snippet.get('title', '')
            }
    except Exception as e:
        print(f"Error fetching channel statistics: {e}")

    return {
        'subscriber_count': 0,
        'total_video_count': 0,
        'total_view_count': 0,
        'channel_created': '',
        'channel_title': ''
    }

def calculate_channel_metrics(channel_stats, video_count):
    """Calculate derived channel metrics."""
    try:
        total_views = channel_stats.get('total_view_count', 0)
        total_videos = channel_stats.get('total_video_count', 1)  # Avoid division by zero

        avg_views_per_video = round(total_views / total_videos, 2) if total_videos > 0 else 0

        return {
            'avg_views_per_video': avg_views_per_video,
            'total_channel_views': total_views
        }
    except:
        return {
            'avg_views_per_video': 0,
            'total_channel_views': 0
        }

def analyze_title_features(title):
    """Analyze title characteristics that might influence views."""
    try:
        features = {
            'title_length': len(title),
            'title_word_count': len(title.split()),
            'has_numbers': 1 if re.search(r'\d', title) else 0,
            'has_caps': 1 if any(c.isupper() for c in title) else 0,
            'has_question': 1 if '?' in title else 0,
            'has_exclamation': 1 if '!' in title else 0,
            'has_brackets': 1 if any(char in title for char in '()[]{}') else 0,
        }

        # Common clickbait words
        clickbait_words = ['amazing', 'shocking', 'unbelievable', 'secret', 'hack', 'trick',
                          'you won\'t believe', 'must see', 'incredible', 'insane', 'epic']
        features['clickbait_score'] = sum(1 for word in clickbait_words if word.lower() in title.lower())

        return features
    except:
        return {
            'title_length': 0, 'title_word_count': 0, 'has_numbers': 0,
            'has_caps': 0, 'has_question': 0, 'has_exclamation': 0,
            'has_brackets': 0, 'clickbait_score': 0
        }

def get_video_details(youtube, video_ids, face_cascade, channel_id=None):
    """Enhanced video details extraction with research-based features."""
    if not video_ids: return None

    # Get channel statistics once for all videos
    channel_stats = {}
    if channel_id:
        channel_stats = get_channel_statistics(youtube, channel_id)
        print(f"Channel stats: {channel_stats['subscriber_count']} subscribers, {channel_stats['total_video_count']} videos")

    video_details_list = []
    total_videos = len(video_ids)
    print(f"Fetching enhanced video details (Total: {total_videos})...")

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

                # Basic video info
                duration_sec = parse_iso8601_duration(content_details.get("duration", "PT0S"))
                length_category = categorize_video_length(duration_sec)

                # Title analysis
                title = snippet.get("title", "")
                title_features = analyze_title_features(title)

                # Enhanced thumbnail analysis
                thumbnail_url = snippet.get("thumbnails", {}).get("high", {}).get("url")
                thumbnail_analysis = analyze_thumbnail_advanced(thumbnail_url, face_cascade)

                # Channel metrics
                channel_metrics = calculate_channel_metrics(channel_stats, len(video_ids))

                # Tags processing
                tags_list = snippet.get("tags", [])
                tags_str = ", ".join(tags_list)
                tag_count = len(tags_list)

                # Date processing
                published_at_utc = datetime.fromisoformat(snippet.get("publishedAt").replace("Z", "+00:00"))
                published_at_jst = published_at_utc.astimezone(datetime.now().astimezone().tzinfo)

                # Compile all data
                video_details_list.append([
                    # Basic info
                    item.get("id"),
                    title,
                    snippet.get("channelTitle"),
                    published_at_jst.strftime('%Y-%m-%d %H:%M:%S'),

                    # Statistics
                    int(statistics.get("viewCount", 0)),
                    int(statistics.get("likeCount", 0)),
                    int(statistics.get("commentCount", 0)),

                    # Video characteristics
                    duration_sec,
                    length_category,

                    # Original thumbnail features
                    thumbnail_analysis['brightness'],
                    thumbnail_analysis['colorfulness'],
                    thumbnail_analysis['person'],

                    # NEW: RGB values
                    thumbnail_analysis['r_mean'],
                    thumbnail_analysis['g_mean'],
                    thumbnail_analysis['b_mean'],

                    # NEW: Advanced thumbnail features
                    thumbnail_analysis['text'],
                    thumbnail_analysis['graphics'],
                    thumbnail_analysis['image_tags'],
                    thumbnail_analysis['category_tags'],

                    # NEW: Title features
                    title_features['title_length'],
                    title_features['title_word_count'],
                    title_features['has_numbers'],
                    title_features['has_caps'],
                    title_features['has_question'],
                    title_features['has_exclamation'],
                    title_features['has_brackets'],
                    title_features['clickbait_score'],

                    # NEW: Channel/Brand features
                    channel_stats.get('subscriber_count', 0),
                    channel_stats.get('total_video_count', 0),
                    channel_metrics['avg_views_per_video'],
                    channel_metrics['total_channel_views'],

                    # Tags
                    tag_count,
                    tags_str,

                    # URL
                    thumbnail_url
                ])
        except googleapiclient.errors.HttpError as e:
            print(f"An error occurred during an API request: {e}")
            return video_details_list if video_details_list else None
    return video_details_list

def save_to_csv(data, filename):
    """Saves the enhanced collected data to a CSV file."""
    header = [
        # Basic info
        "Video ID", "Title", "Channel Name", "Published At (JST)",

        # Statistics
        "View Count", "Like Count", "Comment Count",

        # Video characteristics
        "Duration (sec)", "Video Length Category",

        # Original thumbnail features
        "Thumbnail Brightness", "Thumbnail Colorfulness", "Person in Thumbnail",

        # RGB values (research-based)
        "Thumbnail R", "Thumbnail G", "Thumbnail B",

        # Advanced thumbnail features (research-based)
        "Text in Thumbnail", "Graphics in Thumbnail", "Image Tags", "Category Tags",

        # Title features (research-based)
        "Title Length", "Title Word Count", "Has Numbers", "Has Caps",
        "Has Question", "Has Exclamation", "Has Brackets", "Clickbait Score",

        # Channel/Brand features (research-based)
        "Channel Subscribers", "Channel Video Count", "Channel Avg Views", "Channel Total Views",

        # Tags
        "Tag Count", "Tags",

        # URL
        "Thumbnail URL"
    ]
    try:
        with open(filename, 'w', newline='', encoding='utf-8-sig') as file:
            writer = csv.writer(file)
            writer.writerow(header)
            if data:
                writer.writerows(data)
        print(f"✅ Enhanced data successfully saved to '{filename}' with {len(header)} features.")
        print(f"📊 New features added: RGB values, text/graphics detection, title analysis, channel metrics")
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
    if API_KEY == "":
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
                    video_data = get_video_details(youtube, video_ids, face_cascade, CHANNEL_ID)
                    if video_data:
                        print(f"📊 Fetched and analyzed enhanced data for {len(video_data)} videos. Saving to CSV file...")
                        save_to_csv(video_data, OUTPUT_CSV_FILE)
                    else:
                        print("❌ Failed to fetch data.")
                else:
                    print("❌ No videos to process. Please check the Channel ID or the URL file.")

            except googleapiclient.errors.HttpError as e:
                print(f"An error occurred while connecting to the API: {e}")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
