import cv2
import yt_dlp
from ultralytics import YOLO

youtube_url = "https://www.youtube.com/watch?v=6dp-bvQ7RWo"

model = YOLO("models/UVH-26-MV-YOLOv11-S.pt")


def get_stream_url():
    ydl_opts = {
        "format": "232",
        "quiet": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)

    return info["url"]


# Get the current live HLS stream URL
url = get_stream_url()

print("Opening live stream...")
video_path="traffic_2.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

print("Stream opened successfully.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Stream ended or failed to read frame.")
        break

    # Run YOLO
    results = model(frame, stream=True)

    # Draw detections
    annotated_frame = frame

    for result in results:
        annotated_frame = result.plot()

    # Display
    cv2.imshow("YouTube Live inference", annotated_frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()