import cv2, json, os

video_path = "testvideo.mp4"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

# every 5 seconds
interval = int(fps * 5)
frame_id = 0
associations = []

# load your chunks to map timestamps
with open("data/transcripts/chunks.jsonl") as f:
    chunks = [json.loads(line) for line in f]

def find_chunk(ts):
    for c in chunks:
        if c["start"] <= ts < c["end"]:
            return c["id"]
    return None

os.makedirs("data/frames", exist_ok=True)

idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    if idx % interval == 0:
        ts = idx / fps
        fname = f"data/frames/frame_{frame_id}.jpg"
        cv2.imwrite(fname, frame)
        associations.append({"frame_id": frame_id, "ts": ts, "chunk_id": find_chunk(ts)})
        frame_id += 1
    idx += 1

with open("data/frames/associations.json", "w", encoding="utf-8") as f:
    json.dump(associations, f, ensure_ascii=False, indent=2)
