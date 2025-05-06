import whisper, json, os

# 1. Load model (choose size per your GPU/CPU)
model = whisper.load_model("small")

# 2. Transcribe video file to get timestamps
result = model.transcribe(
    "testvideo.mp4",
    language="en",
    verbose=False
)

# 3. Ensure output directory exists
os.makedirs("data/transcripts", exist_ok=True)

# 4. Save full JSON (includes 'segments')
with open("data/transcripts/transcript.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print("Transcription saved to data/transcripts/transcript.json")
