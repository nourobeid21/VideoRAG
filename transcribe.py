import whisper, json, os


model = whisper.load_model("small")


result = model.transcribe(
    "testvideo.mp4",
    language="en",
    verbose=False
)


os.makedirs("data/transcripts", exist_ok=True)

with open("data/transcripts/transcript.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print("Transcription saved to data/transcripts/transcript.json")
