import json

# Load Whisper output
with open("data/transcripts/transcript.json", "r", encoding="utf-8") as f:
    whisper_out = json.load(f)

segments = whisper_out["segments"]  # list of {'start','end','text',…}

# Merge into 20s windows
chunk_duration = 20
chunks = []
current = {"start": segments[0]["start"], "end": None, "text": ""}
for seg in segments:
    if seg["start"] - current["start"] >= chunk_duration:
        # finalize previous chunk
        current["end"] = prev_end
        chunks.append(current.copy())
        current = {"start": seg["start"], "end": None, "text": ""}
    current["text"] += " " + seg["text"]
    prev_end = seg["end"]

current["end"] = prev_end
chunks.append(current)

#Save your chunks
with open("data/transcripts/chunks.jsonl", "w", encoding="utf-8") as out:
    for i, c in enumerate(chunks):
        c["id"] = i
        out.write(json.dumps(c, ensure_ascii=False) + "\n")
