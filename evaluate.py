import json
import time
import numpy as np
from retrieve import retrieve
from retrieve_pg import retrieve_pg

# Helper: turn "MM:SS" or "HH:MM" into seconds
def parse_timestamp(ts_str):
    parts = list(map(int, ts_str.split(":")))
    # If it's "MM:SS":
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    # If it's "HH:MM":
    elif len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    else:
        raise ValueError(f"Unexpected timestamp format: {ts_str}")

# 1) Load your gold array
with open("data/gold_tests.jsonl", encoding="utf-8") as f:
    gold = json.load(f)

# Precompute counts
num_answerable   = sum(1 for e in gold if e["answerable"])
num_unanswerable = sum(1 for e in gold if not e["answerable"])

# 2) Define methods
methods = {
    "faiss":    lambda q: retrieve(q, top_k=1, semantic=True, threshold=0.3),
    "tfidf":    lambda q: retrieve(q, top_k=1, lexical=True, lexical_method="tfidf", threshold=0.3),
    "bm25":     lambda q: retrieve(q, top_k=1, lexical=True, lexical_method="bm25", threshold=8.5),
    "pg_hnsw":  lambda q: retrieve_pg(q, top_k=1, index_type="hnsw", threshold=0.3),
    "pg_ivf":   lambda q: retrieve_pg(q, top_k=1, index_type="ivfflat", threshold=0.3),
}

# 3) Prepare storage
results = {
    name: {"correct": 0, "rejected": 0, "times": []}
    for name in methods
}

# 4) Evaluate
for entry in gold:
    q           = entry["question"]
    is_ans      = entry["answerable"]
    gold_ts     = entry["timestamp"] if is_ans else None

    for name, fn in methods.items():
        start = time.perf_counter()
        res   = fn(q)
        elapsed = time.perf_counter() - start
        results[name]["times"].append(elapsed)

        # No result → rejection
        if not res:
            if not is_ans:
                results[name]["rejected"] += 1
        else:
            pred_ts = res[0]["start"]
            if is_ans and abs(pred_ts - gold_ts) <= 30:
                results[name]["correct"] += 1

# 5) Print summary table
print(f"{'Method':<10}  Acc@±5s   Rej@   AvgLatency(s)")
for name, stats in results.items():
    acc      = stats["correct"] / num_answerable
    rej_rate = stats["rejected"] / num_unanswerable
    avg_lat  = np.mean(stats["times"])
    print(f"{name:<10}  {acc:>7.2%}   {rej_rate:>5.2%}   {avg_lat:>6.3f}")
