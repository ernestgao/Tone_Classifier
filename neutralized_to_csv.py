import json
from pathlib import Path

import pandas as pd

rows = []
input_path = Path("artifacts/neutralizer_ruder_final/neutralized_results.jsonl")
with input_path.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        rows.append({
            "index": obj.get("index"),
            "original_text": obj.get("original_text"),
            "final_text": obj.get("final_text"),
            "status": obj.get("status"),
            "success": (obj.get("status") == "success_neutralized"),
        })

df = pd.DataFrame(rows).sort_values("index")

# 输出 CSV
df.to_csv("neutralization_table.csv", index=False, encoding="utf-8-sig")


print(df.head(10))
print(f"Loaded from: {input_path}")
print("Saved neutralization_table.csv")