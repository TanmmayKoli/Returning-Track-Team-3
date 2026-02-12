from pathlib import Path
import json

root = Path("data/raw/openneuro/ds002578")

# pick one subject you saw exists
sub = "001"
ev_json = root / f"sub-{sub}" / "eeg" / f"sub-{sub}_task-attention_events.json"
print("events.json exists?", ev_json.exists(), ev_json)

if ev_json.exists():
    obj = json.loads(ev_json.read_text())
    print("\nTop-level keys:", list(obj.keys())[:20])

    # Common places where mappings live:
    for k in ["value", "trial_type", "stim_file"]:
        if k in obj:
            print(f"\n== Column '{k}' ==")
            col = obj[k]
            print("keys:", list(col.keys())[:30])
            if "Levels" in col:
                levels = col["Levels"]
                # show a few levels
                for i, (lvl, meta) in enumerate(levels.items()):
                    if i >= 15: break
                    print(lvl, "->", meta)