"""Send all 20 test prompts through the live gateway server."""
import json
import time
import requests

BASE = "http://localhost:8000"

with open("test_suite.json", "r", encoding="utf-8") as f:
    prompts = json.load(f)

print(f"Sending {len(prompts)} prompts to {BASE}/chat ...\n")

for i, item in enumerate(prompts, 1):
    prompt = item["prompt"]
    gt = item["ground_truth"]
    snippet = prompt[:70] + ("..." if len(prompt) > 70 else "")

    print(f"[{i:>2}/{len(prompts)}] ({gt:>7}) {snippet}")

    try:
        r = requests.post(f"{BASE}/chat", json={"prompt": prompt}, timeout=60)
        if r.ok:
            d = r.json()
            model = d.get("model_label", d.get("model_used", "?"))
            reason = d.get("routing_reason", "?")[:60]
            cache = "HIT" if d.get("cache_hit") else "MISS"
            lat = d.get("latency_ms", 0)
            print(f"         -> {model} | {cache} | {lat:.0f}ms | {reason}")
        else:
            print(f"         -> ERROR {r.status_code}: {r.text[:100]}")
    except Exception as e:
        print(f"         -> FAILED: {e}")

    # Small delay to avoid rate limiting on free tier
    time.sleep(1)

    print()

print("Done! All 20 prompts sent. Now sending a duplicate to demonstrate cache hit...\n")

# Send a duplicate prompt to show cache hit
duplicate_prompt = prompts[0]["prompt"]
gt = prompts[0]["ground_truth"]
snippet = duplicate_prompt[:70] + ("..." if len(duplicate_prompt) > 70 else "")

print(f"[{21:>2}/21] ({gt:>7}) {snippet}")

try:
    r = requests.post(f"{BASE}/chat", json={"prompt": duplicate_prompt}, timeout=60)
    if r.ok:
        d = r.json()
        model = d.get("model_label", d.get("model_used", "?"))
        reason = d.get("routing_reason", "?")[:60]
        cache = "HIT" if d.get("cache_hit") else "MISS"
        lat = d.get("latency_ms", 0)
        print(f"         -> {model} | {cache} | {lat:.0f}ms | {reason}")
    else:
        print(f"         -> ERROR {r.status_code}: {r.text[:100]}")
except Exception as e:
    print(f"         -> FAILED: {e}")

print()

print("Done! All 20 prompts + 1 cache demo sent. Check dashboard at http://localhost:8501")
