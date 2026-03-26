"""
RouteWise — Log Viewer Dashboard
==================================
A Streamlit app that reads the gateway's in-memory logs and displays
them in a table for judges to inspect routing decisions during the demo.

Columns
-------
Timestamp | Prompt Snippet | Model Used | Routing Reason | Confidence |
Latency (ms) | Cost (USD) | Cache Hit

Usage
-----
    streamlit run dashboard/app.py

The dashboard polls the gateway's /logs endpoint every 3 seconds.
"""

from __future__ import annotations

import requests
import pandas as pd
import streamlit as st
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GATEWAY_URL = "http://localhost:8000"
POLL_INTERVAL = 3  # seconds

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="RouteWise — Log Viewer",
    page_icon="🔀",
    layout="wide",
)

st.title("🔀 RouteWise — AI Gateway Log Viewer")
st.caption("Real-time request log from the RouteWise AI Gateway. Auto-refreshes every 3 seconds.")

# ---------------------------------------------------------------------------
# Fetch logs (defined early so sidebar and main section can both use it)
# ---------------------------------------------------------------------------

def fetch_logs(url: str, limit: int) -> list[dict]:
    """Fetch logs from the gateway."""
    try:
        resp = requests.get(f"{url}/logs", params={"limit": limit}, timeout=5)
        if resp.ok:
            return resp.json().get("logs", [])
    except Exception:
        pass
    return []


# ---------------------------------------------------------------------------
# Sidebar — controls
# ---------------------------------------------------------------------------

# Fetch logs once before the sidebar renders (max_rows will be used on subsequent reruns after user adjusts slider)
DEFAULT_LIMIT = 100
logs = fetch_logs(GATEWAY_URL, DEFAULT_LIMIT)

with st.sidebar:
    st.header("⚙️ Controls")
    gateway_url = st.text_input("Gateway URL", value=GATEWAY_URL)
    max_rows = st.slider("Max rows to display", 10, 500, DEFAULT_LIMIT)
    auto_refresh = st.checkbox("Auto-refresh (3s)", value=True)

    # Refetch with user's chosen settings
    logs = fetch_logs(gateway_url, max_rows)

    st.divider()

    # Cache stats — computed from fetched logs so they always match the table
    st.header("📊 Cache Stats")
    if logs:
        total = len(logs)
        # Only count explicit cache_hit=True as hits; None/missing/False all count as miss
        cache_hits_sidebar = sum(1 for l in logs if l.get("cache_hit") is True)
        cache_misses_sidebar = total - cache_hits_sidebar
        hit_rate_sidebar = cache_hits_sidebar / total if total > 0 else 0.0
        col1, col2 = st.columns(2)
        col1.metric("Hit Rate", f"{hit_rate_sidebar:.1%}")
        col2.metric("Entries", total)
        col1.metric("Hits", cache_hits_sidebar)
        col2.metric("Misses", cache_misses_sidebar)
        st.caption(f"Threshold: {logs[0].get('cache_similarity', 'N/A')}")
    else:
        st.caption("No data yet — send requests to the gateway.")

    st.divider()

    # Health check
    st.header("🏥 Gateway Health")
    try:
        health_resp = requests.get(f"{gateway_url}/health", timeout=2)
        if health_resp.ok:
            health = health_resp.json()
            st.success(f"Status: {health.get('status', 'unknown')}")
            st.caption(f"Fast: {health.get('models', {}).get('fast', 'N/A')}")
            st.caption(f"Capable: {health.get('models', {}).get('capable', 'N/A')}")
            st.caption(f"Total requests: {health.get('total_requests', 0)}")
        else:
            st.error("Gateway returned non-OK status")
    except Exception:
        st.error("Gateway is not reachable")

# ---------------------------------------------------------------------------
# Main display
# ---------------------------------------------------------------------------

if not logs:
    st.info(
        "No logs yet. Send a request to the gateway:\n\n"
        '```bash\n'
        'curl -X POST http://localhost:8000/chat \\\n'
        '  -H "Content-Type: application/json" \\\n'
        '  -d \'{"prompt": "What is the capital of France?"}\'\n'
        '```'
    )
else:
    # Build DataFrame
    df = pd.DataFrame(logs)

    # Select and rename columns for display
    display_columns = {
        "timestamp": "Timestamp",
        "prompt_snippet": "Prompt",
        "model_used": "Model",
        "routing_reason": "Routing Reason",
        "confidence": "Confidence",
        "latency_ms": "Latency (ms)",
        "cost_usd": "Cost (USD)",
        "cache_hit": "Cache Hit",
    }

    # Only include columns that exist
    available = [c for c in display_columns if c in df.columns]
    df_display = df[available].rename(columns=display_columns)

    # Format timestamp
    if "Timestamp" in df_display.columns:
        try:
            df_display["Timestamp"] = pd.to_datetime(df_display["Timestamp"]).dt.strftime("%H:%M:%S")
        except Exception:
            pass

    # Format confidence
    if "Confidence" in df_display.columns:
        df_display["Confidence"] = df_display["Confidence"].apply(
            lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x
        )

    # Format cost
    if "Cost (USD)" in df_display.columns:
        df_display["Cost (USD)"] = df_display["Cost (USD)"].apply(
            lambda x: f"${x:.6f}" if isinstance(x, (int, float)) else x
        )

    # Format cache hit
    if "Cache Hit" in df_display.columns:
        df_display["Cache Hit"] = df_display["Cache Hit"].apply(
            lambda x: "✅ HIT" if x is True else ("⚠️ N/A" if x is None else "❌ MISS")
        )

    # Summary metrics
    total = len(df_display)
    cache_hits = sum(1 for l in logs if l.get("cache_hit", False))
    fast_count = sum(1 for l in logs if "llama" in str(l.get("model_used", "")).lower() or "groq" in str(l.get("model_used", "")).lower())
    capable_count = sum(1 for l in logs if "gemini" in str(l.get("model_used", "")).lower())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Requests", total)
    col2.metric("→ Fast Model", fast_count)
    col3.metric("→ Capable Model", capable_count)
    col4.metric("Cache Hits", cache_hits)

    st.divider()

    # Display table (newest first)
    st.dataframe(
        df_display.iloc[::-1].reset_index(drop=True),
        use_container_width=True,
        height=min(600, 35 * len(df_display) + 50),
    )

# ---------------------------------------------------------------------------
# Auto-refresh
# ---------------------------------------------------------------------------

if auto_refresh:
    st.empty()
    import time as _time
    _time.sleep(POLL_INTERVAL)
    st.rerun()
