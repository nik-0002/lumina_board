"""
Lumina Board - Enhanced Agricultural Marketing API
Flask backend integrating CSV-based RAG, Qwen2.5 LLM, urgency detection,
and multilingual campaign message generation.
"""

import os
import json
import logging
import glob
import re
from typing import Dict, List, Optional
from datetime import datetime

import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Internal modules
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.rag_engine import CSVRagEngine
from models.urgency_detector import UrgencyDetector
from messaging.campaign_generator import CampaignMessageGenerator
from utils.data_processors import DataProcessor

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.FileHandler("../logs/api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("lumina.api")

# ─── App Setup ────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="../dashboard", static_url_path="")
CORS(app)

DATA_DIR = os.environ.get("DATA_DIR", "../data")

# ─── Singletons (lazy-init) ───────────────────────────────────────────────────
_rag_engine: Optional[CSVRagEngine] = None
_urgency_detector: Optional[UrgencyDetector] = None
_campaign_gen: Optional[CampaignMessageGenerator] = None
_data_processor: Optional[DataProcessor] = None
_datasets: Dict[str, pd.DataFrame] = {}


def get_rag_engine() -> CSVRagEngine:
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = CSVRagEngine(DATA_DIR)
        _rag_engine.build_index()
    return _rag_engine


def get_urgency_detector() -> UrgencyDetector:
    global _urgency_detector
    if _urgency_detector is None:
        _urgency_detector = UrgencyDetector(DATA_DIR)
        _urgency_detector.train()
    return _urgency_detector


def get_campaign_gen() -> CampaignMessageGenerator:
    global _campaign_gen
    if _campaign_gen is None:
        _campaign_gen = CampaignMessageGenerator()
    return _campaign_gen


def get_datasets() -> Dict[str, pd.DataFrame]:
    global _datasets, _data_processor
    if not _datasets:
        _data_processor = DataProcessor(DATA_DIR)
        _datasets = _data_processor.load_all_datasets()
    return _datasets


# ─── Serve Dashboard ──────────────────────────────────────────────────────────
@app.route("/")
def serve_dashboard():
    return send_from_directory("../dashboard", "index.html")


# ─── Health ───────────────────────────────────────────────────────────────────
@app.route("/api/health", methods=["GET"])
def health():
    ds = get_datasets()
    return jsonify({
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "datasets_loaded": list(ds.keys()),
        "record_counts": {k: len(v) for k, v in ds.items()}
    })


# ─── CSV Listing & Preview ────────────────────────────────────────────────────
@app.route("/api/csv/list", methods=["GET"])
def list_csv_files():
    """List all available CSV files in the data directory."""
    csv_files = glob.glob(os.path.join(DATA_DIR, "**/*.csv"), recursive=True)
    csv_files += glob.glob(os.path.join(DATA_DIR, "*.csv"))
    # deduplicate
    csv_files = list(set(csv_files))
    result = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, nrows=1)
            size = os.path.getsize(f)
            full_df = pd.read_csv(f)
            result.append({
                "name": os.path.basename(f),
                "path": f,
                "columns": list(df.columns),
                "row_count": len(full_df),
                "size_kb": round(size / 1024, 2)
            })
        except Exception as e:
            result.append({"name": os.path.basename(f), "error": str(e)})
    return jsonify({"files": result, "count": len(result)})


@app.route("/api/csv/preview", methods=["GET"])
def preview_csv():
    """Preview rows from a CSV file."""
    filename = request.args.get("file")
    rows = int(request.args.get("rows", 20))
    if not filename:
        return jsonify({"error": "file parameter required"}), 400
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404
    try:
        df = pd.read_csv(filepath, nrows=rows)
        # Replace NaN with None for JSON serialization
        df = df.where(pd.notnull(df), None)
        return jsonify({
            "file": filename,
            "columns": list(df.columns),
            "rows": df.to_dict(orient="records"),
            "total_shown": len(df)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/csv/stats", methods=["GET"])
def csv_stats():
    """Return statistical summary of a CSV file."""
    filename = request.args.get("file")
    if not filename:
        return jsonify({"error": "file parameter required"}), 400
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404
    try:
        df = pd.read_csv(filepath)
        desc = df.describe(include="all").fillna("").to_dict()
        return jsonify({
            "file": filename,
            "shape": {"rows": len(df), "cols": len(df.columns)},
            "columns": list(df.columns),
            "dtypes": {c: str(t) for c, t in df.dtypes.items()},
            "null_counts": df.isnull().sum().to_dict(),
            "stats": desc
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── RAG Chat Endpoint ────────────────────────────────────────────────────────
@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Main chat endpoint. Uses CSV RAG to retrieve relevant data,
    then calls Qwen2.5 via Ollama/OpenAI-compatible API to generate response.
    No hallucination: answers grounded strictly in retrieved CSV context.
    """
    body = request.get_json(force=True)
    query = body.get("query", "").strip()
    history = body.get("history", [])  # list of {role, content}
    csv_filter = body.get("csv_filter")  # optional: restrict to specific CSV

    if not query:
        return jsonify({"error": "query required"}), 400

    try:
        rag = get_rag_engine()
        # Retrieve top-k relevant rows from CSVs
        retrieved = rag.query(query, top_k=12, csv_filter=csv_filter)

        # Build grounded context string
        context_parts = []
        sources_used = []
        for item in retrieved:
            context_parts.append(
                f"[Source: {item['source']} | Row {item['row_idx']}]\n{item['text']}"
            )
            if item["source"] not in sources_used:
                sources_used.append(item["source"])

        context_str = "\n\n".join(context_parts) if context_parts else "No relevant data found in CSV files."

        # Build Qwen2.5 prompt
        system_prompt = f"""You are Lumina AI — an intelligent agricultural marketing assistant for Syngenta India.
You have access to real grower data, campaign data, retailer data, and representative data from CSV files.

CRITICAL RULES:
1. ONLY use information from the provided CSV context below. NEVER invent numbers or facts.
2. If the context doesn't contain enough information, say so explicitly.
3. Always cite which data source (CSV file / column) you're drawing from.
4. When asked about trends, always calculate from actual data in context.
5. For campaign planning, use real grower segmentation data only.
6. whenever the user doesn't ask fo specific data point just give them a brief overview of data
7. users might use terms interchangebly or misspell just try to form an understanding of what user is trying to convey
Current date: {datetime.now().strftime("%d %B %Y")}
Data sources available: {', '.join(sources_used) if sources_used else 'None found'}

CSV DATA CONTEXT:
{context_str}
"""

        messages_for_llm = [{"role": "system", "content": system_prompt}]
        # Add conversation history (last 6 turns max for context window)
        for h in history[-6:]:
            messages_for_llm.append({"role": h["role"], "content": h["content"]})
        messages_for_llm.append({"role": "user", "content": query})

        # Call Qwen2.5 via Ollama OpenAI-compatible endpoint
        llm_response = _call_qwen(messages_for_llm)

        return jsonify({
            "response": llm_response,
            "sources": sources_used,
            "retrieved_count": len(retrieved),
            "context_preview": context_str[:500] + "..." if len(context_str) > 500 else context_str
        })

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


def _call_qwen(messages: List[Dict]) -> str:
    """Call Qwen2.5 via Ollama's OpenAI-compatible API."""
    import requests as req
    OLLAMA_BASE = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    MODEL = os.environ.get("LLM_MODEL", "qwen2.5")

    try:
        resp = req.post(
            f"{OLLAMA_BASE}/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": messages,
                "temperature": 0.3,
                "max_tokens": 1200,
                "stream": False
            },
            timeout=60
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except req.exceptions.ConnectionError:
        # Fallback: rule-based response from context
        return _fallback_response(messages)
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return _fallback_response(messages)


def _fallback_response(messages: List[Dict]) -> str:
    """Rule-based fallback when Qwen2.5 is unavailable."""
    user_msg = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    system_msg = next((m["content"] for m in messages if m["role"] == "system"), "")

    # Extract context from system prompt
    if "CSV DATA CONTEXT:" in system_msg:
        context = system_msg.split("CSV DATA CONTEXT:")[-1].strip()
        if len(context) > 100:
            return (
                f"[Qwen2.5 unavailable — showing raw retrieved data]\n\n"
                f"Query: {user_msg}\n\n"
                f"Retrieved context from CSV files:\n{context[:1500]}\n\n"
                f"⚠️ Start Ollama with `ollama run qwen2.5:7b` for AI-powered responses."
            )
    return (
        "⚠️ Qwen2.5 LLM is not running. Please start Ollama: `ollama run qwen2.5:7b`\n"
        "CSV data is loaded and RAG is active — only LLM response generation is unavailable."
    )


# ─── Dashboard Metrics ────────────────────────────────────────────────────────
@app.route("/api/metrics/overview", methods=["GET"])
def metrics_overview():
    """High-level platform metrics from all CSVs."""
    ds = get_datasets()
    out = {"timestamp": datetime.utcnow().isoformat()}

    if "growers" in ds:
        g = ds["growers"]
        out["growers"] = {
            "total": len(g),
            "states": g["state"].value_counts().head(10).to_dict() if "state" in g.columns else {},
            "languages": g["language"].value_counts().to_dict() if "language" in g.columns else {},
            "devices": g["device_type"].value_counts().to_dict() if "device_type" in g.columns else {},
            "avg_farm_size": round(float(g["grower_farm_size"].mean()), 2) if "grower_farm_size" in g.columns else None,
            "avg_age": round(float(g["grower_age"].mean()), 1) if "grower_age" in g.columns else None,
        }

    if "campaigns" in ds:
        c = ds["campaigns"]
        impressions = int(c["social_post_impression"].sum()) if "social_post_impression" in c.columns else 0
        visits = int(c["landing_page_visits"].sum()) if "landing_page_visits" in c.columns else 0
        submissions = int(c["lead_form_submission"].sum()) if "lead_form_submission" in c.columns else 0
        out["campaigns"] = {
            "total_campaigns": c["campaign_id"].nunique() if "campaign_id" in c.columns else len(c),
            "total_impressions": impressions,
            "total_visits": visits,
            "total_submissions": submissions,
            "ctr_pct": round(visits / impressions * 100, 2) if impressions > 0 else 0,
            "conversion_pct": round(submissions / visits * 100, 2) if visits > 0 else 0,
            "by_crop": c.groupby("crop")["lead_form_submission"].sum().to_dict() if "crop" in c.columns else {}
        }

    if "retailers" in ds:
        r = ds["retailers"]
        out["retailers"] = {
            "total": len(r),
            "by_state": r["state"].value_counts().head(8).to_dict() if "state" in r.columns else {}
        }

    if "reps" in ds:
        out["reps"] = {"total": len(ds["reps"])}

    return jsonify(out)


@app.route("/api/metrics/campaigns", methods=["GET"])
def campaign_metrics():
    """Detailed campaign analytics."""
    ds = get_datasets()
    if "campaigns" not in ds:
        return jsonify({"error": "campaigns dataset not loaded"}), 404

    c = ds["campaigns"]
    result = {
        "by_product": {},
        "by_crop": {},
        "by_state": {},
        "top_campaigns": []
    }

    if "product_name" in c.columns and "lead_form_submission" in c.columns:
        result["by_product"] = c.groupby("product_name")["lead_form_submission"].sum().sort_values(ascending=False).head(10).to_dict()

    if "crop" in c.columns and "lead_form_submission" in c.columns:
        result["by_crop"] = c.groupby("crop")["lead_form_submission"].sum().sort_values(ascending=False).to_dict()

    if "state" in c.columns and "lead_form_submission" in c.columns:
        result["by_state"] = c.groupby("state")["lead_form_submission"].sum().sort_values(ascending=False).head(10).to_dict()

    # Top campaigns by conversion
    if "campaign_id" in c.columns and "landing_page_visits" in c.columns and "lead_form_submission" in c.columns:
        grp = c.groupby("campaign_id").agg({
            "landing_page_visits": "sum",
            "lead_form_submission": "sum",
            "social_post_impression": "sum"
        }).reset_index()
        grp["conversion_pct"] = (grp["lead_form_submission"] / grp["landing_page_visits"].replace(0, np.nan) * 100).round(2)
        top = grp.nlargest(5, "lead_form_submission").fillna(0)
        result["top_campaigns"] = top.to_dict(orient="records")

    return jsonify(result)


@app.route("/api/metrics/growers", methods=["GET"])
def grower_metrics():
    """Grower segmentation metrics."""
    ds = get_datasets()
    if "growers" not in ds:
        return jsonify({"error": "growers dataset not loaded"}), 404

    g = ds["growers"]
    state_filter = request.args.get("state")
    if state_filter and "state" in g.columns:
        g = g[g["state"] == state_filter]

    result = {
        "total": len(g),
        "gender": g["gender"].value_counts().to_dict() if "gender" in g.columns else {},
        "age_buckets": {},
        "farm_size_buckets": {},
        "languages": g["language"].value_counts().to_dict() if "language" in g.columns else {},
        "devices": g["device_type"].value_counts().to_dict() if "device_type" in g.columns else {},
        "states": g["state"].value_counts().head(15).to_dict() if "state" in g.columns else {}
    }

    if "grower_age" in g.columns:
        bins = [0, 25, 35, 45, 55, 65, 100]
        labels = ["<25", "25-34", "35-44", "45-54", "55-64", "65+"]
        g["age_bucket"] = pd.cut(g["grower_age"], bins=bins, labels=labels, right=False)
        result["age_buckets"] = g["age_bucket"].value_counts().to_dict()

    if "grower_farm_size" in g.columns:
        bins = [0, 2, 5, 10, 25, 50, 10000]
        labels = ["<2ac", "2-5ac", "5-10ac", "10-25ac", "25-50ac", "50ac+"]
        g["farm_bucket"] = pd.cut(g["grower_farm_size"], bins=bins, labels=labels, right=False)
        result["farm_size_buckets"] = g["farm_bucket"].value_counts().to_dict()

    return jsonify(result)


# ─── Urgency Detection ────────────────────────────────────────────────────────
@app.route("/api/urgency/detect", methods=["POST"])
def detect_urgency():
    """
    Real urgency detection from CSV data analysis + ML model.
    Analyzes grower engagement drop, campaign underperformance, seasonal signals.
    """
    body = request.get_json(force=True)
    state = body.get("state")
    crop = body.get("crop")
    product = body.get("product")

    try:
        detector = get_urgency_detector()
        ds = get_datasets()

        result = detector.detect(
            datasets=ds,
            state_filter=state,
            crop_filter=crop,
            product_filter=product
        )

        return jsonify(result)
    except Exception as e:
        logger.error(f"Urgency detection error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/urgency/bulk", methods=["GET"])
def bulk_urgency():
    """Scan all states/crops for urgency signals."""
    try:
        detector = get_urgency_detector()
        ds = get_datasets()
        result = detector.bulk_scan(ds)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── Campaign Message Generation ─────────────────────────────────────────────
@app.route("/api/campaign/generate", methods=["POST"])
def generate_campaign():
    """
    Generate multilingual SMS + audio scripts for campaigns.
    Languages: Hindi, Marathi, Punjabi, Telugu, Tamil, Kannada, Bengali, Gujarati, English.
    Data grounded: uses actual grower language distribution from CSV.
    """
    body = request.get_json(force=True)
    campaign_type = body.get("campaign_type", "product_launch")
    product = body.get("product", "")
    crop = body.get("crop", "")
    target_state = body.get("state", "")
    languages = body.get("languages", [])  # if empty, auto-detect from CSV
    custom_context = body.get("context", "")

    try:
        ds = get_datasets()
        gen = get_campaign_gen()

        # Auto-detect languages from grower data if not specified
        if not languages and "growers" in ds:
            g = ds["growers"]
            filters = pd.Series([True] * len(g))
            if target_state and "state" in g.columns:
                filters &= g["state"] == target_state
            filtered = g[filters]
            if "language" in filtered.columns and len(filtered) > 0:
                lang_dist = filtered["language"].value_counts()
                languages = lang_dist.head(5).index.tolist()

        if not languages:
            languages = ["Hindi", "English", "Marathi"]

        # Get grower stats for this segment
        segment_stats = {}
        if "growers" in ds:
            g = ds["growers"]
            if target_state and "state" in g.columns:
                seg = g[g["state"] == target_state]
            else:
                seg = g
            segment_stats = {
                "total_growers": len(seg),
                "avg_farm_size": round(float(seg["grower_farm_size"].mean()), 1) if "grower_farm_size" in seg.columns else None,
                "dominant_device": seg["device_type"].mode()[0] if "device_type" in seg.columns and len(seg) > 0 else "unknown"
            }

        messages = gen.generate_multilingual(
            campaign_type=campaign_type,
            product=product,
            crop=crop,
            state=target_state,
            languages=languages,
            context=custom_context,
            segment_stats=segment_stats
        )

        return jsonify({
            "campaign_type": campaign_type,
            "product": product,
            "crop": crop,
            "state": target_state,
            "languages_generated": languages,
            "segment_stats": segment_stats,
            "messages": messages
        })

    except Exception as e:
        logger.error(f"Campaign gen error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# ─── RAG Index Status ─────────────────────────────────────────────────────────
@app.route("/api/rag/status", methods=["GET"])
def rag_status():
    try:
        rag = get_rag_engine()
        return jsonify(rag.get_status())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/rag/rebuild", methods=["POST"])
def rag_rebuild():
    global _rag_engine
    _rag_engine = None
    try:
        rag = get_rag_engine()
        return jsonify({"status": "rebuilt", **rag.get_status()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    debug = os.environ.get("DEBUG", "true").lower() == "true"
    logger.info(f"Starting Lumina Board API on port {port}")
    app.run(host="0.0.0.0", port=port, debug=debug)