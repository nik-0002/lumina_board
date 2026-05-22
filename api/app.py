"""
Lumina Board - Enhanced Agricultural Marketing API with Qwen2.5 Integration
Flask backend integrating CSV-based RAG, local Qwen2.5 LLM, urgency detection,
multilingual campaign generation, and comprehensive data analytics.
"""

import os
import json
import logging
import glob
import re
import requests
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import traceback

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
QWEN_API_URL = os.environ.get("QWEN_API_URL", "http://localhost:11434/api/generate")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen2.5:latest")

# ─── Singletons (lazy-init) ───────────────────────────────────────────────────
_rag_engine: Optional[CSVRagEngine] = None
_urgency_detector: Optional[UrgencyDetector] = None
_campaign_gen: Optional[CampaignMessageGenerator] = None
_data_processor: Optional[DataProcessor] = None
_datasets: Dict[str, pd.DataFrame] = {}
_data_cache: Dict[str, Any] = {}


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
    """Load all CSV datasets with proper type handling"""
    global _datasets, _data_processor
    if not _datasets:
        _data_processor = DataProcessor(DATA_DIR)
        _datasets = {}
        
        # Load all CSVs comprehensively
        csv_files = {
            'growers': 'growers.csv',
            'retailers': 'retailers.csv',
            'retailer_pos': 'retailer_pos.csv',
            'retailer_inventory': 'retailer_inventory_weekly.csv',
            'retailer_visits': 'retailer_visit_log.csv',
            'reps_territory': 'reps_territory.csv',
            'campaigns': 'digital_funnel_weekly.csv',
            'whatsapp': 'whatsapp_campaign.csv'
        }
        
        for key, filename in csv_files.items():
            filepath = os.path.join(DATA_DIR, filename)
            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath, low_memory=False)
                    # Convert date columns
                    date_columns = [col for col in df.columns if 'date' in col.lower() or 'datetime' in col.lower()]
                    for col in date_columns:
                        try:
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                        except:
                            pass
                    _datasets[key] = df
                    logger.info(f"Loaded {key}: {len(df)} rows, {len(df.columns)} columns")
                except Exception as e:
                    logger.error(f"Error loading {filename}: {e}")
    
    return _datasets


def call_qwen_api(prompt: str, system_prompt: str = None, max_tokens: int = 2000) -> str:
    """Call local Qwen2.5 API (Ollama compatible)"""
    try:
        payload = {
            "model": QWEN_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": max_tokens
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        response = requests.post(QWEN_API_URL, json=payload, timeout=180)
        response.raise_for_status()
        
        result = response.json()
        return result.get('response', '').strip()
    
    except Exception as e:
        logger.error(f"Qwen API error: {e}")
        return f"Error calling AI model: {str(e)}"


def analyze_dataframe_comprehensive(df: pd.DataFrame, name: str) -> Dict[str, Any]:
    """Comprehensive analysis of a dataframe"""
    analysis = {
        'name': name,
        'shape': {'rows': len(df), 'cols': len(df.columns)},
        'columns': list(df.columns),
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
        'null_counts': df.isnull().sum().to_dict(),
        'null_percentages': (df.isnull().sum() / len(df) * 100).to_dict(),
    }
    
    # Numeric columns statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        analysis['numeric_stats'] = df[numeric_cols].describe().to_dict()
    
    # Categorical columns value counts (top 10)
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    analysis['categorical_stats'] = {}
    for col in categorical_cols[:10]:  # Limit to first 10 to avoid huge output
        value_counts = df[col].value_counts().head(10).to_dict()
        analysis['categorical_stats'][col] = value_counts
    
    # Date range for date columns
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    analysis['date_ranges'] = {}
    for col in date_cols:
        valid_dates = df[col].dropna()
        if len(valid_dates) > 0:
            analysis['date_ranges'][col] = {
                'min': str(valid_dates.min()),
                'max': str(valid_dates.max()),
                'range_days': (valid_dates.max() - valid_dates.min()).days
            }
    
    return analysis


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
        "record_counts": {k: len(v) for k, v in ds.items()},
        "qwen_model": QWEN_MODEL,
        "qwen_api": QWEN_API_URL
    })


# ─── CSV Listing & Comprehensive Analysis ─────────────────────────────────────
@app.route("/api/csv/list", methods=["GET"])
def list_csv_files():
    """List all available CSV files with comprehensive metadata"""
    csv_files = glob.glob(os.path.join(DATA_DIR, "**/*.csv"), recursive=True)
    csv_files += glob.glob(os.path.join(DATA_DIR, "*.csv"))
    csv_files = list(set(csv_files))
    
    result = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, nrows=1)
            full_df = pd.read_csv(f, low_memory=False)
            size = os.path.getsize(f)
            
            result.append({
                "name": os.path.basename(f),
                "path": f,
                "columns": list(df.columns),
                "column_count": len(df.columns),
                "row_count": len(full_df),
                "size_kb": round(size / 1024, 2),
                "size_mb": round(size / 1024 / 1024, 2),
                "dtypes": {col: str(dtype) for col, dtype in full_df.dtypes.items()}
            })
        except Exception as e:
            result.append({"name": os.path.basename(f), "error": str(e)})
    
    return jsonify({"files": result, "count": len(result)})


@app.route("/api/csv/comprehensive-analysis", methods=["GET"])
def comprehensive_csv_analysis():
    """Comprehensive analysis of all CSV datasets"""
    ds = get_datasets()
    
    analyses = {}
    for name, df in ds.items():
        analyses[name] = analyze_dataframe_comprehensive(df, name)
    
    # Cross-dataset insights
    total_records = sum(len(df) for df in ds.values())
    total_columns = sum(len(df.columns) for df in ds.values())
    
    summary = {
        'total_datasets': len(ds),
        'total_records': total_records,
        'total_columns': total_columns,
        'datasets': analyses,
        'timestamp': datetime.utcnow().isoformat()
    }
    
    return jsonify(summary)


@app.route("/api/csv/preview", methods=["GET"])
def preview_csv():
    """Preview rows from a CSV file"""
    filename = request.args.get("file")
    rows = int(request.args.get("rows", 20))
    if not filename:
        return jsonify({"error": "file parameter required"}), 400
    
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404
    
    try:
        df = pd.read_csv(filepath, nrows=rows, low_memory=False)
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
    """Return comprehensive statistical summary of a CSV file"""
    filename = request.args.get("file")
    if not filename:
        return jsonify({"error": "file parameter required"}), 400
    
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404
    
    try:
        df = pd.read_csv(filepath, low_memory=False)
        analysis = analyze_dataframe_comprehensive(df, filename)
        return jsonify(analysis)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── Enhanced RAG Chat with Qwen2.5 ───────────────────────────────────────────
@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Enhanced chat endpoint with local Qwen2.5 integration.
    Uses comprehensive CSV RAG to retrieve ALL relevant data,
    then generates grounded responses via local Qwen2.5 API.
    """
    body = request.get_json(force=True)
    query = body.get("query", "").strip()
    history = body.get("history", [])
    csv_filter = body.get("csv_filter")
    
    if not query:
        return jsonify({"error": "query required"}), 400
    
    try:
        # Get all datasets for comprehensive context
        ds = get_datasets()
        
        # Build comprehensive data context
        context_parts = []
        
        # If specific data query, provide relevant statistics
        if any(keyword in query.lower() for keyword in ['grower', 'farmer', 'campaign', 'retailer', 'sales', 'inventory']):
            for name, df in ds.items():
                if csv_filter and name != csv_filter:
                    continue
                
                relevant = False
                if 'grower' in query.lower() and 'grower' in name:
                    relevant = True
                elif 'campaign' in query.lower() and 'campaign' in name:
                    relevant = True
                elif 'retailer' in query.lower() and 'retailer' in name:
                    relevant = True
                elif 'sales' in query.lower() and 'pos' in name:
                    relevant = True
                elif not csv_filter:
                    relevant = True
                
                if relevant:
                    # Add comprehensive dataset summary
                    summary = f"\n[Dataset: {name}]\n"
                    summary += f"Total Records: {len(df)}\n"
                    summary += f"Columns: {', '.join(df.columns.tolist()[:10])}\n"
                    
                    # Add sample data
                    if len(df) > 0:
                        sample = df.head(5).to_dict(orient='records')
                        summary += f"Sample Data: {json.dumps(sample, default=str)}\n"
                    
                    context_parts.append(summary)
        
        # Also use RAG engine for semantic search
        try:
            rag = get_rag_engine()
            retrieved = rag.query(query, top_k=15, csv_filter=csv_filter)
            
            for item in retrieved:
                context_parts.append(
                    f"[Source: {item['source']} | Row {item['row_idx']}]\n{item['text']}"
                )
        except Exception as rag_error:
            logger.error(f"RAG error: {rag_error}")
        
        context_str = "\n\n".join(context_parts) if context_parts else "No specific data context available."
        
        # Build conversation context
        conv_context = ""
        for msg in history[-5:]:  # Last 5 messages
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            conv_context += f"{role.upper()}: {content}\n"
        
        # System prompt for Qwen2.5
        system_prompt = f"""You are Lumina AI — an intelligent agricultural marketing assistant for Syngenta India.
You have comprehensive access to real agricultural data including grower profiles, campaign performance, 
retailer analytics, sales data, and territory information from CSV files.

CRITICAL RULES:
1. ONLY use information from the provided data context. NEVER invent statistics or numbers.
2. If data doesn't contain enough information, explicitly state what's missing.
3. Always cite which dataset and specific metrics you're referencing.
4. Be conversational but precise. Format numbers with proper separators.
5. When showing data, present it clearly with proper formatting.
6. For trends, calculate from actual data in context.
7. Provide actionable insights based on the data.
8. If asked about something not in the data, say so and offer related information you do have.

AVAILABLE DATASETS:
{', '.join(ds.keys())}

CURRENT DATA CONTEXT:
{context_str}

CONVERSATION HISTORY:
{conv_context}"""
        
        # Call Qwen2.5
        full_prompt = f"{system_prompt}\n\nUSER QUERY: {query}\n\nASSISTANT:"
        
        response = call_qwen_api(full_prompt, max_tokens=3000)
        
        # Extract sources used
        sources_used = list(set([item['source'] for item in retrieved])) if 'retrieved' in locals() else []
        
        return jsonify({
            "response": response,
            "sources": sources_used,
            "context_size": len(context_str),
            "datasets_consulted": list(ds.keys()),
            "timestamp": datetime.utcnow().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Chat error: {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


# ─── Enhanced Metrics Endpoints ───────────────────────────────────────────────
@app.route("/api/metrics/overview", methods=["GET"])
def metrics_overview():
    """Comprehensive dashboard overview using ALL data"""
    try:
        ds = get_datasets()
        
        overview = {
            "timestamp": datetime.utcnow().isoformat(),
            "total_growers": len(ds.get('growers', pd.DataFrame())),
            "total_retailers": len(ds.get('retailers', pd.DataFrame())),
            "total_territories": len(ds.get('reps_territory', pd.DataFrame())),
            "active_campaigns": len(ds.get('campaigns', pd.DataFrame())['campaign_id'].unique()) if 'campaigns' in ds else 0,
        }
        
        # Calculate revenue from POS data
        if 'retailer_pos' in ds:
            pos = ds['retailer_pos']
            if 'sku_price' in pos.columns and 'sku_qty' in pos.columns:
                pos['revenue'] = pos['sku_price'] * pos['sku_qty']
                overview['total_revenue'] = float(pos['revenue'].sum())
                overview['avg_transaction_value'] = float(pos['revenue'].mean())
                overview['total_transactions'] = len(pos)
        
        # Campaign performance
        if 'campaigns' in ds:
            camp = ds['campaigns']
            if 'lead_form_submission' in camp.columns:
                overview['total_leads'] = int(camp['lead_form_submission'].sum())
            if 'landing_page_visits' in camp.columns:
                overview['total_visits'] = int(camp['landing_page_visits'].sum())
                overview['overall_conversion_rate'] = round(
                    (overview.get('total_leads', 0) / overview.get('total_visits', 1)) * 100, 2
                )
        
        # Grower engagement
        if 'growers' in ds:
            growers = ds['growers']
            if 'product_scan' in growers.columns:
                overview['growers_engaged'] = int(growers['product_scan'].sum())
                overview['engagement_rate'] = round(
                    (overview['growers_engaged'] / len(growers)) * 100, 2
                )
        
        # Retailer activity
        if 'retailer_visits' in ds:
            visits = ds['retailer_visits']
            overview['total_field_visits'] = len(visits)
            if 'visit_date' in visits.columns:
                visits['visit_date'] = pd.to_datetime(visits['visit_date'], errors='coerce')
                recent_visits = visits[visits['visit_date'] >= datetime.now() - timedelta(days=30)]
                overview['visits_last_30_days'] = len(recent_visits)
        
        # WhatsApp campaign stats
        if 'whatsapp' in ds:
            wa = ds['whatsapp']
            overview['whatsapp_messages_sent'] = len(wa)
            if 'delivered_status' in wa.columns:
                overview['whatsapp_delivered'] = int(wa['delivered_status'].sum())
                overview['whatsapp_delivery_rate'] = round(
                    (overview['whatsapp_delivered'] / len(wa)) * 100, 2
                )
            if 'clicked_status' in wa.columns:
                overview['whatsapp_clicks'] = int(wa['clicked_status'].sum())
                overview['whatsapp_ctr'] = round(
                    (overview['whatsapp_clicks'] / len(wa)) * 100, 2
                )
        
        return jsonify(overview)
    
    except Exception as e:
        logger.error(f"Overview error: {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/metrics/campaigns", methods=["GET"])
def campaign_metrics():
    """Enhanced campaign analytics using ALL campaign data"""
    ds = get_datasets()
    if "campaigns" not in ds:
        return jsonify({"error": "campaigns dataset not loaded"}), 404
    
    c = ds["campaigns"]
    result = {
        "by_product": {},
        "by_crop": {},
        "by_campaign": {},
        "weekly_trends": [],
        "top_campaigns": [],
        "conversion_funnel": {}
    }
    
    # By product
    if "campaign_product" in c.columns and "lead_form_submission" in c.columns:
        result["by_product"] = c.groupby("campaign_product")["lead_form_submission"].sum().sort_values(ascending=False).to_dict()
    
    # By crop
    if "campaign_crop" in c.columns and "lead_form_submission" in c.columns:
        result["by_crop"] = c.groupby("campaign_crop")["lead_form_submission"].sum().sort_values(ascending=False).to_dict()
    
    # Weekly trends
    if "week_start_date" in c.columns:
        c['week_start_date'] = pd.to_datetime(c['week_start_date'], errors='coerce')
        weekly = c.groupby('week_start_date').agg({
            'social_post_impression': 'sum',
            'landing_page_visits': 'sum',
            'lead_form_submission': 'sum'
        }).reset_index()
        weekly['week_start_date'] = weekly['week_start_date'].astype(str)
        result["weekly_trends"] = weekly.to_dict(orient='records')
    
    # Top campaigns by multiple metrics
    if "campaign_id" in c.columns:
        grp = c.groupby("campaign_id").agg({
            "landing_page_visits": "sum",
            "lead_form_submission": "sum",
            "social_post_impression": "sum"
        }).reset_index()
        
        grp["conversion_pct"] = (grp["lead_form_submission"] / grp["landing_page_visits"].replace(0, np.nan) * 100).round(2)
        grp["ctr"] = (grp["landing_page_visits"] / grp["social_post_impression"].replace(0, np.nan) * 100).round(2)
        
        top = grp.nlargest(10, "lead_form_submission").fillna(0)
        result["top_campaigns"] = top.to_dict(orient="records")
    
    # Overall conversion funnel
    result["conversion_funnel"] = {
        "impressions": int(c["social_post_impression"].sum()) if "social_post_impression" in c.columns else 0,
        "visits": int(c["landing_page_visits"].sum()) if "landing_page_visits" in c.columns else 0,
        "leads": int(c["lead_form_submission"].sum()) if "lead_form_submission" in c.columns else 0
    }
    
    return jsonify(result)


@app.route("/api/metrics/growers", methods=["GET"])
def grower_metrics():
    """Comprehensive grower segmentation using ALL grower data"""
    ds = get_datasets()
    if "growers" not in ds:
        return jsonify({"error": "growers dataset not loaded"}), 404
    
    g = ds["growers"]
    state_filter = request.args.get("state")
    
    if state_filter and "state" in g.columns:
        g = g[g["state"] == state_filter]
    
    result = {
        "total": len(g),
        "state_filter": state_filter,
        "demographics": {},
        "engagement": {},
        "geographic": {},
        "technology": {}
    }
    
    # Demographics
    if "gender" in g.columns:
        result["demographics"]["gender"] = g["gender"].value_counts().to_dict()
    
    if "grower_age" in g.columns:
        bins = [0, 25, 35, 45, 55, 65, 100]
        labels = ["<25", "25-34", "35-44", "45-54", "55-64", "65+"]
        g["age_bucket"] = pd.cut(g["grower_age"], bins=bins, labels=labels, right=False)
        result["demographics"]["age_buckets"] = g["age_bucket"].value_counts().to_dict()
        result["demographics"]["avg_age"] = round(float(g["grower_age"].mean()), 1)
    
    if "grower_farm_size" in g.columns:
        bins = [0, 2, 5, 10, 25, 50, 10000]
        labels = ["<2ac", "2-5ac", "5-10ac", "10-25ac", "25-50ac", "50ac+"]
        g["farm_bucket"] = pd.cut(g["grower_farm_size"], bins=bins, labels=labels, right=False)
        result["demographics"]["farm_size_buckets"] = g["farm_bucket"].value_counts().to_dict()
        result["demographics"]["avg_farm_size"] = round(float(g["grower_farm_size"].mean()), 2)
        result["demographics"]["total_acreage"] = round(float(g["grower_farm_size"].sum()), 2)
    
    # Engagement metrics
    if "product_scan" in g.columns:
        result["engagement"]["product_scans"] = int(g["product_scan"].sum())
        result["engagement"]["scan_rate"] = round((g["product_scan"].sum() / len(g)) * 100, 2)
    
    if "offline_campaign_attended" in g.columns:
        result["engagement"]["campaign_attendees"] = int(g["offline_campaign_attended"].sum())
        result["engagement"]["attendance_rate"] = round((g["offline_campaign_attended"].sum() / len(g)) * 100, 2)
    
    if "product_name" in g.columns:
        result["engagement"]["top_products_scanned"] = g["product_name"].value_counts().head(10).to_dict()
    
    # Geographic distribution
    if "state" in g.columns:
        result["geographic"]["states"] = g["state"].value_counts().to_dict()
    
    if "district" in g.columns:
        result["geographic"]["top_districts"] = g["district"].value_counts().head(15).to_dict()
    
    if "tehsil" in g.columns:
        result["geographic"]["tehsil_count"] = len(g["tehsil"].unique())
    
    # Technology adoption
    if "language" in g.columns:
        result["technology"]["languages"] = g["language"].value_counts().to_dict()
    
    if "device_type" in g.columns:
        result["technology"]["devices"] = g["device_type"].value_counts().to_dict()
        smartphone_pct = (g["device_type"] == "smartphone").sum() / len(g) * 100
        result["technology"]["smartphone_penetration"] = round(smartphone_pct, 2)
    
    return jsonify(result)


@app.route("/api/metrics/retailers", methods=["GET"])
def retailer_metrics():
    """Comprehensive retailer analytics using ALL retailer data"""
    ds = get_datasets()
    
    result = {
        "overview": {},
        "sales_performance": {},
        "inventory_health": {},
        "visit_activity": {}
    }
    
    # Overview from retailers master
    if "retailers" in ds:
        retailers = ds["retailers"]
        result["overview"]["total_retailers"] = len(retailers)
        
        if "state" in retailers.columns:
            result["overview"]["states_covered"] = len(retailers["state"].unique())
            result["overview"]["by_state"] = retailers["state"].value_counts().to_dict()
        
        if "district" in retailers.columns:
            result["overview"]["districts_covered"] = len(retailers["district"].unique())
    
    # Sales performance from POS data
    if "retailer_pos" in ds:
        pos = ds["retailer_pos"]
        
        if "sku_price" in pos.columns and "sku_qty" in pos.columns:
            pos["revenue"] = pos["sku_price"] * pos["sku_qty"]
            result["sales_performance"]["total_revenue"] = float(pos["revenue"].sum())
            result["sales_performance"]["total_units_sold"] = int(pos["sku_qty"].sum())
            result["sales_performance"]["avg_transaction_value"] = float(pos["revenue"].mean())
        
        if "retailer_id" in pos.columns:
            retailer_sales = pos.groupby("retailer_id")["revenue"].sum() if "revenue" in pos.columns else None
            if retailer_sales is not None:
                result["sales_performance"]["top_retailers"] = retailer_sales.nlargest(10).to_dict()
        
        if "sku_name" in pos.columns:
            product_sales = pos.groupby("sku_name")["sku_qty"].sum().nlargest(15)
            result["sales_performance"]["top_products"] = product_sales.to_dict()
        
        if "transaction_date" in pos.columns:
            pos["transaction_date"] = pd.to_datetime(pos["transaction_date"], errors='coerce')
            recent_sales = pos[pos["transaction_date"] >= datetime.now() - timedelta(days=30)]
            result["sales_performance"]["sales_last_30_days"] = len(recent_sales)
    
    # Inventory health
    if "retailer_inventory" in ds:
        inv = ds["retailer_inventory"]
        result["inventory_health"]["total_sku_records"] = len(inv)
        
        if "sku_qty" in inv.columns:
            out_of_stock = (inv["sku_qty"] == 0).sum()
            result["inventory_health"]["out_of_stock_instances"] = int(out_of_stock)
            result["inventory_health"]["stock_availability_rate"] = round(
                ((len(inv) - out_of_stock) / len(inv)) * 100, 2
            )
        
        if "sku_name" in inv.columns:
            result["inventory_health"]["unique_skus"] = len(inv["sku_name"].unique())
    
    # Visit activity
    if "retailer_visits" in ds:
        visits = ds["retailer_visits"]
        result["visit_activity"]["total_visits"] = len(visits)
        
        if "visit_type" in visits.columns:
            result["visit_activity"]["by_type"] = visits["visit_type"].value_counts().to_dict()
        
        if "product_recommended" in visits.columns:
            result["visit_activity"]["top_recommended_products"] = visits["product_recommended"].value_counts().head(10).to_dict()
        
        if "visit_date" in visits.columns:
            visits["visit_date"] = pd.to_datetime(visits["visit_date"], errors='coerce')
            recent_visits = visits[visits["visit_date"] >= datetime.now() - timedelta(days=30)]
            result["visit_activity"]["visits_last_30_days"] = len(recent_visits)
    
    return jsonify(result)


# ─── Urgency Detection (unchanged but using comprehensive data) ────────────────
@app.route("/api/urgency/detect", methods=["POST"])
def detect_urgency():
    """Real urgency detection from comprehensive CSV data analysis"""
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
        logger.error(f"Urgency detection error: {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/urgency/bulk", methods=["GET"])
def bulk_urgency():
    """Scan all states/crops for urgency signals using comprehensive data"""
    try:
        detector = get_urgency_detector()
        ds = get_datasets()
        result = detector.bulk_scan(ds)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Bulk urgency error: {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


# ─── Campaign Message Generation (using Qwen2.5) ──────────────────────────────
@app.route("/api/campaign/generate", methods=["POST"])
def generate_campaign():
    """
    Generate multilingual SMS + audio scripts using Qwen2.5.
    Grounded in actual grower language distribution from comprehensive CSV data.
    """
    body = request.get_json(force=True)
    campaign_type = body.get("campaign_type", "product_launch")
    product = body.get("product", "")
    crop = body.get("crop", "")
    target_state = body.get("state", "")
    languages = body.get("languages", [])
    custom_context = body.get("context", "")
    
    try:
        ds = get_datasets()
        
        # Auto-detect languages from comprehensive grower data
        if not languages and "growers" in ds:
            g = ds["growers"]
            filters = pd.Series([True] * len(g))
            if target_state and "state" in g.columns:
                filters &= g["state"] == target_state
            if crop and "grower_crop_calendar" in g.columns:
                # Parse JSON crop calendar
                filters &= g["grower_crop_calendar"].str.contains(crop, case=False, na=False)
            
            filtered = g[filters]
            if "language" in filtered.columns and len(filtered) > 0:
                lang_dist = filtered["language"].value_counts()
                languages = lang_dist.head(5).index.tolist()
        
        if not languages:
            languages = ["Hindi", "English", "Marathi"]
        
        # Get comprehensive segment statistics
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
                "dominant_device": seg["device_type"].mode()[0] if "device_type" in seg.columns and len(seg) > 0 else "unknown",
                "smartphone_pct": round((seg["device_type"] == "smartphone").sum() / len(seg) * 100, 2) if "device_type" in seg.columns else 0,
                "avg_age": round(float(seg["grower_age"].mean()), 1) if "grower_age" in seg.columns else None,
                "language_distribution": seg["language"].value_counts().to_dict() if "language" in seg.columns else {}
            }
        
        # Use local Qwen2.5 for message generation
        messages = {}
        for lang in languages:
            prompt = f"""Generate a {campaign_type} campaign message for:
Product: {product}
Crop: {crop}
Target State: {target_state}
Language: {lang}
Additional Context: {custom_context}

Segment Statistics: {json.dumps(segment_stats)}

Generate:
1. SMS message (max 160 characters, culturally appropriate for {lang})
2. Audio script (30-45 seconds, engaging tone)

Format as JSON:
{{
    "sms": "...",
    "audio_script": "...",
    "script": "{lang} ({product})",
    "char_count": ...,
    "sms_parts": ...,
    "estimated_audio_duration_sec": ...
}}"""
            
            response = call_qwen_api(prompt, max_tokens=1000)
            
            try:
                # Try to parse JSON response
                msg_data = json.loads(response)
            except:
                # If not JSON, create structured response
                msg_data = {
                    "sms": response[:160],
                    "audio_script": response,
                    "script": f"{lang} ({product})",
                    "char_count": len(response[:160]),
                    "sms_parts": (len(response[:160]) // 160) + 1,
                    "estimated_audio_duration_sec": 30
                }
            
            messages[lang] = msg_data
        
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
        logger.error(f"Campaign gen error: {e}\n{traceback.format_exc()}")
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
    logger.info(f"Starting Lumina Board Enhanced API on port {port}")
    logger.info(f"Qwen2.5 API: {QWEN_API_URL}")
    logger.info(f"Model: {QWEN_MODEL}")
    app.run(host="0.0.0.0", port=port, debug=debug)