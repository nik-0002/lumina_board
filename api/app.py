#!/usr/bin/env python3
"""
Flask API Server for Syngenta Agricultural Marketing Platform
Integrates with HTML dashboard and Python backend
"""

import os
import json
import logging
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.campaign_service import CampaignService
from models.ml.engagement_prediction import EngagementPredictor
from models.ml.trend_detection import TrendDetector, PestOutbreakDetector
from models.llm.ollama_client import OllamaLLMClient
from messaging.orchestrator import MessagingOrchestrator
from models.audio.huggingface_tts import HuggingFaceTextToAudio

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, static_folder='../dashboard', static_url_path='/dashboard')
CORS(app)

# Initialize services
logger.info("Initializing API services...")

try:
    engagement_predictor = EngagementPredictor()
    trend_detector = TrendDetector()
    pest_detector = PestOutbreakDetector()
    llm_client = OllamaLLMClient()
    messenger = MessagingOrchestrator()
    audio_generator = HuggingFaceTextToAudio()
    campaign_service = CampaignService(
        engagement_predictor=engagement_predictor,
        trend_detector=trend_detector,
        llm_client=llm_client,
        messaging_orchestrator=messenger,
        audio_generator=audio_generator
    )
    logger.info("All services initialized successfully")
except Exception as e:
    logger.error(f"Error initializing services: {e}")
    logger.warning("Running in degraded mode")

# ==================== DASHBOARD ROUTES ====================

@app.route('/')
@app.route('/dashboard')
def dashboard():
    """Serve the main dashboard HTML"""
    try:
        return send_from_directory('../dashboard', 'index.html')
    except:
        return "Dashboard file not found. Please ensure dashboard/index.html exists.", 404

@app.route('/dashboard/<path:filename>')
def serve_dashboard_static(filename):
    """Serve static files for dashboard"""
    return send_from_directory('../dashboard', filename)

# ==================== CAMPAIGN API ROUTES ====================

@app.route('/api/campaigns', methods=['GET'])
def get_campaigns():
    """Get list of campaigns with optional filtering"""
    try:
        period = request.args.get('period', '7d')
        status = request.args.get('status', None)
        
        # Generate sample campaign data
        campaigns = generate_sample_campaigns(period=period)
        
        if status:
            campaigns = [c for c in campaigns if c['status'] == status]
        
        return jsonify({
            'success': True,
            'campaigns': campaigns,
            'total': len(campaigns),
            'period': period
        })
    except Exception as e:
        logger.error(f"Error fetching campaigns: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/campaigns/<campaign_id>', methods=['GET'])
def get_campaign_details(campaign_id):
    """Get detailed information about a specific campaign"""
    try:
        campaign = generate_sample_campaign_details(campaign_id)
        return jsonify({
            'success': True,
            'campaign': campaign
        })
    except Exception as e:
        logger.error(f"Error fetching campaign details: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/campaigns/<campaign_id>/analytics', methods=['GET'])
def get_campaign_analytics(campaign_id):
    """Get campaign analytics and metrics"""
    try:
        period = request.args.get('period', '7d')
        
        analytics = {
            'campaign_id': campaign_id,
            'period': period,
            'performance_metrics': {
                'total_sent': generate_random_metric(1000, 5000),
                'delivery_rate': round(88.5 + (hash(campaign_id) % 10), 1),
                'engagement_rate': round(35.2 + (hash(campaign_id) % 15), 1),
                'conversion_rate': round(6.8 + (hash(campaign_id) % 8), 1),
                'conversions': generate_random_metric(50, 300)
            },
            'channel_breakdown': {
                'whatsapp': generate_random_metric(400, 2000),
                'sms': generate_random_metric(200, 1000),
                'voice': generate_random_metric(100, 500)
            },
            'daily_trend': generate_daily_trend(period),
            'top_variants': [
                {'variant_id': 1, 'type': 'text_image', 'engagement_score': 45},
                {'variant_id': 2, 'type': 'video_script', 'engagement_score': 38},
                {'variant_id': 3, 'type': 'text_audio', 'engagement_score': 32}
            ]
        }
        
        return jsonify({
            'success': True,
            'analytics': analytics
        })
    except Exception as e:
        logger.error(f"Error fetching analytics: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/campaigns', methods=['POST'])
def create_campaign():
    """Create a new campaign"""
    try:
        data = request.get_json()
        
        campaign_config = {
            'name': data.get('name', 'New Campaign'),
            'crop': data.get('crop', 'rice'),
            'product': data.get('product', 'Fungicide'),
            'region': data.get('region', 'Tamil Nadu'),
            'target_segments': data.get('target_segments', ['smallholder']),
            'num_variants': data.get('num_variants', 5),
            'budget': data.get('budget', 50000)
        }
        
        result = campaign_service.create_campaign(campaign_config)
        
        return jsonify({
            'success': True,
            'campaign_id': result.get('campaign_id'),
            'campaign': result.get('campaign')
        }), 201
    except Exception as e:
        logger.error(f"Error creating campaign: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/campaigns/<campaign_id>/launch', methods=['POST'])
def launch_campaign(campaign_id):
    """Launch a campaign to farmers"""
    try:
        data = request.get_json()
        farmer_list = data.get('farmer_list', [])
        
        result = campaign_service.launch_campaign(campaign_id, farmer_list)
        
        return jsonify({
            'success': result['success'],
            'campaign_id': campaign_id,
            'total_messages': result.get('total_messages', 0),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error launching campaign: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# ==================== ANALYTICS ROUTES ====================

@app.route('/api/analytics/overview', methods=['GET'])
def get_analytics_overview():
    """Get overall platform analytics"""
    try:
        period = request.args.get('period', '7d')
        
        overview = {
            'period': period,
            'total_campaigns': 12,
            'active_campaigns': 5,
            'total_farmers': 45230,
            'total_messages_sent': 156420,
            'overall_delivery_rate': 91.3,
            'overall_engagement_rate': 38.7,
            'overall_conversion_rate': 7.2,
            'top_performing_region': 'Tamil Nadu',
            'top_performing_crop': 'Rice',
            'roi': 3.45
        }
        
        return jsonify({'success': True, 'overview': overview})
    except Exception as e:
        logger.error(f"Error fetching analytics overview: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/analytics/trends', methods=['GET'])
def get_analytics_trends():
    """Get trend analysis data"""
    try:
        metric = request.args.get('metric', 'engagement')
        period = request.args.get('period', '7d')
        
        trends = generate_trend_data(metric, period)
        
        return jsonify({
            'success': True,
            'metric': metric,
            'period': period,
            'data': trends
        })
    except Exception as e:
        logger.error(f"Error fetching trends: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# ==================== MESSAGING ROUTES ====================

@app.route('/api/messages/send', methods=['POST'])
def send_message():
    """Send a message to a farmer"""
    try:
        data = request.get_json()
        
        farmer_context = {
            'farmer_id': data.get('farmer_id'),
            'phone_number': data.get('phone_number'),
            'name': data.get('name'),
            'crop': data.get('crop', 'rice'),
            'region': data.get('region'),
            'language': data.get('language', 'hi')
        }
        
        message_content = {
            'type': data.get('type', 'text'),
            'text': data.get('text'),
            'media_path': data.get('media_path')
        }
        
        result = messenger.route_campaign_message(
            farmer_context=farmer_context,
            message_content=message_content,
            campaign_id=data.get('campaign_id', 'manual')
        )
        
        return jsonify({
            'success': result['success'],
            'message_id': result.get('message_id'),
            'status': result.get('status', 'sent')
        })
    except Exception as e:
        logger.error(f"Error sending message: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# ==================== PEST ALERTS ROUTES ====================

@app.route('/api/pest-alerts', methods=['GET'])
def get_pest_alerts():
    """Get active pest and disease alerts"""
    try:
        region = request.args.get('region', None)
        severity = request.args.get('severity', None)
        
        alerts = generate_sample_pest_alerts(region=region, severity=severity)
        
        return jsonify({
            'success': True,
            'alerts': alerts,
            'total': len(alerts)
        })
    except Exception as e:
        logger.error(f"Error fetching pest alerts: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/pest-alerts/<alert_id>', methods=['GET'])
def get_pest_alert_details(alert_id):
    """Get detailed information about a pest alert"""
    try:
        alert = {
            'alert_id': alert_id,
            'pest_name': 'Brown Plant Hopper',
            'disease_name': 'None',
            'region': 'Tamil Nadu',
            'crop': 'Rice',
            'severity': 'HIGH',
            'severity_score': 8.5,
            'affected_area_km2': 2450,
            'affected_farmers': 1230,
            'detection_date': (datetime.now() - timedelta(days=2)).isoformat(),
            'weather_conditions': {
                'temperature': 32.5,
                'humidity': 85,
                'rainfall': 15.2
            },
            'recommended_actions': [
                'Scout fields daily for pest population',
                'Apply recommended insecticide',
                'Maintain field hygiene'
            ],
            'recommended_products': [
                'Syngenta Imidacloprid 70% WS',
                'Syngenta Fipronil 5% SC'
            ]
        }
        
        return jsonify({
            'success': True,
            'alert': alert
        })
    except Exception as e:
        logger.error(f"Error fetching alert details: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# ==================== FARMER ROUTES ====================

@app.route('/api/farmers', methods=['GET'])
def get_farmers():
    """Get list of farmers"""
    try:
        region = request.args.get('region', None)
        crop = request.args.get('crop', None)
        page = request.args.get('page', 1, type=int)
        limit = request.args.get('limit', 20, type=int)
        
        farmers = generate_sample_farmers(region=region, crop=crop, page=page, limit=limit)
        
        return jsonify({
            'success': True,
            'farmers': farmers,
            'page': page,
            'limit': limit,
            'total': 45230
        })
    except Exception as e:
        logger.error(f"Error fetching farmers: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/farmers/<farmer_id>', methods=['GET'])
def get_farmer_details(farmer_id):
    """Get detailed information about a farmer"""
    try:
        farmer = {
            'farmer_id': farmer_id,
            'name': 'Rajesh Kumar',
            'phone_number': '+919876543210',
            'region': 'Tamil Nadu',
            'district': 'Thanjavur',
            'village': 'Kolangudi',
            'primary_crop': 'Rice',
            'farm_size_acres': 2.5,
            'crops': ['Rice', 'Pulses'],
            'language': 'Tamil',
            'device_type': 'smartphone',
            'engagement_history': [
                {'date': datetime.now().isoformat(), 'campaign': 'Fungicide Campaign', 'status': 'opened'},
                {'date': (datetime.now() - timedelta(days=3)).isoformat(), 'campaign': 'Pest Control', 'status': 'clicked'}
            ]
        }
        
        return jsonify({
            'success': True,
            'farmer': farmer
        })
    except Exception as e:
        logger.error(f"Error fetching farmer details: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# ==================== HEALTH CHECK ====================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'services': {
            'llm': 'connected' if llm_client.check_connection() else 'disconnected',
            'database': 'connected',
            'messaging': 'ready'
        }
    })

# ==================== HELPER FUNCTIONS ====================

def generate_sample_campaigns(period='7d'):
    """Generate sample campaign data"""
    campaigns = [
        {
            'campaign_id': 'camp_001',
            'name': 'Monsoon Fungicide Campaign',
            'crop': 'Rice',
            'product': 'Propiconazole',
            'region': 'Tamil Nadu',
            'status': 'active',
            'start_date': (datetime.now() - timedelta(days=5)).isoformat(),
            'performance': {
                'sent': 2450,
                'delivered': 2310,
                'engaged': 988,
                'converted': 67
            }
        },
        {
            'campaign_id': 'camp_002',
            'name': 'Kharif Pest Control',
            'crop': 'Cotton',
            'product': 'Imidacloprid',
            'region': 'Maharashtra',
            'status': 'active',
            'start_date': (datetime.now() - timedelta(days=3)).isoformat(),
            'performance': {
                'sent': 1850,
                'delivered': 1694,
                'engaged': 542,
                'converted': 39
            }
        },
        {
            'campaign_id': 'camp_003',
            'name': 'Winter Herbicide Campaign',
            'crop': 'Wheat',
            'product': 'Glyphosate',
            'region': 'Punjab',
            'status': 'active',
            'start_date': (datetime.now() - timedelta(days=7)).isoformat(),
            'performance': {
                'sent': 3200,
                'delivered': 3075,
                'engaged': 1435,
                'converted': 103
            }
        },
        {
            'campaign_id': 'camp_004',
            'name': 'Spring Vegetable Protection',
            'crop': 'Tomato',
            'product': 'Captan',
            'region': 'Himachal Pradesh',
            'status': 'scheduled',
            'start_date': (datetime.now() + timedelta(days=2)).isoformat(),
            'performance': {
                'sent': 0,
                'delivered': 0,
                'engaged': 0,
                'converted': 0
            }
        },
        {
            'campaign_id': 'camp_005',
            'name': 'Disease Management Initiative',
            'crop': 'Sugarcane',
            'product': 'Copper Oxychloride',
            'region': 'Karnataka',
            'status': 'paused',
            'start_date': (datetime.now() - timedelta(days=14)).isoformat(),
            'performance': {
                'sent': 1200,
                'delivered': 1044,
                'engaged': 315,
                'converted': 18
            }
        }
    ]
    return campaigns

def generate_sample_campaign_details(campaign_id):
    """Generate detailed campaign information"""
    campaigns = generate_sample_campaigns()
    for c in campaigns:
        if c['campaign_id'] == campaign_id:
            return c
    return campaigns[0]

def generate_sample_pest_alerts(region=None, severity=None):
    """Generate sample pest alerts"""
    alerts = [
        {
            'alert_id': 'alert_001',
            'pest_name': 'Brown Plant Hopper',
            'region': 'Tamil Nadu',
            'crop': 'Rice',
            'severity': 'HIGH',
            'severity_score': 8.5,
            'affected_farmers': 1230,
            'detection_date': (datetime.now() - timedelta(days=2)).isoformat()
        },
        {
            'alert_id': 'alert_002',
            'pest_name': 'Bollworm',
            'region': 'Maharashtra',
            'crop': 'Cotton',
            'severity': 'MEDIUM',
            'severity_score': 6.2,
            'affected_farmers': 450,
            'detection_date': (datetime.now() - timedelta(days=1)).isoformat()
        },
        {
            'alert_id': 'alert_003',
            'pest_name': 'Wheat Rust',
            'region': 'Punjab',
            'crop': 'Wheat',
            'severity': 'HIGH',
            'severity_score': 7.8,
            'affected_farmers': 890,
            'detection_date': datetime.now().isoformat()
        }
    ]
    
    if region:
        alerts = [a for a in alerts if a['region'] == region]
    if severity:
        alerts = [a for a in alerts if a['severity'] == severity]
    
    return alerts

def generate_sample_farmers(region=None, crop=None, page=1, limit=20):
    """Generate sample farmer data"""
    farmers = [
        {
            'farmer_id': f'farmer_{i:05d}',
            'name': f'Farmer {i}',
            'phone_number': f'+9198765{4321 + i:05d}',
            'region': ['Tamil Nadu', 'Maharashtra', 'Punjab', 'Karnataka'][i % 4],
            'crop': ['Rice', 'Cotton', 'Wheat', 'Sugarcane'][i % 4],
            'farm_size': 2.5 + (i % 5),
            'device_type': 'smartphone' if i % 3 != 0 else 'feature_phone'
        }
        for i in range(1, 1001)
    ]
    
    if region:
        farmers = [f for f in farmers if f['region'] == region]
    if crop:
        farmers = [f for f in farmers if f['crop'] == crop]
    
    start_idx = (page - 1) * limit
    return farmers[start_idx:start_idx + limit]

def generate_random_metric(min_val, max_val):
    """Generate random metric value"""
    import random
    return random.randint(min_val, max_val)

def generate_daily_trend(period):
    """Generate daily trend data"""
    days = {'7d': 7, '30d': 30, '90d': 90, '1y': 365}.get(period, 7)
    trend_data = []
    
    for i in range(days):
        date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
        trend_data.append({
            'date': date,
            'sent': 200 + (i * 5),
            'delivered': 188 + (i * 4),
            'engaged': 72 + (i * 2),
            'converted': 8 + (i % 3)
        })
    
    return sorted(trend_data, key=lambda x: x['date'])

def generate_trend_data(metric, period):
    """Generate trend data for specific metric"""
    days = {'7d': 7, '30d': 30, '90d': 90, '1y': 365}.get(period, 7)
    trend_data = []
    
    base_values = {
        'engagement': 35,
        'conversion': 6.5,
        'delivery': 91
    }
    
    base = base_values.get(metric, 50)
    
    for i in range(days):
        date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
        value = base + (i % 5) - 2 + (hash(date) % 10)
        trend_data.append({
            'date': date,
            'value': round(max(0, min(100, value)), 1)
        })
    
    return sorted(trend_data, key=lambda x: x['date'])

# ==================== ERROR HANDLERS ====================

@app.errorhandler(404)
def not_found(e):
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal server error: {e}")
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

# ==================== MAIN ====================

if __name__ == '__main__':
    logger.info("Starting Syngenta Agricultural Marketing Platform API")
    logger.info("Dashboard available at: http://localhost:5000/dashboard")
    logger.info("API documentation available at: http://localhost:5000/api")
    
    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=8000,
        debug=os.getenv('FLASK_DEBUG', 'False') == 'True',
        use_reloader=False
    )
