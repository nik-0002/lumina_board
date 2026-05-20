#!/usr/bin/env python3
"""
Flask API Server for Syngenta Agricultural Marketing Platform
Integrates with HTML dashboard and CSV data backend
"""

import os
import json
import logging
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import sys
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.campaign_service import CampaignService
from utils.data_processors import DataProcessor
from utils.metrics import MetricsCalculator

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
    campaign_service = CampaignService()
    data_processor = DataProcessor()
    metrics_calc = MetricsCalculator()
    datasets = data_processor.load_all_datasets()
    logger.info("All services initialized successfully")
except Exception as e:
    logger.error(f"Error initializing services: {e}")
    logger.warning("Running in degraded mode")
    datasets = {}

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

# ==================== DATA INSIGHT ROUTES ====================

@app.route('/api/data/insights', methods=['GET'])
def get_data_insights():
    """Get insights from loaded CSV datasets"""
    try:
        insights = campaign_service.get_dataset_insights()
        return jsonify({
            'success': True,
            'insights': insights,
            'datasets_loaded': list(datasets.keys())
        })
    except Exception as e:
        logger.error(f"Error fetching data insights: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/growers', methods=['GET'])
def get_growers():
    """Get growers with optional filtering"""
    try:
        state = request.args.get('state')
        crop = request.args.get('crop')
        page = request.args.get('page', 1, type=int)
        limit = request.args.get('limit', 20, type=int)
        
        if 'growers' not in datasets:
            return jsonify({'success': False, 'error': 'Grower data not loaded'}), 404
        
        growers_df = datasets['growers'].copy()
        
        # Apply filters
        if state:
            growers_df = growers_df[growers_df['state'].str.contains(state, case=False, na=False)]
        
        if crop:
            growers_df = growers_df[growers_df['grower_crop_calendar'].str.contains(crop, case=False, na=False)]
        
        # Pagination
        total = len(growers_df)
        start_idx = (page - 1) * limit
        growers_df = growers_df.iloc[start_idx:start_idx + limit]
        
        growers = growers_df.to_dict('records')
        
        return jsonify({
            'success': True,
            'growers': growers,
            'page': page,
            'limit': limit,
            'total': total
        })
    except Exception as e:
        logger.error(f"Error fetching growers: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/growers/<grower_id>', methods=['GET'])
def get_grower_details(grower_id):
    """Get detailed information about a grower from CSV"""
    try:
        if 'growers' not in datasets:
            return jsonify({'success': False, 'error': 'Grower data not loaded'}), 404
        
        grower = datasets['growers'][datasets['growers']['grower_id'] == grower_id]
        
        if grower.empty:
            return jsonify({'success': False, 'error': 'Grower not found'}), 404
        
        grower_dict = grower.to_dict('records')[0]
        
        return jsonify({
            'success': True,
            'grower': grower_dict
        })
    except Exception as e:
        logger.error(f"Error fetching grower details: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# ==================== CAMPAIGN API ROUTES ====================

@app.route('/api/campaigns', methods=['GET'])
def get_campaigns():
    """Get list of campaigns with optional filtering"""
    try:
        state = request.args.get('state')
        product = request.args.get('product')
        
        if 'campaigns' not in datasets:
            return jsonify({'success': False, 'error': 'Campaign data not loaded'}), 404
        
        campaigns_df = datasets['campaigns'].copy()
        
        # Group by campaign_id
        campaigns_list = []
        for campaign_id in campaigns_df['campaign_id'].unique():
            camp_data = campaigns_df[campaigns_df['campaign_id'] == campaign_id]
            
            campaign = {
                'campaign_id': campaign_id,
                'product': camp_data['campaign_product'].iloc[0],
                'crop': camp_data['campaign_crop'].iloc[0],
                'total_impressions': camp_data['social_post_impression'].sum(),
                'total_visits': camp_data['landing_page_visits'].sum(),
                'total_submissions': camp_data['lead_form_submission'].sum(),
                'weeks': len(camp_data)
            }
            campaigns_list.append(campaign)
        
        # Apply filters
        if product:
            campaigns_list = [c for c in campaigns_list if product.lower() in c['product'].lower()]
        
        return jsonify({
            'success': True,
            'campaigns': campaigns_list,
            'total': len(campaigns_list)
        })
    except Exception as e:
        logger.error(f"Error fetching campaigns: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/campaigns/<campaign_id>/analytics', methods=['GET'])
def get_campaign_analytics(campaign_id):
    """Get campaign analytics from CSV data"""
    try:
        if 'campaigns' not in datasets:
            return jsonify({'success': False, 'error': 'Campaign data not loaded'}), 404
        
        campaigns_df = datasets['campaigns']
        campaign_data = campaigns_df[campaigns_df['campaign_id'] == campaign_id]
        
        if campaign_data.empty:
            return jsonify({'success': False, 'error': 'Campaign not found'}), 404
        
        # Calculate metrics
        total_impressions = campaign_data['social_post_impression'].sum()
        total_visits = campaign_data['landing_page_visits'].sum()
        total_submissions = campaign_data['lead_form_submission'].sum()
        
        ctr = MetricsCalculator.calculate_impression_to_visit(total_impressions, total_visits)
        conversion = MetricsCalculator.calculate_visit_conversion(total_visits, total_submissions)
        
        analytics = {
            'campaign_id': campaign_id,
            'product': campaign_data['campaign_product'].iloc[0],
            'crop': campaign_data['campaign_crop'].iloc[0],
            'performance_metrics': {
                'total_impressions': total_impressions,
                'total_visits': total_visits,
                'total_submissions': total_submissions,
                'click_through_rate': round(ctr, 2),
                'conversion_rate': round(conversion, 2)
            },
            'weekly_breakdown': campaign_data[['week_start_date', 'social_post_impression', 'landing_page_visits', 'lead_form_submission']].to_dict('records')
        }
        
        return jsonify({
            'success': True,
            'analytics': analytics
        })
    except Exception as e:
        logger.error(f"Error fetching campaign analytics: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/campaigns', methods=['POST'])
def create_campaign():
    """Create a new campaign"""
    try:
        data = request.get_json()
        
        campaign_config = {
            'name': data.get('name', 'New Campaign'),
            'crop': data.get('crop', 'wheat'),
            'product': data.get('product', 'Fungicide'),
            'region': data.get('region', 'Punjab'),
            'target_segments': data.get('target_segments', []),
            'num_variants': data.get('num_variants', 3),
            'budget': data.get('budget', 50000)
        }
        
        result = campaign_service.create_campaign(campaign_config)
        
        return jsonify({
            'success': True,
            'campaign_id': result.get('campaign_id'),
            'campaign': result.get('campaign'),
            'target_farmer_count': result.get('target_farmer_count')
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
        
        if not farmer_list:
            return jsonify({'success': False, 'error': 'No farmers provided'}), 400
        
        result = campaign_service.launch_campaign(campaign_id, farmer_list)
        
        return jsonify({
            'success': result['success'],
            'campaign_id': campaign_id,
            'total_messages': result.get('total_messages', 0),
            'successful': result.get('successful', 0),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error launching campaign: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# ==================== ANALYTICS ROUTES ====================

@app.route('/api/analytics/overview', methods=['GET'])
def get_analytics_overview():
    """Get overall platform analytics from CSV data"""
    try:
        overview = {
            'total_growers': len(datasets.get('growers', pd.DataFrame())),
            'total_campaigns': len(datasets.get('campaigns', pd.DataFrame()).groupby('campaign_id')),
            'total_retailers': len(datasets.get('retailers', pd.DataFrame())),
            'total_impressions': datasets.get('campaigns', pd.DataFrame())['social_post_impression'].sum() if 'campaigns' in datasets else 0,
            'total_submissions': datasets.get('campaigns', pd.DataFrame())['lead_form_submission'].sum() if 'campaigns' in datasets else 0
        }
        
        return jsonify({'success': True, 'overview': overview})
    except Exception as e:
        logger.error(f"Error fetching analytics overview: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/analytics/regional-performance', methods=['GET'])
def get_regional_performance():
    """Get regional performance from grower and campaign data"""
    try:
        if 'growers' not in datasets:
            return jsonify({'success': False, 'error': 'Grower data not loaded'}), 404
        
        growers_df = datasets['growers']
        regional = growers_df.groupby('state').size().to_dict()
        
        return jsonify({
            'success': True,
            'regional_data': regional
        })
    except Exception as e:
        logger.error(f"Error fetching regional performance: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# ==================== MESSAGING ROUTES ====================

@app.route('/api/campaigns/<campaign_id>/messaging-setup', methods=['GET'])
def get_messaging_setup(campaign_id):
    """Get messaging setup for campaign - phone numbers and default text"""
    try:
        data = request.args
        grower_ids = data.getlist('grower_ids')
        
        if 'growers' not in datasets:
            return jsonify({'success': False, 'error': 'Grower data not loaded'}), 404
        
        growers_df = datasets['growers']
        
        if grower_ids:
            growers_df = growers_df[growers_df['grower_id'].isin(grower_ids)]
        
        # Create messaging payload
        messaging_list = []
        for _, grower in growers_df.iterrows():
            # Extract phone if available from dataset
            phone = grower.get('phone_number', '')
            
            messaging_list.append({
                'grower_id': grower['grower_id'],
                'name': f"Grower {grower['grower_id']}",
                'phone': phone if phone else '+91XXXXXXXXXX',
                'state': grower['state'],
                'device_type': grower['device_type'],
                'language': grower['language'],
                'selected': False
            })
        
        return jsonify({
            'success': True,
            'campaign_id': campaign_id,
            'messaging_list': messaging_list,
            'default_message': f"Check our latest agricultural solution!"
        })
    except Exception as e:
        logger.error(f"Error getting messaging setup: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# ==================== HEALTH CHECK ====================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'services': {
            'api': 'running',
            'data_processor': 'running',
            'datasets_loaded': list(datasets.keys())
        }
    })

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
    logger.info("API documentation available at: http://localhost:5000/api/health")
    
    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=8000,
        debug=os.getenv('FLASK_DEBUG', 'False') == 'True',
        use_reloader=False
    )
