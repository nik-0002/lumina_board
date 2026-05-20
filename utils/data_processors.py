import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List

class DataProcessor:
    """
    Utility functions for data transformation and validation
    """
    
    @staticmethod
    def clean_farmer_data(farmer_data: Dict) -> Dict:
        """Validate and clean farmer profile data"""
        cleaned = {}
        
        # Required fields
        required_fields = ['farmer_id', 'phone_number', 'region', 'primary_crop']
        for field in required_fields:
            if field not in farmer_data:
                raise ValueError(f"Missing required field: {field}")
            cleaned[field] = farmer_data[field]
        
        # Optional fields with defaults
        cleaned['farm_size'] = farmer_data.get('farm_size', 2.5)
        cleaned['crops'] = farmer_data.get('crops', [farmer_data['primary_crop']])
        cleaned['device_type'] = farmer_data.get('device_type', 'smartphone')
        cleaned['language'] = farmer_data.get('language', 'hi')
        cleaned['connectivity_level'] = farmer_data.get('connectivity_level', 'medium')
        
        return cleaned
    
    @staticmethod
    def extract_temporal_features(timestamp: str) -> Dict:
        """Extract temporal features from ISO timestamp"""
        dt = datetime.fromisoformat(timestamp)
        
        return {
            'hour': dt.hour,
            'day_of_week': dt.weekday(),
            'day_of_month': dt.day,
            'month': dt.month,
            'quarter': (dt.month - 1) // 3 + 1,
            'is_weekend': dt.weekday() >= 5,
            'season': DataProcessor._get_season(dt.month)
        }
    
    @staticmethod
    def _get_season(month: int) -> str:
        """Get agricultural season for India"""
        if month in [3, 4, 5]:
            return 'summer'
        elif month in [6, 7, 8, 9]:
            return 'monsoon'
        elif month in [10, 11]:
            return 'post_monsoon'
        else:
            return 'winter'
    
    @staticmethod
    def aggregate_campaign_data(raw_data: List[Dict], group_by: str = 'region') -> pd.DataFrame:
        """Aggregate campaign data by specified dimension"""
        df = pd.DataFrame(raw_data)
        
        if group_by == 'region':
            return df.groupby('region').agg({
                'sent': 'sum',
                'delivered': 'sum',
                'engaged': 'sum',
                'converted': 'sum'
            }).reset_index()
        
        elif group_by == 'crop':
            return df.groupby('crop').agg({
                'sent': 'sum',
                'delivered': 'sum',
                'engaged': 'sum',
                'converted': 'sum'
            }).reset_index()
        
        return df