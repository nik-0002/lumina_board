"""
Syngenta AI-Powered Agricultural Marketing Platform
Main application orchestrator
"""

import os
import json
import logging
from typing import Dict

from config import Config
from models.ml.trend_detection import TrendDetector, PestOutbreakDetector
from models.ml.engagement_prediction import EngagementPredictor, CampaignOptimizer
from models.ml.crop_segmentation import CropSegmentationEngine
from models.llm.ollama_client import OllamaLLMClient, ContentGenerator
from models.audio.huggingface_tts import HuggingFaceTextToAudio
from rag.retriever import RAGRetriever
from messaging.orchestrator import MessagingOrchestrator
from messaging.channels.adb_whatsapp import ADBWhatsAppController
from messaging.channels.twillio_gateway import TwilioMessagingGateway
from api.campaign_service import CampaignService

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agrimarketing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AgriculturalMarketingPlatform:
    """
    Main platform orchestrator
    Integrates all ML, LLM, messaging, and analytics components
    """
    
    def __init__(self, config_path: str = "./config"):
        """Initialize the platform with all components"""
        logger.info("Initializing Agricultural Marketing Platform...")
        
        self.config = Config(config_path)
        
        # Initialize ML Models
        logger.info("Loading ML models...")
        self.trend_detector = TrendDetector()
        self.pest_detector = PestOutbreakDetector()
        self.engagement_predictor = EngagementPredictor()
        self.campaign_optimizer = CampaignOptimizer()
        self.crop_segmentation = CropSegmentationEngine()
        
        # Initialize LLM
        logger.info("Initializing LLM client...")
        self.llm_client = OllamaLLMClient(
            base_url=self.config.ollama_url,
            model=self.config.ollama_model
        )
        
        # Verify LLM connection
        if not self.llm_client.check_connection():
            logger.warning("Warning: Ollama service not responding. Running in degraded mode.")
        
        # Initialize RAG
        logger.info("Loading RAG retriever...")
        self.rag_retriever = RAGRetriever()
        
        # Initialize Content Generator
        self.content_generator = ContentGenerator(self.llm_client, self.rag_retriever)
        
        # Initialize Audio
        logger.info("Loading audio generation model...")
        self.audio_generator = HuggingFaceTextToAudio()
        
        # Initialize Messaging
        logger.info("Initializing messaging channels...")
        adb_controller = ADBWhatsAppController() if self.config.adb_enabled else None
        twilio_gateway = TwilioMessagingGateway(
            account_sid=self.config.twilio_account_sid,
            auth_token=self.config.twilio_auth_token,
            from_number=self.config.twilio_phone
        ) if self.config.twilio_enabled else None
        
        self.messenger = MessagingOrchestrator(
            adb_controller=adb_controller,
            twilio_gateway=twilio_gateway
        )
        
        # Initialize Campaign Service
        self.campaign_service = CampaignService(
            engagement_predictor=self.engagement_predictor,
            trend_detector=self.trend_detector,
            llm_client=self.llm_client,
            messaging_orchestrator=self.messenger,
            audio_generator=self.audio_generator
        )
        
        logger.info("Platform initialization complete!")
    
    def create_and_launch_campaign(self, campaign_config: Dict) -> Dict:
        """
        End-to-end campaign creation and launch
        """
        logger.info(f"Creating campaign: {campaign_config.get('name', 'Unnamed')}")
        
        # Create campaign
        campaign_result = self.campaign_service.create_campaign(campaign_config)
        
        if not campaign_result['success']:
            logger.error(f"Campaign creation failed: {campaign_result}")
            return campaign_result
        
        campaign_id = campaign_result['campaign_id']
        logger.info(f"Campaign created: {campaign_id}")
        
        # Get target farmer list
        target_farmers = self._get_target_farmers(campaign_config)
        logger.info(f"Targeting {len(target_farmers)} farmers")
        
        # Launch campaign
        launch_result = self.campaign_service.launch_campaign(campaign_id, target_farmers)
        
        return {
            'success': True,
            'campaign_id': campaign_id,
            'launch_result': launch_result
        }
    
    def _get_target_farmers(self, campaign_config: Dict) -> list:
        """
        Get list of farmers matching campaign targeting criteria
        In production, this would query a farmer database
        """
        # Placeholder - would connect to farmer database
        return []
    
    def analyze_campaign_performance(self, campaign_id: str) -> Dict:
        """
        Get comprehensive campaign performance analytics
        """
        return self.campaign_service.get_campaign_analytics(campaign_id)
    
    def get_pest_outbreak_predictions(self, region: str, crop: str) -> Dict:
        """
        Get pest outbreak risk predictions for region/crop
        """
        # This would fetch actual historical pest data and weather
        predictions = self.pest_detector.predict_outbreak_risk(
            historical_data={},
            weather_data={},
            region=region
        )
        return predictions

# Example usage
if __name__ == "__main__":
    # Initialize platform
    platform = AgriculturalMarketingPlatform()
    
    # Create example campaign
    campaign_config = {
        'name': 'Fungicide Campaign - Tamil Nadu Monsoon',
        'crop': 'rice',
        'product': 'Dhanuka Propiconazole',
        'region': 'Tamil Nadu',
        'target_segments': ['smallholder', 'medium_farmer'],
        'num_variants': 5,
        'budget': 50000,
        'scheduling': {
            'start_date': '2024-06-01',
            'duration_days': 14,
            'send_hours': [7, 12, 17]
        }
    }
    
    result = platform.create_and_launch_campaign(campaign_config)
    print(json.dumps(result, indent=2))