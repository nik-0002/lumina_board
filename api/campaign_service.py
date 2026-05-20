from typing import Dict, List
from datetime import datetime, timedelta
import json

class CampaignService:
    """
    Core campaign management service
    Handles campaign creation, optimization, and performance tracking
    """
    
    def __init__(self, engagement_predictor, trend_detector, llm_client, 
                 messaging_orchestrator, audio_generator):
        self.engagement_predictor = engagement_predictor
        self.trend_detector = trend_detector
        self.llm = llm_client
        self.messenger = messaging_orchestrator
        self.audio_gen = audio_generator
        self.campaigns = {}
        self.campaign_metrics = {}
    
    def create_campaign(self, campaign_config: Dict) -> Dict:
        """
        Create new marketing campaign with AI optimization
        """
        campaign_id = self._generate_campaign_id()
        
        # Get target farmer segments
        target_segments = campaign_config.get('target_segments', [])
        
        # Generate content variants for each segment
        content_variants = self._generate_content_variants(
            crop=campaign_config['crop'],
            product=campaign_config['product'],
            region=campaign_config['region'],
            segments=target_segments,
            num_variants=campaign_config.get('num_variants', 5)
        )
        
        campaign = {
            'campaign_id': campaign_id,
            'name': campaign_config.get('name', ''),
            'product': campaign_config['product'],
            'crop': campaign_config['crop'],
            'region': campaign_config['region'],
            'target_segments': target_segments,
            'content_variants': content_variants,
            'scheduling': campaign_config.get('scheduling', {}),
            'budget': campaign_config.get('budget', 0),
            'created_at': datetime.now().isoformat(),
            'status': 'draft',
            'performance': {
                'sent': 0,
                'delivered': 0,
                'engaged': 0,
                'converted': 0
            }
        }
        
        self.campaigns[campaign_id] = campaign
        self.campaign_metrics[campaign_id] = self._initialize_metrics()
        
        return {
            'success': True,
            'campaign_id': campaign_id,
            'campaign': campaign
        }
    
    def _generate_content_variants(self, crop: str, product: str, region: str,
                                   segments: List[Dict], num_variants: int) -> List[Dict]:
        """Generate multiple content variants for A/B testing"""
        variants = []
        
        for i in range(num_variants):
            # Generate base message
            message_result = self.llm.generate_content(
                f"Generate a marketing message variant {i+1} for {crop} farmers in {region} about {product}",
                temperature=0.5 + (i * 0.1)  # Vary temperature for diversity
            )
            
            variant = {
                'variant_id': i + 1,
                'type': self._determine_variant_type(i),
                'content': {
                    'text': message_result['content'],
                    'has_image': i % 3 == 0,
                    'has_audio': i % 4 == 0
                },
                'predicted_engagement': self._predict_variant_engagement(i),
                'ctr_estimate': 0.05 + (i * 0.02)
            }
            variants.append(variant)
        
        return variants
    
    def _determine_variant_type(self, index: int) -> str:
        """Determine content type for variant"""
        types = ['text_only', 'text_image', 'text_audio', 'video_script', 'voice_message']
        return types[index % len(types)]
    
    def _predict_variant_engagement(self, variant_index: int) -> float:
        """Estimate engagement for each variant"""
        base_engagement = 0.4
        variant_boost = (variant_index % 3) * 0.1
        return min(0.95, base_engagement + variant_boost)
    
    def _initialize_metrics(self) -> Dict:
        """Initialize campaign metrics tracker"""
        return {
            'daily_metrics': {},
            'channel_performance': {},
            'segment_performance': {},
            'engagement_by_variant': {},
            'conversion_funnel': {
                'sent': 0,
                'delivered': 0,
                'opened': 0,
                'clicked': 0,
                'converted': 0
            }
        }
    
    def launch_campaign(self, campaign_id: str, farmer_list: List[Dict]) -> Dict:
        """
        Launch campaign to farmer segments
        """
        if campaign_id not in self.campaigns:
            return {'success': False, 'error': 'Campaign not found'}
        
        campaign = self.campaigns[campaign_id]
        campaign['status'] = 'active'
        campaign['launched_at'] = datetime.now().isoformat()
        
        # Route messages to farmers
        delivery_results = []
        for farmer in farmer_list:
            # Select best variant for farmer
            selected_variant = self._select_variant_for_farmer(campaign['content_variants'], farmer)
            
            # Generate personalized content
            content = self._personalize_content(selected_variant, farmer, campaign)
            
            # Route message
            delivery = self.messenger.route_campaign_message(
                farmer_context=farmer,
                message_content=content,
                campaign_id=campaign_id
            )
            
            delivery_results.append(delivery)
            campaign['performance']['sent'] += 1
        
        return {
            'success': True,
            'campaign_id': campaign_id,
            'total_messages': len(farmer_list),
            'delivery_results': delivery_results,
            'launch_time': datetime.now().isoformat()
        }
    
    def _select_variant_for_farmer(self, variants: List[Dict], farmer: Dict) -> Dict:
        """Select best variant for individual farmer using MAB"""
        # Thompson sampling or epsilon-greedy
        variant_scores = []
        
        for variant in variants:
            score = variant['predicted_engagement']
            # Adjust based on farmer characteristics
            if variant['type'] == 'text_audio' and farmer.get('device_type') == 'feature_phone':
                score *= 1.3
            elif variant['type'] == 'video_script' and farmer.get('device_type') == 'smartphone':
                score *= 1.2
            
            variant_scores.append(score)
        
        best_variant_idx = max(range(len(variant_scores)), key=lambda i: variant_scores[i])
        return variants[best_variant_idx]
    
    def _personalize_content(self, variant: Dict, farmer: Dict, campaign: Dict) -> Dict:
        """Personalize content for specific farmer"""
        personalized = {
            'type': variant['type'],
            'text': variant['content']['text'].replace('{farmer_name}', farmer.get('name', 'Farmer')),
            'farmer_id': farmer.get('farmer_id'),
            'variant_id': variant['variant_id']
        }
        
        # Add media if needed
        if variant['content']['has_image']:
            personalized['image'] = f"/media/campaign_{campaign['campaign_id']}_image.jpg"
        
        if variant['content']['has_audio']:
            audio_result = self.audio_gen.generate_campaign_audio(
                personalized['text'],
                language=farmer.get('language', 'hi')
            )
            if audio_result['success']:
                personalized['audio_url'] = audio_result.get('audio_path')
        
        return personalized
    
    def get_campaign_analytics(self, campaign_id: str, time_period: str = '7d') -> Dict:
        """
        Get comprehensive campaign analytics
        """
        if campaign_id not in self.campaigns:
            return {'success': False, 'error': 'Campaign not found'}
        
        campaign = self.campaigns[campaign_id]
        metrics = self.campaign_metrics.get(campaign_id, {})
        
        # Calculate KPIs
        delivery_rate = (campaign['performance']['delivered'] / 
                        campaign['performance']['sent']) if campaign['performance']['sent'] > 0 else 0
        engagement_rate = (campaign['performance']['engaged'] / 
                          campaign['performance']['delivered']) if campaign['performance']['delivered'] > 0 else 0
        conversion_rate = (campaign['performance']['converted'] / 
                          campaign['performance']['engaged']) if campaign['performance']['engaged'] > 0 else 0
        
        return {
            'campaign_id': campaign_id,
            'campaign_name': campaign['name'],
            'status': campaign['status'],
            'period': time_period,
            'performance_metrics': {
                'total_sent': campaign['performance']['sent'],
                'delivery_rate': round(delivery_rate * 100, 2),
                'engagement_rate': round(engagement_rate * 100, 2),
                'conversion_rate': round(conversion_rate * 100, 2),
                'conversions': campaign['performance']['converted']
            },
            'channel_breakdown': metrics.get('channel_performance', {}),
            'segment_breakdown': metrics.get('segment_performance', {}),
            'top_variants': self._get_top_variants(campaign_id),
            'daily_trend': metrics.get('daily_metrics', {})
        }
    
    def _get_top_variants(self, campaign_id: str, top_n: int = 3) -> List[Dict]:
        """Get top performing variants"""
        campaign = self.campaigns[campaign_id]
        variants_with_perf = []
        
        for variant in campaign['content_variants']:
            perf = self.campaign_metrics[campaign_id].get('engagement_by_variant', {}).get(
                variant['variant_id'], 0
            )
            variants_with_perf.append({
                'variant_id': variant['variant_id'],
                'type': variant['type'],
                'engagement_score': perf
            })
        
        return sorted(variants_with_perf, key=lambda x: x['engagement_score'], reverse=True)[:top_n]
    
    def _generate_campaign_id(self) -> str:
        """Generate unique campaign ID"""
        import uuid
        return f"camp_{uuid.uuid4().hex[:8]}"