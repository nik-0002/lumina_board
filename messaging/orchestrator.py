import json
from typing import Dict, List
from datetime import datetime
from enum import Enum

class MessageChannel(Enum):
    WHATSAPP_ADB = "whatsapp_adb"
    WHATSAPP_API = "whatsapp_api"
    SMS = "sms"
    VOICE = "voice"

class MessagingOrchestrator:
    """
    Intelligent message routing across multiple channels
    Selects optimal channel based on farmer device, connectivity, and time
    """
    
    def __init__(self, adb_controller=None, twilio_gateway=None):
        self.adb = adb_controller
        self.twilio = twilio_gateway
        self.routing_rules = {}
        self.delivery_logs = []
    
    def route_campaign_message(self, farmer_context: Dict, message_content: Dict, 
                              campaign_id: str) -> Dict:
        """
        Route message through optimal channel based on farmer profile
        """
        optimal_channel = self._select_channel(farmer_context)
        
        delivery_result = self._send_via_channel(
            channel=optimal_channel,
            farmer_context=farmer_context,
            message_content=message_content,
            campaign_id=campaign_id
        )
        
        # Log delivery
        self.delivery_logs.append({
            'campaign_id': campaign_id,
            'farmer_id': farmer_context.get('farmer_id'),
            'channel': optimal_channel.value,
            'status': 'sent' if delivery_result['success'] else 'failed',
            'timestamp': datetime.now().isoformat(),
            'message_type': message_content.get('type'),
            'retry_count': 0
        })
        
        return delivery_result
    
    def _select_channel(self, farmer_context: Dict) -> MessageChannel:
        """
        Select optimal channel based on farmer characteristics
        Priority: Device type > Connectivity > Time of day > Language support
        """
        device_type = farmer_context.get('device_type', 'smartphone')
        connectivity = farmer_context.get('connectivity_level', 'medium')  # low/medium/high
        hour = datetime.now().hour
        language = farmer_context.get('language', 'hi')
        
        # Decision tree
        if device_type == 'smartphone' and connectivity in ['medium', 'high']:
            return MessageChannel.WHATSAPP_ADB
        
        if connectivity == 'low':
            return MessageChannel.SMS
        
        if hour >= 8 and hour <= 20:  # Business hours
            return MessageChannel.WHATSAPP_ADB
        
        # Default fallback
        return MessageChannel.SMS
    
    def _send_via_channel(self, channel: MessageChannel, farmer_context: Dict, 
                         message_content: Dict, campaign_id: str) -> Dict:
        """Execute sending via selected channel"""
        phone_number = farmer_context.get('phone_number')
        
        if channel == MessageChannel.WHATSAPP_ADB and self.adb:
            return self.adb.send_whatsapp_message(
                phone_number=phone_number,
                message=message_content['text'],
                media_path=message_content.get('media_path')
            )
        
        elif channel == MessageChannel.WHATSAPP_API and self.twilio:
            return self.twilio.send_whatsapp_message(
                to_number=phone_number,
                message_text=message_content['text'],
                media_urls=message_content.get('media_urls')
            )
        
        elif channel == MessageChannel.SMS and self.twilio:
            return self.twilio.send_sms(
                to_number=phone_number,
                message_text=message_content['text']
            )
        
        else:
            return {'success': False, 'error': 'Channel not configured'}
    
    def get_delivery_status(self, campaign_id: str) -> Dict:
        """Get delivery status for a campaign"""
        logs = [l for l in self.delivery_logs if l['campaign_id'] == campaign_id]
        
        successful = len([l for l in logs if l['status'] == 'sent'])
        total = len(logs)
        
        return {
            'campaign_id': campaign_id,
            'total_messages': total,
            'successful_delivery': successful,
            'delivery_rate': (successful / total * 100) if total > 0 else 0,
            'by_channel': self._group_by_channel(logs)
        }
    
    def _group_by_channel(self, logs: List[Dict]) -> Dict:
        """Group delivery logs by channel"""
        grouped = {}
        for log in logs:
            channel = log['channel']
            if channel not in grouped:
                grouped[channel] = {'sent': 0, 'failed': 0}
            
            if log['status'] == 'sent':
                grouped[channel]['sent'] += 1
            else:
                grouped[channel]['failed'] += 1
        
        return grouped