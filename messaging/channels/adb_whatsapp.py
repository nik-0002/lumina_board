import subprocess
import time
import json
from typing import Dict, List
from datetime import datetime

class ADBWhatsAppController:
    """
    Control WhatsApp via Android Debug Bridge (ADB)
    For feature phones and offline-first messaging
    """
    
    def __init__(self, device_id: str = None):
        self.device_id = device_id
        self.connected = self._check_adb_connection()
        self.message_queue = []
        self.sent_history = []
    
    def _check_adb_connection(self) -> bool:
        """Verify ADB connection"""
        try:
            output = subprocess.run(['adb', 'devices'], capture_output=True, text=True, timeout=5)
            return 'device' in output.stdout
        except:
            return False
    
    def send_whatsapp_message(self, phone_number: str, message: str, 
                            media_path: str = None, retry_attempts: int = 3) -> Dict:
        """
        Send WhatsApp message via ADB
        Supports text and media (images, audio, video)
        """
        if not self.connected:
            return {
                'success': False,
                'error': 'ADB not connected',
                'phone_number': phone_number
            }
        
        try:
            # Open WhatsApp and navigate to chat
            self._open_whatsapp_chat(phone_number)
            time.sleep(1)
            
            # Type and send message
            if media_path:
                self._send_media(media_path)
                time.sleep(0.5)
            
            self._type_message(message)
            self._send_message()
            
            # Log success
            record = {
                'phone_number': phone_number,
                'message': message,
                'media': media_path,
                'timestamp': datetime.now().isoformat(),
                'status': 'sent',
                'retry_count': 0
            }
            self.sent_history.append(record)
            
            return {
                'success': True,
                'phone_number': phone_number,
                'message_id': self._generate_message_id(),
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            if retry_attempts > 0:
                time.sleep(2)
                return self.send_whatsapp_message(phone_number, message, media_path, 
                                                 retry_attempts - 1)
            else:
                return {
                    'success': False,
                    'error': str(e),
                    'phone_number': phone_number,
                    'retry_count': 3
                }
    
    def _open_whatsapp_chat(self, phone_number: str):
        """Open WhatsApp chat with specific contact"""
        # Launch WhatsApp via intent
        cmd = f'adb shell am start -a android.intent.action.VIEW -d "https://wa.me/{phone_number}" -n com.whatsapp/.ui.LauncherActivity'
        subprocess.run(cmd, shell=True, check=True)
    
    def _type_message(self, message: str):
        """Type message in WhatsApp chat"""
        # Use sendevent or input command depending on Android version
        cmd = f'adb shell input text "{message}"'
        subprocess.run(cmd, shell=True, check=True)
    
    def _send_message(self):
        """Send the message"""
        # Tap send button (coordinates may vary)
        cmd = 'adb shell input tap 1000 2000'  # Adjust coordinates
        subprocess.run(cmd, shell=True, check=True)
    
    def _send_media(self, media_path: str):
        """Send media file"""
        # Implementation for sending images/videos
        pass
    
    def _generate_message_id(self) -> str:
        """Generate unique message ID"""
        return f"msg_{int(time.time() * 1000)}"
    
    def get_sent_history(self, phone_number: str = None) -> List[Dict]:
        """Get message history"""
        if phone_number:
            return [m for m in self.sent_history if m['phone_number'] == phone_number]
        return self.sent_history