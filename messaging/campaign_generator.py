"""
Lumina Board - Multilingual Campaign Message Generator
Generates SMS messages and audio scripts for Indian agricultural campaigns
in multiple Indian languages using Qwen2.5 LLM with template fallback.

Languages supported:
Hindi, Marathi, Punjabi, Telugu, Tamil, Kannada, Bengali, Gujarati,
Odia, Malayalam, Assamese, English

No hallucination: product/crop/state data comes from CSV, language selection
from actual grower language distribution.
"""

import os
import logging
from typing import Dict, List, Optional
import requests

logger = logging.getLogger("lumina.messaging")

# ─── Language Config ──────────────────────────────────────────────────────────
LANGUAGE_META = {
    "Hindi": {
        "code": "hi",
        "script": "Devanagari",
        "greeting": "नमस्ते किसान भाई",
        "sms_limit": 160,
        "audio_wpm": 90  # words per minute for audio scripts
    },
    "Marathi": {
        "code": "mr",
        "script": "Devanagari",
        "greeting": "नमस्कार शेतकरी बंधू",
        "sms_limit": 160,
        "audio_wpm": 85
    },
    "Punjabi": {
        "code": "pa",
        "script": "Gurmukhi",
        "greeting": "ਸਤ ਸ੍ਰੀ ਅਕਾਲ ਕਿਸਾਨ ਜੀ",
        "sms_limit": 160,
        "audio_wpm": 90
    },
    "Telugu": {
        "code": "te",
        "script": "Telugu",
        "greeting": "నమస్కారం రైతు అన్నా",
        "sms_limit": 160,
        "audio_wpm": 85
    },
    "Tamil": {
        "code": "ta",
        "script": "Tamil",
        "greeting": "வணக்கம் விவசாயி தாத்தா",
        "sms_limit": 160,
        "audio_wpm": 85
    },
    "Kannada": {
        "code": "kn",
        "script": "Kannada",
        "greeting": "ನಮಸ್ಕಾರ ರೈತ ಬಂಧು",
        "sms_limit": 160,
        "audio_wpm": 80
    },
    "Bengali": {
        "code": "bn",
        "script": "Bengali",
        "greeting": "নমস্কার কৃষক ভাই",
        "sms_limit": 160,
        "audio_wpm": 90
    },
    "Gujarati": {
        "code": "gu",
        "script": "Gujarati",
        "greeting": "નમસ્તે ખેડૂત ભાઈ",
        "sms_limit": 160,
        "audio_wpm": 85
    },
    "Odia": {
        "code": "or",
        "script": "Odia",
        "greeting": "ନମସ୍କାର ଚାଷୀ ଭାଇ",
        "sms_limit": 160,
        "audio_wpm": 80
    },
    "Malayalam": {
        "code": "ml",
        "script": "Malayalam",
        "greeting": "നമസ്കാരം കർഷക സഹോദരാ",
        "sms_limit": 160,
        "audio_wpm": 80
    },
    "English": {
        "code": "en",
        "script": "Latin",
        "greeting": "Dear Farmer",
        "sms_limit": 160,
        "audio_wpm": 130
    }
}

# ─── Campaign Templates (fallback when LLM unavailable) ──────────────────────
TEMPLATES = {
    "product_launch": {
        "Hindi": (
            "{greeting},\nSyngenta का नया उत्पाद {product} अब {state} में उपलब्ध है।"
            " {crop} के लिए बेहतरीन सुरक्षा। अपने नजदीकी डीलर से संपर्क करें।\nSyngenta India"
        ),
        "Marathi": (
            "{greeting},\nSyngenta चे नवीन उत्पादन {product} आता {state} मध्ये उपलब्ध आहे।"
            " {crop} साठी उत्कृष्ट संरक्षण। जवळच्या डीलरशी संपर्क साधा।\nSyngenta India"
        ),
        "Punjabi": (
            "{greeting},\nSyngenta ਦਾ ਨਵਾਂ ਉਤਪਾਦ {product} ਹੁਣ {state} ਵਿੱਚ ਉਪਲਬਧ ਹੈ।"
            " {crop} ਲਈ ਵਧੀਆ ਸੁਰੱਖਿਆ। ਆਪਣੇ ਨੇੜੇ ਦੇ ਡੀਲਰ ਨਾਲ ਸੰਪਰਕ ਕਰੋ।"
        ),
        "Telugu": (
            "{greeting},\nSyngenta యొక్క కొత్త ఉత్పత్తి {product} ఇప్పుడు {state}లో అందుబాటులో ఉంది।"
            " {crop} కోసం అద్భుతమైన రక్షణ। మీ సమీపంలోని డీలర్‌ని సంప్రదించండి।"
        ),
        "Tamil": (
            "{greeting},\nSyngenta இன் புதிய தயாரிப்பு {product} இப்போது {state}ல் கிடைக்கிறது।"
            " {crop} க்கு சிறந்த பாதுகாப்பு। உங்கள் அருகிலுள்ள டீலரை தொடர்பு கொள்ளுங்கள்।"
        ),
        "Kannada": (
            "{greeting},\nSyngenta ಯ ಹೊಸ ಉತ್ಪನ್ನ {product} ಈಗ {state}ದಲ್ಲಿ ಲಭ್ಯವಿದೆ।"
            " {crop} ಗಾಗಿ ಅತ್ಯುತ್ತಮ ರಕ್ಷಣೆ। ನಿಮ್ಮ ಹತ್ತಿರದ ಡೀಲರ್ ಅನ್ನು ಸಂಪರ್ಕಿಸಿ।"
        ),
        "Bengali": (
            "{greeting},\nSyngenta-র নতুন পণ্য {product} এখন {state}-এ পাওয়া যাচ্ছে।"
            " {crop}-এর জন্য দুর্দান্ত সুরক্ষা। আপনার কাছের ডিলারের সাথে যোগাযোগ করুন।"
        ),
        "Gujarati": (
            "{greeting},\nSyngenta નું નવું ઉત્પાદ {product} હવે {state}માં ઉપલબ્ધ છે।"
            " {crop} માટે શ્રેષ્ઠ સુરક્ષા। તમારા નજીકના ડીલરને સંપર્ક કરો।"
        ),
        "English": (
            "{greeting},\nSyngenta's new product {product} is now available in {state}."
            " Excellent protection for {crop}. Contact your nearest dealer today.\nSyngenta India"
        )
    },
    "urgency_offer": {
        "Hindi": (
            "{greeting},\nसीमित समय ऑफर! {product} पर विशेष छूट।"
            " {crop} की सुरक्षा के लिए अभी संपर्क करें। ऑफर {state} में मात्र 3 दिन।\nSyngenta"
        ),
        "English": (
            "{greeting},\nLimited time offer! Special discount on {product}."
            " Protect your {crop} crop now. Offer valid in {state} for 3 days only.\nSyngenta"
        ),
        "Marathi": (
            "{greeting},\nमर्यादित वेळाची ऑफर! {product} वर विशेष सूट।"
            " {crop} संरक्षणासाठी आत्ताच संपर्क साधा. {state} मध्ये फक्त ३ दिवस.\nSyngenta"
        ),
        "Punjabi": (
            "{greeting},\nਸੀਮਤ ਸਮੇਂ ਦੀ ਪੇਸ਼ਕਸ਼! {product} 'ਤੇ ਵਿਸ਼ੇਸ਼ ਛੋਟ।"
            " {crop} ਸੁਰੱਖਿਆ ਲਈ ਹੁਣੇ ਸੰਪਰਕ ਕਰੋ। {state} ਵਿੱਚ ਸਿਰਫ਼ 3 ਦਿਨ।"
        ),
        "Telugu": (
            "{greeting},\nపరిమిత కాల ఆఫర్! {product}పై ప్రత్యేక తగ్గింపు।"
            " {crop} రక్షణకు ఇప్పుడే సంప్రదించండి। {state}లో 3 రోజులు మాత్రమే।"
        ),
        "Tamil": (
            "{greeting},\nவரையறுக்கப்பட்ட நேர சலுகை! {product}பை சிறப்பு தள்ளுபடி।"
            " {crop} பாதுகாப்பிற்கு இப்போதே தொடர்பு கொள்ளுங்கள்। {state}ல் 3 நாட்கள் மட்டுமே।"
        ),
        "Kannada": (
            "{greeting},\nಸೀಮಿತ ಸಮಯದ ಆಫರ್! {product}ಗೆ ವಿಶೇಷ ರಿಯಾಯಿತಿ।"
            " {crop} ರಕ್ಷಣೆಗಾಗಿ ಈಗಲೇ ಸಂಪರ್ಕಿಸಿ। {state}ದಲ್ಲಿ 3 ದಿನಗಳು ಮಾತ್ರ।"
        ),
        "Bengali": (
            "{greeting},\nসীমিত সময়ের অফার! {product}-এ বিশেষ ছাড়।"
            " {crop} সুরক্ষার জন্য এখনই যোগাযোগ করুন। {state}-এ মাত্র ৩ দিন।"
        ),
        "Gujarati": (
            "{greeting},\nમર્યાદિત સમય ઓફર! {product} પર વિશેષ છૂટ।"
            " {crop} સુરક્ષા માટે અત્યારે સંપર્ક કરો। {state}માં માત્ર ૩ દિવસ।"
        )
    },
    "season_reminder": {
        "Hindi": (
            "{greeting},\n{crop} की बुवाई का सही समय आ गया है।"
            " Syngenta के {product} से फसल सुरक्षित रखें। {state} के किसान आज ही डीलर से मिलें।"
        ),
        "English": (
            "{greeting},\nIt's the right time to sow {crop}."
            " Protect your crop with Syngenta's {product}. Meet your dealer in {state} today."
        ),
        "Marathi": (
            "{greeting},\n{crop} पेरणीचा योग्य वेळ आला आहे।"
            " Syngenta च्या {product} ने पीक सुरक्षित ठेवा। {state} मधील शेतकरी आज डीलरला भेटा।"
        ),
        "Punjabi": (
            "{greeting},\n{crop} ਬੀਜਣ ਦਾ ਸਹੀ ਸਮਾਂ ਆ ਗਿਆ ਹੈ।"
            " Syngenta ਦੇ {product} ਨਾਲ ਫਸਲ ਸੁਰੱਖਿਅਤ ਰੱਖੋ। {state} ਦੇ ਕਿਸਾਨ ਅੱਜ ਡੀਲਰ ਨਾਲ ਮਿਲੋ।"
        ),
        "Telugu": (
            "{greeting},\n{crop} విత్తడానికి సరైన సమయం వచ్చింది।"
            " Syngenta యొక్క {product}తో పంట సురక్షితంగా ఉంచుకోండి। {state} రైతులు నేడే డీలర్‌ని కలవండి।"
        ),
        "Tamil": (
            "{greeting},\n{crop} விதைக்க சரியான நேரம் வந்துவிட்டது।"
            " Syngenta இன் {product} மூலம் பயிரை பாதுகாக்கவும்। {state} விவசாயிகள் இன்றே டீலரை சந்தியுங்கள்।"
        ),
        "Kannada": (
            "{greeting},\n{crop} ಬಿತ್ತನೆಯ ಸರಿಯಾದ ಸಮಯ ಬಂದಿದೆ।"
            " Syngenta ಯ {product}ನಿಂದ ಬೆಳೆ ರಕ್ಷಿಸಿಕೊಳ್ಳಿ। {state} ರೈತರು ಇಂದೇ ಡೀಲರ್ ಅನ್ನು ಭೇಟಿ ಮಾಡಿ।"
        ),
        "Bengali": (
            "{greeting},\n{crop} বপনের সঠিক সময় এসেছে।"
            " Syngenta-র {product} দিয়ে ফসল সুরক্ষিত রাখুন। {state}-এর কৃষকরা আজই ডিলারের সাথে দেখা করুন।"
        ),
        "Gujarati": (
            "{greeting},\n{crop} વાવণીનો સાચો સમય આવ્યો છે।"
            " Syngenta ના {product} થી પાક સુરક્ષિત રાખો। {state} ના ખેડૂતો આજે જ ડીલરને મળો।"
        )
    }
}

AUDIO_SCRIPT_TEMPLATES = {
    "Hindi": """
[ऑडियो स्क्रिप्ट - {duration} सेकंड]
[शुरुआत - उत्साहवर्धक संगीत]

"किसान भाइयों और बहनों, नमस्कार!

{campaign_body}

याद रखें — Syngenta के साथ आपकी फसल, आपका भविष्य सुरक्षित है।
अधिक जानकारी के लिए अपने नजदीकी Syngenta डीलर से संपर्क करें।

सिंजेंटा इंडिया — किसान का साथी।"

[संगीत - फेड आउट]
""",
    "English": """
[AUDIO SCRIPT - {duration} seconds]
[Intro - uplifting agricultural jingle]

"Hello farmers of {state}!

{campaign_body}

Remember — with Syngenta, your crop and future are protected.
For more information, contact your nearest Syngenta dealer.

Syngenta India — Your Farming Partner."

[Music - fade out]
""",
    "Marathi": """
[ऑडिओ स्क्रिप्ट - {duration} सेकंद]
[सुरुवात - उत्साहवर्धक संगीत]

"शेतकरी बंधू आणि भगिनींनो, नमस्कार!

{campaign_body}

लक्षात ठेवा — Syngenta सोबत तुमचे पीक, तुमचे भविष्य सुरक्षित आहे.
अधिक माहितीसाठी तुमच्या जवळच्या Syngenta डीलरशी संपर्क साधा.

सिंजेंटा इंडिया — शेतकऱ्याचा साथीदार।"

[संगीत - फेड आउट]
""",
    "Punjabi": """
[ਆਡੀਓ ਸਕ੍ਰਿਪਟ - {duration} ਸਕਿੰਟ]
[ਸ਼ੁਰੂਆਤ - ਉਤਸ਼ਾਹਜਨਕ ਸੰਗੀਤ]

"ਕਿਸਾਨ ਭੈਣਾਂ ਅਤੇ ਭਰਾਵੋ, ਸਤ ਸ੍ਰੀ ਅਕਾਲ!

{campaign_body}

ਯਾਦ ਰੱਖੋ — Syngenta ਨਾਲ ਤੁਹਾਡੀ ਫਸਲ, ਤੁਹਾਡਾ ਭਵਿੱਖ ਸੁਰੱਖਿਅਤ ਹੈ।
ਹੋਰ ਜਾਣਕਾਰੀ ਲਈ ਆਪਣੇ ਨੇੜੇ ਦੇ Syngenta ਡੀਲਰ ਨਾਲ ਸੰਪਰਕ ਕਰੋ।

Syngenta India — ਕਿਸਾਨ ਦਾ ਸਾਥੀ।"

[ਸੰਗੀਤ - ਫੇਡ ਆਊਟ]
"""
}


class CampaignMessageGenerator:
    """
    Generates multilingual SMS messages and audio scripts.
    Uses Qwen2.5 for high-quality generation; falls back to templates.
    """

    def __init__(self):
        self.ollama_base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        self.model = os.environ.get("LLM_MODEL", "qwen2.5:7b")

    def generate_multilingual(
        self,
        campaign_type: str,
        product: str,
        crop: str,
        state: str,
        languages: List[str],
        context: str = "",
        segment_stats: Optional[Dict] = None
    ) -> Dict[str, Dict]:
        """
        Generate SMS + audio script for each language.
        Returns: {language: {sms, audio_script, char_count, estimated_duration_sec}}
        """
        results = {}

        for lang in languages:
            if lang not in LANGUAGE_META:
                logger.warning(f"Unsupported language: {lang}, skipping")
                continue

            try:
                sms = self._generate_sms(campaign_type, product, crop, state, lang, context, segment_stats)
                audio = self._generate_audio_script(campaign_type, product, crop, state, lang, sms)

                results[lang] = {
                    "sms": sms,
                    "audio_script": audio,
                    "char_count": len(sms),
                    "sms_parts": max(1, -(-len(sms) // 160)),  # ceil division
                    "estimated_audio_duration_sec": self._estimate_duration(audio, lang),
                    "script": LANGUAGE_META[lang]["script"],
                    "language_code": LANGUAGE_META[lang]["code"]
                }
            except Exception as e:
                logger.error(f"Generation failed for {lang}: {e}")
                results[lang] = {"error": str(e)}

        return results

    def _generate_sms(
        self,
        campaign_type: str,
        product: str,
        crop: str,
        state: str,
        lang: str,
        context: str,
        segment_stats: Optional[Dict]
    ) -> str:
        """Generate SMS using LLM or template fallback."""
        # Try LLM first
        llm_result = self._llm_generate_sms(campaign_type, product, crop, state, lang, context, segment_stats)
        if llm_result:
            return llm_result

        # Template fallback
        return self._template_sms(campaign_type, product, crop, state, lang)

    def _llm_generate_sms(self, campaign_type, product, crop, state, lang, context, segment_stats) -> Optional[str]:
        """Call Qwen2.5 to generate SMS."""
        meta = LANGUAGE_META.get(lang, {})
        stats_str = ""
        if segment_stats:
            stats_str = (
                f"Target segment: {segment_stats.get('total_growers', '?')} growers, "
                f"avg farm {segment_stats.get('avg_farm_size', '?')} acres, "
                f"dominant device: {segment_stats.get('dominant_device', '?')}"
            )

        prompt = f"""Generate a compelling SMS message for an Indian agricultural campaign.

Campaign Type: {campaign_type}
Product: {product}
Crop: {crop}
State/Region: {state}
Target Language: {lang}
{stats_str}
{f"Additional Context: {context}" if context else ""}

REQUIREMENTS:
- Write ONLY in {lang} language (use native script)
- Maximum 160 characters (SMS limit)
- Start with: {meta.get('greeting', 'Dear Farmer')}
- Be specific about {product} and {crop}
- Include a clear call to action
- Sound natural and relatable to Indian farmers
- No English mixing unless absolutely necessary
- End with "Syngenta India"

Output ONLY the SMS text, nothing else."""

        try:
            resp = requests.post(
                f"{self.ollama_base}/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "max_tokens": 200
                },
                timeout=30
            )
            resp.raise_for_status()
            sms = resp.json()["choices"][0]["message"]["content"].strip()
            # Trim if over limit
            if len(sms) > 320:  # 2 SMS parts max
                sms = sms[:317] + "..."
            return sms
        except Exception:
            return None

    def _template_sms(self, campaign_type: str, product: str, crop: str, state: str, lang: str) -> str:
        """Use pre-built template."""
        templates = TEMPLATES.get(campaign_type, TEMPLATES.get("product_launch", {}))
        # Find template for language or fall back to English
        template = templates.get(lang) or templates.get("English", "{greeting}, Syngenta {product} for {crop} in {state}.")

        meta = LANGUAGE_META.get(lang, {})
        return template.format(
            greeting=meta.get("greeting", "Dear Farmer"),
            product=product or "our product",
            crop=crop or "your crop",
            state=state or "your region"
        )

    def _generate_audio_script(
        self, campaign_type: str, product: str, crop: str, state: str, lang: str, sms: str
    ) -> str:
        """Generate 30-60 second audio/IVR script based on SMS content."""
        # Try LLM
        llm_audio = self._llm_generate_audio(product, crop, state, lang, sms)
        if llm_audio:
            return llm_audio

        # Template fallback
        base_template = AUDIO_SCRIPT_TEMPLATES.get(lang) or AUDIO_SCRIPT_TEMPLATES.get("English", "")
        campaign_body = self._expand_sms_to_audio_body(sms, lang, product, crop)

        return base_template.format(
            duration=45,
            state=state or "India",
            campaign_body=campaign_body
        )

    def _llm_generate_audio(self, product, crop, state, lang, sms_text) -> Optional[str]:
        """Call Qwen2.5 for audio script generation."""
        prompt = f"""Create a 30-45 second IVR/radio audio script in {lang} for Indian farmers.

Based on this SMS campaign:
"{sms_text}"

Product: {product}, Crop: {crop}, State: {state}

REQUIREMENTS:
- Write in {lang} language (native script)  
- Include [intro music] and [outro music] stage directions
- Warm, trusted voice tone — like talking to a neighbor
- Natural speech rhythm — not too fast
- 30-45 seconds when read aloud at normal pace (~{LANGUAGE_META.get(lang, {}).get('audio_wpm', 90)} wpm)
- Include: greeting → problem (pest/disease/yield risk) → solution ({product}) → call to action → Syngenta tagline
- Stage directions in [square brackets]

Output only the script text."""

        try:
            resp = requests.post(
                f"{self.ollama_base}/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.6,
                    "max_tokens": 400
                },
                timeout=40
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            return None

    def _expand_sms_to_audio_body(self, sms: str, lang: str, product: str, crop: str) -> str:
        """Expand SMS to 3-4 sentence audio body."""
        # Simple expansion: repeat key points with pauses
        return sms.replace("।", "।\n").replace(".", ".\n")

    def _estimate_duration(self, script: str, lang: str) -> int:
        """Estimate audio duration in seconds based on word count."""
        wpm = LANGUAGE_META.get(lang, {}).get("audio_wpm", 90)
        # Rough word count (works for both Latin and Indic scripts)
        word_count = len(script.split())
        # Subtract stage directions (in brackets)
        import re
        clean = re.sub(r'\[.*?\]', '', script)
        clean_words = len(clean.split())
        return max(20, int(clean_words / wpm * 60))