#!/usr/bin/env python3
"""
🌍 Multilingual Ethical Growth Gating Service - IMPROVED THAI SUPPORT
✅ Uses Ollama LLM with better multilingual prompts
✅ Enhanced Thai language classification
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import re
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor
import json
import httpx
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Ethical Growth Gating Service")

# ============================================================
# OLLAMA CONFIGURATION
# ============================================================

OLLAMA_URL = os.getenv("OLLAMA_EXTERNAL_URL", "http://ollama.ollama.svc.cluster.local:11434")
EMBEDDING_MODEL = "nomic-embed-text"  # 768 dimensions
LLM_MODEL = "tinyllama"  # For classification

# ============================================================
# IMPROVED MULTILINGUAL CLASSIFICATION
# ============================================================

async def classify_with_llm(text: str, lang: str) -> Dict:
    """Use Ollama LLM to classify memory with UNIVERSAL multilingual support"""
    
    # ✅ UNIVERSAL: Single multilingual prompt that works for ALL languages
    prompt = f"""You are an ethical growth analyst. You understand ALL languages including English, Thai, Chinese, Japanese, Korean, Spanish, French, German, Arabic, Hindi, and more.

Analyze this text in its original language and respond ONLY with valid JSON.

Text: "{text}"
Language detected: {lang.upper()}

Classify into ONE category. Consider cultural context and language-specific expressions:

Categories (universal across all languages):
- growth_memory: Positive emotions, gratitude, spiritual/religious growth, faith, love, learning, appreciation, thankfulness, worship, devotion, nature appreciation, kindness
- challenge_memory: Negative emotions, aggression, violence, anger, conflict, harm, hatred, destruction, revenge, hostility
- wisdom_moment: Deep philosophical reflection, insights, enlightenment, meditation, contemplation, self-discovery, transcendence
- needs_support: Crisis, despair, self-harm thoughts, severe distress, hopelessness, suicidal ideation
- neutral_interaction: Everyday conversation, neutral statements, factual information, casual chat

Important notes:
- Religious/spiritual content (God, Buddha, Allah, prayer, worship, meditation) = growth_memory or wisdom_moment
- Nature appreciation (trees, sea, mountains, beauty) = growth_memory
- Expressions of love/gratitude = growth_memory
- Violence/harm words = challenge_memory
- Philosophical reflections = wisdom_moment

Provide ethical scores (0.0-1.0) for each dimension based on the content's intent and emotion.

Respond with ONLY this JSON (no markdown, no explanatory text):
{{
  "classification": "category_name",
  "self_awareness": 0.0-1.0,
  "emotional_regulation": 0.0-1.0,
  "compassion": 0.0-1.0,
  "integrity": 0.0-1.0,
  "growth_mindset": 0.0-1.0,
  "wisdom": 0.0-1.0,
  "transcendence": 0.0-1.0,
  "reasoning": "brief explanation in English"
}}"""
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": LLM_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "top_p": 0.9,
                    }
                }
            )
            
            if response.status_code != 200:
                logger.error(f"LLM classification error: {response.status_code}")
                return get_fallback_classification(text, lang)
            
            data = response.json()
            llm_response = data.get("response", "")
            
            # Extract JSON from response
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', llm_response)
            if json_match:
                result = json.loads(json_match.group())
                
                # Validate classification
                valid_classifications = [
                    'growth_memory', 'challenge_memory', 'wisdom_moment', 
                    'needs_support', 'neutral_interaction'
                ]
                
                if result.get('classification') not in valid_classifications:
                    result['classification'] = 'neutral_interaction'
                
                # Ensure all scores are present and valid
                for key in ['self_awareness', 'emotional_regulation', 'compassion', 
                           'integrity', 'growth_mindset', 'wisdom', 'transcendence']:
                    if key not in result or not isinstance(result[key], (int, float)):
                        result[key] = 0.5
                    result[key] = max(0.0, min(1.0, float(result[key])))
                
                logger.info(f"✅ LLM classified as: {result['classification']}")
                return result
            else:
                logger.warning("⚠️ Could not parse LLM JSON response")
                return get_fallback_classification(text, lang)
                
    except Exception as e:
        logger.error(f"❌ LLM classification error: {e}")
        return get_fallback_classification(text, lang)

def get_fallback_classification(text: str, lang: str) -> Dict:
    """UNIVERSAL fallback with multilingual keyword detection"""
    text_lower = text.lower()
    
    # ✅ MULTILINGUAL: Universal keywords across languages
    
    # Growth/Positive keywords (multilingual)
    growth_keywords = {
        'en': ['love', 'thank', 'grateful', 'learn', 'improve', 'grow', 'appreciate', 'god', 'buddha', 'jesus', 'allah', 'prayer', 'worship', 'nature', 'beautiful', 'tree', 'mountain', 'sea', 'kind', 'help', 'compassion'],
        'th': ['รัก', 'ขอบคุณ', 'กตัญญู', 'เรียนรู้', 'พัฒนา', 'เติบโต', 'พระพุทธเจ้า', 'พระ', 'ธรรม', 'บูชา', 'สวดมนต์', 'ทำบุญ', 'ธรรมชาติ', 'ต้นไม้', 'ภูเขา', 'ทะเล', 'สวยงาม', 'ซาบซึ้ง', 'ดีงาม', 'ใจดี', 'เมตตา', 'กรุณา'],
        'zh': ['爱', '感谢', '感恩', '学习', '成长', '进步', '佛', '上帝', '祷告', '冥想', '自然', '美丽', '树', '山', '海', '善良', '帮助', '慈悲'],
        'ja': ['愛', '感謝', '学ぶ', '成長', '仏', '神', '祈り', '瞑想', '自然', '美しい', '木', '山', '海', '優しい', '助ける', '慈悲'],
        'ko': ['사랑', '감사', '배우다', '성장', '부처', '하나님', '기도', '명상', '자연', '아름다운', '나무', '산', '바다', '친절', '돕다', '자비'],
        'es': ['amor', 'gracias', 'agradecer', 'aprender', 'crecer', 'mejorar', 'dios', 'jesús', 'oración', 'rezar', 'naturaleza', 'hermoso', 'árbol', 'montaña', 'mar', 'amable', 'ayudar', 'compasión'],
        'fr': ['amour', 'merci', 'reconnaissant', 'apprendre', 'grandir', 'améliorer', 'dieu', 'jésus', 'prière', 'prier', 'nature', 'beau', 'arbre', 'montagne', 'mer', 'gentil', 'aider', 'compassion'],
        'de': ['liebe', 'danke', 'dankbar', 'lernen', 'wachsen', 'verbessern', 'gott', 'jesus', 'gebet', 'beten', 'natur', 'schön', 'baum', 'berg', 'meer', 'freundlich', 'helfen', 'mitgefühl'],
        'ar': ['حب', 'شكر', 'ممتن', 'تعلم', 'نمو', 'تحسن', 'الله', 'صلاة', 'دعاء', 'طبيعة', 'جميل', 'شجرة', 'جبل', 'بحر', 'لطيف', 'مساعدة', 'رحمة'],
        'hi': ['प्यार', 'धन्यवाद', 'आभारी', 'सीखना', 'बढ़ना', 'सुधार', 'भगवान', 'प्रार्थना', 'पूजा', 'प्रकृति', 'सुंदर', 'पेड़', 'पहाड़', 'समुद्र', 'दयालु', 'मदद', 'करुणा'],
        'pt': ['amor', 'obrigado', 'grato', 'aprender', 'crescer', 'melhorar', 'deus', 'jesus', 'oração', 'rezar', 'natureza', 'bonito', 'árvore', 'montanha', 'mar', 'gentil', 'ajudar', 'compaixão'],
        'ru': ['любовь', 'спасибо', 'благодарен', 'учиться', 'расти', 'улучшать', 'бог', 'иисус', 'молитва', 'молиться', 'природа', 'красивый', 'дерево', 'гора', 'море', 'добрый', 'помогать', 'сострадание'],
        'it': ['amore', 'grazie', 'grato', 'imparare', 'crescere', 'migliorare', 'dio', 'gesù', 'preghiera', 'pregare', 'natura', 'bello', 'albero', 'montagna', 'mare', 'gentile', 'aiutare', 'compassione'],
    }
    
    # Challenge/Negative keywords (multilingual)
    challenge_keywords = {
        'en': ['kill', 'murder', 'hurt', 'harm', 'attack', 'hate', 'destroy', 'revenge', 'violent', 'angry', 'rage', 'fight'],
        'th': ['ฆ่า', 'ทำร้าย', 'โกรธ', 'เกลียด', 'ทำลาย', 'ร้าย', 'แก้แค้น', 'รุนแรง', 'ต่อสู้', 'โกง', 'หลอกลวง'],
        'zh': ['杀', '谋杀', '伤害', '攻击', '恨', '毁灭', '报复', '暴力', '愤怒', '打架'],
        'ja': ['殺す', '殺人', '傷つける', '攻撃', '憎む', '破壊', '復讐', '暴力', '怒り', '戦う'],
        'ko': ['죽이다', '살인', '해치다', '공격', '미워하다', '파괴', '복수', '폭력', '분노', '싸우다'],
        'es': ['matar', 'asesinar', 'herir', 'dañar', 'atacar', 'odiar', 'destruir', 'venganza', 'violento', 'enojado'],
        'fr': ['tuer', 'assassiner', 'blesser', 'nuire', 'attaquer', 'haïr', 'détruire', 'vengeance', 'violent', 'en colère'],
        'de': ['töten', 'morden', 'verletzen', 'schaden', 'angreifen', 'hassen', 'zerstören', 'rache', 'gewalttätig', 'wütend'],
        'ar': ['قتل', 'جريمة', 'إيذاء', 'ضرر', 'هجوم', 'كراهية', 'تدمير', 'انتقام', 'عنف', 'غضب'],
        'hi': ['मारना', 'हत्या', 'चोट', 'नुकसान', 'हमला', 'नफरत', 'नष्ट', 'बदला', 'हिंसक', 'गुस्सा'],
        'pt': ['matar', 'assassinar', 'ferir', 'prejudicar', 'atacar', 'odiar', 'destruir', 'vingança', 'violento', 'irritado'],
        'ru': ['убить', 'убийство', 'ранить', 'вред', 'атака', 'ненавидеть', 'уничтожить', 'месть', 'насилие', 'злой'],
        'it': ['uccidere', 'assassinare', 'ferire', 'danneggiare', 'attaccare', 'odiare', 'distruggere', 'vendetta', 'violento', 'arrabbiato'],
    }
    
    # Wisdom keywords (multilingual)
    wisdom_keywords = {
        'en': ['wisdom', 'insight', 'enlightenment', 'meditation', 'contemplation', 'reflection', 'philosophy', 'truth', 'understanding', 'awareness'],
        'th': ['ปัญญา', 'สติ', 'สมาธิ', 'ตรัสรู้', 'ไตร่ตรอง', 'ปรัชญา', 'ธรรมะ', 'วิปัสสนา', 'รู้แจ้ง'],
        'zh': ['智慧', '洞察', '觉悟', '冥想', '沉思', '反思', '哲学', '真理', '理解', '意识'],
        'ja': ['知恵', '洞察', '悟り', '瞑想', '熟考', '反省', '哲学', '真理', '理解', '意識'],
        'ko': ['지혜', '통찰', '깨달음', '명상', '숙고', '반성', '철학', '진리', '이해', '인식'],
        'es': ['sabiduría', 'perspicacia', 'iluminación', 'meditación', 'contemplación', 'reflexión', 'filosofía', 'verdad', 'comprensión'],
        'fr': ['sagesse', 'perspicacité', 'illumination', 'méditation', 'contemplation', 'réflexion', 'philosophie', 'vérité', 'compréhension'],
        'de': ['weisheit', 'einsicht', 'erleuchtung', 'meditation', 'kontemplation', 'reflexion', 'philosophie', 'wahrheit', 'verständnis'],
        'ar': ['حكمة', 'بصيرة', 'تنوير', 'تأمل', 'تفكير', 'فلسفة', 'حقيقة', 'فهم', 'وعي'],
        'hi': ['ज्ञान', 'अंतर्दृष्टि', 'ज्ञानोदय', 'ध्यान', 'चिंतन', 'दर्शन', 'सत्य', 'समझ', 'जागरूकता'],
        'pt': ['sabedoria', 'percepção', 'iluminação', 'meditação', 'contemplação', 'reflexão', 'filosofia', 'verdade', 'compreensão'],
        'ru': ['мудрость', 'прозрение', 'просветление', 'медитация', 'созерцание', 'размышление', 'философия', 'истина', 'понимание'],
        'it': ['saggezza', 'intuizione', 'illuminazione', 'meditazione', 'contemplazione', 'riflessione', 'filosofia', 'verità', 'comprensione'],
    }
    
    # Get keywords for detected language (with English as fallback)
    growth_kw = growth_keywords.get(lang, []) + growth_keywords.get('en', [])
    challenge_kw = challenge_keywords.get(lang, []) + challenge_keywords.get('en', [])
    wisdom_kw = wisdom_keywords.get(lang, []) + wisdom_keywords.get('en', [])
    
    # Check growth keywords
    if any(keyword in text for keyword in growth_kw):
        return {
            'classification': 'growth_memory',
            'self_awareness': 0.7,
            'emotional_regulation': 0.6,
            'compassion': 0.7,
            'integrity': 0.6,
            'growth_mindset': 0.7,
            'wisdom': 0.6,
            'transcendence': 0.6,
            'reasoning': f'Fallback: Growth keywords detected in {lang}'
        }
    
    # Check challenge keywords
    if any(keyword in text for keyword in challenge_kw):
        return {
            'classification': 'challenge_memory',
            'self_awareness': 0.3,
            'emotional_regulation': 0.2,
            'compassion': 0.4,
            'integrity': 0.4,
            'growth_mindset': 0.3,
            'wisdom': 0.3,
            'transcendence': 0.2,
            'reasoning': f'Fallback: Challenge keywords detected in {lang}'
        }
    
    # Check wisdom keywords
    if any(keyword in text for keyword in wisdom_kw):
        return {
            'classification': 'wisdom_moment',
            'self_awareness': 0.7,
            'emotional_regulation': 0.7,
            'compassion': 0.7,
            'integrity': 0.7,
            'growth_mindset': 0.7,
            'wisdom': 0.8,
            'transcendence': 0.7,
            'reasoning': f'Fallback: Wisdom keywords detected in {lang}'
        }
    
    # Default neutral
    return {
        'classification': 'neutral_interaction',
        'self_awareness': 0.5,
        'emotional_regulation': 0.5,
        'compassion': 0.5,
        'integrity': 0.5,
        'growth_mindset': 0.5,
        'wisdom': 0.5,
        'transcendence': 0.3,
        'reasoning': f'Fallback: Neutral classification for {lang}'
    }

# ============================================================
# EMBEDDING GENERATION
# ============================================================

async def generate_embedding(text: str) -> Optional[List[float]]:
    """Generate embedding using Ollama nomic-embed-text"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{OLLAMA_URL}/api/embeddings",
                json={
                    "model": EMBEDDING_MODEL,
                    "prompt": text
                }
            )
            
            if response.status_code != 200:
                logger.error(f"Ollama error: {response.status_code}")
                return None
            
            data = response.json()
            embedding = data.get("embedding")
            
            if not embedding or len(embedding) != 768:
                logger.error(f"Invalid embedding dimension: {len(embedding) if embedding else 0}")
                return None
            
            return embedding
            
    except Exception as e:
        logger.error(f"Embedding generation error: {e}")
        return None

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def detect_language(text: str) -> str:
    """Enhanced language detection for 15+ languages"""
    # Thai
    if re.search(r'[\u0E00-\u0E7F]', text):
        return 'th'
    # Chinese (Simplified/Traditional)
    elif re.search(r'[\u4E00-\u9FFF]', text):
        return 'zh'
    # Japanese (Hiragana/Katakana/Kanji)
    elif re.search(r'[\u3040-\u309F\u30A0-\u30FF]', text):
        return 'ja'
    # Korean (Hangul)
    elif re.search(r'[\uAC00-\uD7AF]', text):
        return 'ko'
    # Arabic
    elif re.search(r'[\u0600-\u06FF]', text):
        return 'ar'
    # Hebrew
    elif re.search(r'[\u0590-\u05FF]', text):
        return 'he'
    # Hindi/Devanagari
    elif re.search(r'[\u0900-\u097F]', text):
        return 'hi'
    # Cyrillic (Russian, Ukrainian, etc.)
    elif re.search(r'[\u0400-\u04FF]', text):
        return 'ru'
    # Greek
    elif re.search(r'[\u0370-\u03FF]', text):
        return 'el'
    # Latin-based languages - detect by common words/patterns
    else:
        text_lower = text.lower()
        # Spanish
        if any(word in text_lower for word in ['el', 'la', 'los', 'las', 'que', 'de', 'y', 'a', 'en', 'es', 'por', 'para', 'con', 'su', 'este', 'una', 'muy', 'qué', 'cómo', 'año', 'español']):
            return 'es'
        # French
        elif any(word in text_lower for word in ['le', 'la', 'les', 'de', 'des', 'un', 'une', 'et', 'est', 'à', 'dans', 'pour', 'ce', 'qui', 'avec', 'être', 'très', 'où', 'comment', 'français']):
            return 'fr'
        # German
        elif any(word in text_lower for word in ['der', 'die', 'das', 'den', 'dem', 'und', 'ist', 'in', 'zu', 'mit', 'von', 'für', 'auf', 'auch', 'wie', 'wo', 'warum', 'deutsch']):
            return 'de'
        # Portuguese
        elif any(word in text_lower for word in ['o', 'a', 'os', 'as', 'de', 'do', 'da', 'dos', 'das', 'em', 'no', 'na', 'por', 'para', 'com', 'que', 'é', 'um', 'uma', 'não', 'muito', 'como', 'português']):
            return 'pt'
        # Italian
        elif any(word in text_lower for word in ['il', 'lo', 'la', 'i', 'gli', 'le', 'di', 'da', 'in', 'con', 'su', 'per', 'tra', 'fra', 'che', 'è', 'un', 'una', 'non', 'molto', 'come', 'italiano']):
            return 'it'
        # Dutch
        elif any(word in text_lower for word in ['de', 'het', 'een', 'van', 'in', 'is', 'en', 'op', 'te', 'voor', 'met', 'dat', 'dit', 'zijn', 'niet', 'zeer', 'hoe', 'waar', 'nederlands']):
            return 'nl'
        # Swedish
        elif any(word in text_lower for word in ['det', 'som', 'en', 'och', 'är', 'på', 'i', 'för', 'att', 'med', 'av', 'till', 'från', 'inte', 'mycket', 'hur', 'var', 'svenska']):
            return 'sv'
        # Norwegian
        elif any(word in text_lower for word in ['det', 'som', 'en', 'og', 'er', 'på', 'i', 'for', 'å', 'med', 'av', 'til', 'fra', 'ikke', 'veldig', 'hvordan', 'hvor', 'norsk']):
            return 'no'
        # Danish
        elif any(word in text_lower for word in ['det', 'som', 'en', 'og', 'er', 'på', 'i', 'for', 'at', 'med', 'af', 'til', 'fra', 'ikke', 'meget', 'hvordan', 'hvor', 'dansk']):
            return 'da'
        # Polish
        elif any(word in text_lower for word in ['to', 'jest', 'że', 'w', 'i', 'na', 'z', 'do', 'o', 'nie', 'się', 'jak', 'bardzo', 'gdzie', 'polski']):
            return 'pl'
        # Turkish
        elif any(word in text_lower for word in ['bu', 've', 'bir', 'için', 'ile', 'de', 'da', 'ne', 'çok', 'nasıl', 'nerede', 'türkçe']):
            return 'tr'
        # Vietnamese
        elif any(word in text_lower for word in ['và', 'của', 'là', 'có', 'trong', 'với', 'cho', 'không', 'rất', 'như', 'thế', 'nào', 'ở', 'đâu', 'tiếng', 'việt']):
            return 'vi'
        # Indonesian/Malay
        elif any(word in text_lower for word in ['yang', 'dan', 'di', 'ke', 'dari', 'dengan', 'untuk', 'ini', 'itu', 'tidak', 'sangat', 'bagaimana', 'dimana', 'bahasa', 'indonesia']):
            return 'id'
        # Default to English
        else:
            return 'en'

def detect_moments(ethical_scores: Dict, classification: str) -> List[Dict]:
    """Detect significant moments based on scores and classification"""
    moments = []
    
    if ethical_scores.get('self_awareness', 0) > 0.7:
        moments.append({
            'type': 'breakthrough',
            'severity': 'positive',
            'description': 'High self-awareness detected',
            'timestamp': datetime.now().isoformat()
        })
    
    if ethical_scores.get('emotional_regulation', 0) < 0.3:
        moments.append({
            'type': 'struggle',
            'severity': 'neutral',
            'description': 'Emotional difficulty detected',
            'timestamp': datetime.now().isoformat()
        })
    
    if classification == 'needs_support':
        moments.append({
            'type': 'crisis',
            'severity': 'critical',
            'description': 'User needs support',
            'timestamp': datetime.now().isoformat(),
            'requires_intervention': True
        })
    
    if classification in ['growth_memory', 'wisdom_moment']:
        moments.append({
            'type': 'growth',
            'severity': 'positive',
            'description': 'Growth or wisdom detected',
            'timestamp': datetime.now().isoformat()
        })
    
    return moments

def determine_growth_stage(ethical_scores: Dict[str, float]) -> int:
    """Determine growth stage from ethical scores"""
    avg_score = sum(ethical_scores.values()) / len(ethical_scores)
    
    if avg_score < 0.3:
        return 1
    elif avg_score < 0.5:
        return 2
    elif avg_score < 0.7:
        return 3
    elif avg_score < 0.85:
        return 4
    else:
        return 5

# ============================================================
# DATABASE OPERATIONS
# ============================================================

def save_ethical_profile(user_id: str, ethical_scores: Dict, stage: int, db_conn):
    cursor = db_conn.cursor()
    
    cursor.execute("""
        INSERT INTO user_data_schema.ethical_profiles 
        (user_id, self_awareness, emotional_regulation, compassion, 
         integrity, growth_mindset, wisdom, transcendence, growth_stage, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
        ON CONFLICT (user_id) 
        DO UPDATE SET
            self_awareness = EXCLUDED.self_awareness,
            emotional_regulation = EXCLUDED.emotional_regulation,
            compassion = EXCLUDED.compassion,
            integrity = EXCLUDED.integrity,
            growth_mindset = EXCLUDED.growth_mindset,
            wisdom = EXCLUDED.wisdom,
            transcendence = EXCLUDED.transcendence,
            growth_stage = EXCLUDED.growth_stage,
            total_interactions = ethical_profiles.total_interactions + 1,
            updated_at = NOW()
    """, (
        user_id,
        ethical_scores['self_awareness'],
        ethical_scores['emotional_regulation'],
        ethical_scores['compassion'],
        ethical_scores['integrity'],
        ethical_scores['growth_mindset'],
        ethical_scores['wisdom'],
        ethical_scores['transcendence'],
        stage
    ))
    
    db_conn.commit()
    cursor.close()

async def save_memory_with_embedding(
    user_id: str, 
    text: str,
    embedding: List[float],
    classification: str,
    lang: str,
    growth_stage: int,
    db_conn
) -> str:
    """Save to memory_embeddings with vector and metadata"""
    cursor = db_conn.cursor()
    
    vector_str = f"[{','.join(map(str, embedding))}]"
    
    metadata = {
        'classification': classification,
        'language': lang,
        'growth_stage': growth_stage,
        'source': 'gating_service',
        'created_at': datetime.now().isoformat()
    }
    
    cursor.execute("""
        INSERT INTO user_data_schema.memory_embeddings
        (user_id, content, embedding, metadata, created_at)
        VALUES (%s, %s, %s::vector, %s, NOW())
        RETURNING id
    """, (
        user_id,
        text,
        vector_str,
        json.dumps(metadata)
    ))
    
    memory_id = cursor.fetchone()[0]
    db_conn.commit()
    cursor.close()
    
    logger.info(f"✅ Memory saved with ID: {memory_id}")
    return str(memory_id)

def save_interaction_memory(
    user_id: str, 
    text: str, 
    classification: str,
    ethical_scores: Dict,
    moments: List[Dict],
    reflection_prompt: str,
    gentle_guidance: Optional[str],
    memory_embedding_id: str,
    db_conn
):
    """Save to interaction_memories with link to memory_embeddings"""
    cursor = db_conn.cursor()
    
    cursor.execute("""
        INSERT INTO user_data_schema.interaction_memories
        (user_id, text, classification, ethical_scores, moments, 
         reflection_prompt, gentle_guidance, metadata, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
        RETURNING id
    """, (
        user_id,
        text,
        classification,
        json.dumps(ethical_scores),
        json.dumps(moments),
        reflection_prompt,
        gentle_guidance,
        json.dumps({
            'source': 'gating_service',
            'memory_embedding_id': memory_embedding_id
        })
    ))
    
    db_conn.commit()
    cursor.close()
    logger.info(f"✅ Interaction memory saved")

def get_user_ethical_history(user_id: str, db_conn) -> Dict:
    cursor = db_conn.cursor(cursor_factory=RealDictCursor)
    
    cursor.execute("""
        SELECT * FROM user_data_schema.ethical_profiles
        WHERE user_id = %s
    """, (user_id,))
    
    profile = cursor.fetchone()
    cursor.close()
    
    if profile:
        return {
            'baseline_self_awareness': profile['self_awareness'],
            'baseline_regulation': profile['emotional_regulation'],
            'baseline_compassion': profile['compassion'],
            'baseline_integrity': profile['integrity'],
            'baseline_growth': profile['growth_mindset'],
            'baseline_wisdom': profile['wisdom'],
            'baseline_transcendence': profile['transcendence'],
            'current_stage': profile['growth_stage']
        }
    
    return {
        'baseline_self_awareness': 0.3,
        'baseline_regulation': 0.4,
        'baseline_compassion': 0.4,
        'baseline_integrity': 0.5,
        'baseline_growth': 0.4,
        'baseline_wisdom': 0.3,
        'baseline_transcendence': 0.2,
        'current_stage': 2
    }

# ============================================================
# GUIDANCE TEMPLATES
# ============================================================

GUIDANCE_TEMPLATES = {
    'crisis': {
        'en': "I'm concerned about you. Please reach out to a mental health professional.",
        'th': "ฉันเป็นห่วงคุณมาก โปรดติดต่อสายด่วนสุขภาพจิต 1323",
    },
    'emotional_dysregulation': {
        'en': "Take a deep breath. These feelings will pass.",
        'th': "ลองหายใจเข้าลึกๆ ความรู้สึกนี้จะผ่านไป",
    },
}

REFLECTION_PROMPTS = {
    1: {
        'en': "What are you feeling right now?",
        'th': "สิ่งที่คุณกำลังรู้สึกตอนนี้คืออะไร?",
    },
    2: {
        'en': "If someone else were in this situation, how would they feel?",
        'th': "ถ้าคนอื่นอยู่ในสถานการณ์นี้ เขาจะรู้สึกยังไง?",
    },
    3: {
        'en': "What values does this decision reflect?",
        'th': "การตัดสินใจนี้สะท้อนคุณค่าอะไร?",
    },
}

def get_guidance(classification: str, ethical_scores: Dict, lang: str) -> Optional[str]:
    if classification == 'needs_support':
        return GUIDANCE_TEMPLATES['crisis'].get(lang, GUIDANCE_TEMPLATES['crisis']['en'])
    
    if ethical_scores.get('emotional_regulation', 0.5) < 0.3:
        return GUIDANCE_TEMPLATES['emotional_dysregulation'].get(lang, GUIDANCE_TEMPLATES['emotional_dysregulation']['en'])
    
    return None

def get_reflection_prompt(stage: int, lang: str) -> str:
    prompts = REFLECTION_PROMPTS.get(stage, REFLECTION_PROMPTS[2])
    return prompts.get(lang, prompts.get('en', ''))

# ============================================================
# API MODELS
# ============================================================

class GatingRequest(BaseModel):
    user_id: str
    text: str
    database_url: str
    session_id: Optional[str] = None
    metadata: Optional[Dict] = {}

class GatingResponse(BaseModel):
    status: str
    routing: str
    ethical_scores: Dict[str, float]
    growth_stage: int
    moments: List[Dict]
    insights: Optional[Dict] = None
    reflection_prompt: Optional[str] = None
    gentle_guidance: Optional[str] = None
    growth_opportunity: Optional[str] = None
    detected_language: Optional[str] = None
    memory_id: Optional[str] = None

# ============================================================
# MAIN ENDPOINT - LLM CLASSIFICATION
# ============================================================

@app.post("/gating/ethical-route", response_model=GatingResponse)
async def ethical_routing(request: GatingRequest):
    """Process text through ethical growth framework with IMPROVED multilingual LLM classification"""
    
    logger.info(f"📝 Processing text for user {request.user_id}: {request.text[:50]}...")
    
    if not request.database_url:
        raise HTTPException(status_code=400, detail="database_url is required")
    
    db_conn = psycopg2.connect(request.database_url)
    
    try:
        # 1. Detect language
        lang = detect_language(request.text)
        logger.info(f"🌍 Detected language: {lang}")
        
        # 2. Generate embedding
        logger.info(f"🧠 Generating embedding...")
        embedding = await generate_embedding(request.text)
        
        if not embedding:
            logger.warning("⚠️  Embedding generation failed")
        
        # 3. ✅ IMPROVED LLM CLASSIFICATION
        logger.info(f"🤖 Using LLM for classification (language: {lang})...")
        llm_result = await classify_with_llm(request.text, lang)
        
        classification = llm_result['classification']
        ethical_scores = {
            'self_awareness': llm_result['self_awareness'],
            'emotional_regulation': llm_result['emotional_regulation'],
            'compassion': llm_result['compassion'],
            'integrity': llm_result['integrity'],
            'growth_mindset': llm_result['growth_mindset'],
            'wisdom': llm_result['wisdom'],
            'transcendence': llm_result['transcendence'],
        }
        
        logger.info(f"✅ Classification: {classification}")
        logger.info(f"📊 Reasoning: {llm_result.get('reasoning', 'N/A')}")
        
        # 4. Determine growth stage
        growth_stage = determine_growth_stage(ethical_scores)
        
        # 5. Detect moments
        moments = detect_moments(ethical_scores, classification)
        
        # 6. Generate guidance
        reflection_prompt = get_reflection_prompt(growth_stage, lang)
        gentle_guidance = get_guidance(classification, ethical_scores, lang)
        
        # 7. Save to memory_embeddings
        memory_id = None
        if embedding:
            logger.info(f"💾 Saving to memory_embeddings...")
            memory_id = await save_memory_with_embedding(
                request.user_id,
                request.text,
                embedding,
                classification,
                lang,
                growth_stage,
                db_conn
            )
        else:
            logger.error("❌ Cannot save without embedding")
            raise HTTPException(status_code=500, detail="Embedding generation failed")
        
        # 8. Save ethical profile
        save_ethical_profile(request.user_id, ethical_scores, growth_stage, db_conn)
        
        # 9. Save interaction memory
        save_interaction_memory(
            request.user_id,
            request.text,
            classification,
            ethical_scores,
            moments,
            reflection_prompt,
            gentle_guidance,
            memory_id,
            db_conn
        )
        
        logger.info(f"✅ Processing completed: {classification}")
        
        return GatingResponse(
            status='success',
            routing=classification,
            ethical_scores=ethical_scores,
            growth_stage=growth_stage,
            moments=moments,
            insights={
                'strongest_dimension': max(ethical_scores, key=ethical_scores.get),
                'growth_area': min(ethical_scores, key=ethical_scores.get),
                'llm_reasoning': llm_result.get('reasoning', 'N/A')
            },
            reflection_prompt=reflection_prompt,
            gentle_guidance=gentle_guidance,
            growth_opportunity=f"Stage {growth_stage}/5",
            detected_language=lang,
            memory_id=memory_id
        )
        
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db_conn.close()

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "service": "ethical_growth_gating",
        "version": "4.0-universal-multilingual",
        "supported_languages": [
            "English (en)", "Thai (th)", "Chinese (zh)", "Japanese (ja)", 
            "Korean (ko)", "Spanish (es)", "French (fr)", "German (de)", 
            "Portuguese (pt)", "Russian (ru)", "Italian (it)", "Arabic (ar)",
            "Hindi (hi)", "Dutch (nl)", "Swedish (sv)", "Norwegian (no)",
            "Danish (da)", "Polish (pl)", "Turkish (tr)", "Vietnamese (vi)",
            "Indonesian (id)", "Greek (el)", "Hebrew (he)",
            "and more..."
        ],
        "multilingual": True,
        "embedding_model": EMBEDDING_MODEL,
        "classification_model": LLM_MODEL,
        "ollama_url": OLLAMA_URL
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)