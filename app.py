"""
Streamlit Law AI Agent - Multi-language Support
Features:
- Supports English, Hindi, and Marathi
- Automatic language detection
- Multi-language sample data
- Fast responses with pre-configured models

DISCLAIMER: This is for informational purposes only, not legal advice.
"""

import os
import re
import json
from typing import List, Tuple, Dict, Any
import streamlit as st
from dotenv import load_dotenv
import pandas as pd
import requests

# Vector search imports
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np
    VECTOR_SEARCH_AVAILABLE = True
except ImportError:
    VECTOR_SEARCH_AVAILABLE = False
    st.error("Vector search libraries not installed. Please install: pip install sentence-transformers faiss-cpu")

# Load environment variables
load_dotenv()

# Fixed Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-70b-8192")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Fixed settings
TOP_K = 3
SIMILARITY_THRESHOLD = 0.7
ENABLE_FALLBACK = True

# Language configuration
LANGUAGES = {
    "english": {"code": "en", "name": "English", "flag": "🇺🇸"},
    "hindi": {"code": "hi", "name": "हिन्दी", "flag": "🇮🇳"},
    "marathi": {"code": "mr", "name": "मराठी", "flag": "🇮🇳"}
}

# Initialize session state
def initialize_session_state():
    if "faiss_index" not in st.session_state:
        st.session_state.faiss_index = None
    if "documents" not in st.session_state:
        st.session_state.documents = []
    if "embedding_model" not in st.session_state:
        st.session_state.embedding_model = None
    if "index_built" not in st.session_state:
        st.session_state.index_built = False
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "api_available" not in st.session_state:
        st.session_state.api_available = {"gemini": True, "groq": True}
    if "current_language" not in st.session_state:
        st.session_state.current_language = "english"

# Multi-language sample data
def load_sample_data() -> List[Dict[str, Any]]:
    """Load comprehensive sample legal data in multiple languages"""
    sample_laws = [
        # English Laws
        {
            "text": "California Vehicle Code § 23152: Driving under the influence of alcohol or drugs. First offense: 96 hours to 6 months jail, $390-$1000 fine, 6-month license suspension, DUI school. Second offense: 90 days to 1 year jail, $390-$1000 fine, 2-year license suspension.",
            "source": "California_Vehicle_Code",
            "jurisdiction": "California",
            "language": "english"
        },
        {
            "text": "Indian Penal Code § 302: Punishment for murder. Whoever commits murder shall be punished with death or imprisonment for life and shall also be liable to fine.",
            "source": "Indian_Penal_Code",
            "jurisdiction": "India",
            "language": "english"
        },
        {
            "text": "Motor Vehicles Act § 185: Driving by a drunken person or by a person under the influence of drugs. Penalty: First offense - imprisonment up to 6 months and/or fine up to ₹10,000. Second offense - imprisonment up to 2 years and/or fine up to ₹15,000.",
            "source": "Indian_Motor_Vehicles_Act",
            "jurisdiction": "India",
            "language": "english"
        },
        
        # Hindi Laws (हिन्दी)
        {
            "text": "भारतीय दंड संहिता धारा 302: हत्या की सजा। जो कोई हत्या करेगा, वह मृत्यु दंड से या आजीवन कारावास से दंडित किया जाएगा और जुर्माने से भी दंडनीय होगा।",
            "source": "भारतीय_दंड_संहिता",
            "jurisdiction": "भारत",
            "language": "hindi"
        },
        {
            "text": "मोटर वाहन अधिनियम धारा 185: नशे में वाहन चलाना। पहला अपराध: 6 महीने तक की कैद और/या 10,000 रुपये तक का जुर्माना। दूसरा अपराध: 2 साल तक की कैद और/या 15,000 रुपये तक का जुर्माना।",
            "source": "मोटर_वाहन_अधिनियम",
            "jurisdiction": "भारत",
            "language": "hindi"
        },
        {
            "text": "भारतीय दंड संहिता धारा 378: चोरी। जो कोई, बेईमानी से किसी व्यक्ति की संपत्ति को बिना सहमति के ले जाने का इरादा रखता है, वह चोरी करता है। सजा: 3 साल तक की कैद या जुर्माना या दोनों।",
            "source": "भारतीय_दंड_संहिता",
            "jurisdiction": "भारत",
            "language": "hindi"
        },
        
        # Marathi Laws (मराठी)
        {
            "text": "भारतीय दंड संहिता कलम 302: खुनाची शिक्षा. जो कोणी खून करेल त्यास मृत्युदंड किंवा आजन्म कारावास आणि दंड ठोठावला जाईल.",
            "source": "भारतीय_दंड_संहिता",
            "jurisdiction": "भारत",
            "language": "marathi"
        },
        {
            "text": "मोटर वाहन कायदा कलम 185: दारू पिऊन गाडी चालवणे. पहिला गुन्हा: 6 महिन्यांपर्यंत तुरुंगवास आणि/किंवा 10,000 रुपयांपर्यंत दंड. दुसरा गुन्हा: 2 वर्षांपर्यंत तुरुंगवास आणि/किंवा 15,000 रुपयांपर्यंत दंड.",
            "source": "मोटर_वाहन_कायदा",
            "jurisdiction": "भारत",
            "language": "marathi"
        },
        {
            "text": "भारतीय दंड संहिता कलम 378: चोरी. जो कोणी, बेईमानीने कोणत्याही व्यक्तीची मालमत्ता त्याच्या परवानगीशिवाय नेण्याचा हेतू ठेवतो, तो चोरी करतो. शिक्षा: 3 वर्षांपर्यंत तुरुंगवास किंवा दंड किंवा दोन्ही.",
            "source": "भारतीय_दंड_संहिता",
            "jurisdiction": "भारत",
            "language": "marathi"
        },
        
        # General Legal Principles in all languages
        {
            "text": "Drunk Driving Penalties Generally: Most jurisdictions impose jail time, fines, license suspension, mandatory alcohol education programs. Penalties increase with prior offenses.",
            "source": "General_Legal_Principles",
            "jurisdiction": "Multiple Jurisdictions",
            "language": "english"
        },
        {
            "text": "ड्रिंक एंड ड्राइव सामान्य दंड: अधिकांश क्षेत्राधिकार जेल की सजा, जुर्माना, लाइसेंस निलंबन, अनिवार्य शराब शिक्षा कार्यक्रम लागू करते हैं। पिछले अपराधों के साथ दंड बढ़ जाते हैं।",
            "source": "सामान्य_कानून_सिद्धांत",
            "jurisdiction": "विभिन्न क्षेत्राधिकार",
            "language": "hindi"
        },
        {
            "text": "ड्रिंक अँड ड्राईव्ह सामान्य शिक्षा: बहुतेक क्षेत्राधिकार तुरुंगवास, दंड, परवाना निलंबन, अनिवार्य दारू शिक्षण कार्यक्रम लागू करतात. मागील गुन्ह्यांसह शिक्षा वाढतात.",
            "source": "सामान्य_कायदा_तत्त्वे",
            "jurisdiction": "विविध क्षेत्राधिकार",
            "language": "marathi"
        }
    ]
    return sample_laws

def detect_language(text: str) -> str:
    """Detect language of the input text"""
    # Simple language detection based on character ranges
    hindi_chars = re.findall(r'[\u0900-\u097F]', text)
    marathi_chars = re.findall(r'[\u0900-\u097F]', text)  # Same Unicode range as Hindi
    
    # If significant Devanagari characters found, check common words
    if len(hindi_chars) > 5:
        # Check for Marathi-specific words
        marathi_words = ['आहे', 'मराठी', 'कायदा', 'शिक्षा', 'तुरुंगवास']
        hindi_words = ['है', 'हिन्दी', 'कानून', 'सजा', 'जेल']
        
        marathi_count = sum(1 for word in marathi_words if word in text)
        hindi_count = sum(1 for word in hindi_words if word in text)
        
        if marathi_count > hindi_count:
            return "marathi"
        else:
            return "hindi"
    
    # Default to English
    return "english"

def get_language_prompt_instruction(language: str) -> str:
    """Get language instruction for the prompt based on selected language"""
    instructions = {
        "english": "Provide your response in English.",
        "hindi": "कृपया अपना उत्तर हिंदी में दें।",
        "marathi": "कृपया आपले उत्तर मराठीत द्या."
    }
    return instructions.get(language, instructions["english"])

def build_semantic_index():
    """Build FAISS index automatically with sample data"""
    if st.session_state.index_built:
        return
    
    if not VECTOR_SEARCH_AVAILABLE:
        st.error("Vector search not available. Install: pip install sentence-transformers faiss-cpu")
        return
    
    try:
        # Load sample data
        all_documents = load_sample_data()
        
        if not all_documents:
            st.error("No documents available for indexing.")
            return
        
        # Initialize embedding model
        if st.session_state.embedding_model is None:
            with st.spinner("🔄 Loading AI models..."):
                st.session_state.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        
        # Generate embeddings
        texts = [doc["text"] for doc in all_documents]
        with st.spinner("📚 Building legal database..."):
            embeddings = st.session_state.embedding_model.encode(texts, normalize_embeddings=True)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings.astype('float32'))
        
        # Store in session state
        st.session_state.faiss_index = index
        st.session_state.documents = all_documents
        st.session_state.index_built = True
        
    except Exception as e:
        st.error(f"Error building index: {str(e)}")

def semantic_search(query: str) -> List[Tuple[float, Dict]]:
    """Perform fast semantic search"""
    if not st.session_state.index_built:
        return []
    
    try:
        # Encode query
        query_embedding = st.session_state.embedding_model.encode([query], normalize_embeddings=True)
        
        # Search
        scores, indices = st.session_state.faiss_index.search(query_embedding.astype('float32'), TOP_K)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(st.session_state.documents) and score >= SIMILARITY_THRESHOLD:
                results.append((float(score), st.session_state.documents[idx]))
        
        return results
    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return []

# Fast LLM Integration with multi-language support
def call_gemini(prompt: str) -> Tuple[bool, str]:
    """Call Gemini API with robust error handling"""
    if not GEMINI_API_KEY:
        st.session_state.api_available["gemini"] = False
        return False, "Gemini API key not configured in .env file"
    
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
        
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 1024,
            }
        }
        
        response = requests.post(url, json=payload, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            if 'candidates' in data and len(data['candidates']) > 0:
                text = data['candidates'][0]['content']['parts'][0]['text']
                st.session_state.api_available["gemini"] = True
                return True, text
            else:
                st.session_state.api_available["gemini"] = False
                return False, "No response generated from Gemini"
        elif response.status_code == 401:
            st.session_state.api_available["gemini"] = False
            return False, "Gemini API key is invalid or expired"
        elif response.status_code == 429:
            st.session_state.api_available["gemini"] = False
            return False, "Gemini API quota exceeded"
        else:
            st.session_state.api_available["gemini"] = False
            return False, f"Gemini API Error: {response.status_code}"
            
    except Exception as e:
        st.session_state.api_available["gemini"] = False
        return False, f"Gemini request failed: {str(e)}"

def call_groq(prompt: str) -> Tuple[bool, str]:
    """Call Groq API with robust error handling"""
    if not GROQ_API_KEY:
        st.session_state.api_available["groq"] = False
        return False, "Groq API key not configured in .env file"
    
    try:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "model": GROQ_MODEL,
            "temperature": 0.1,
            "max_tokens": 1024,
            "top_p": 0.9
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            st.session_state.api_available["groq"] = True
            return True, data['choices'][0]['message']['content']
        elif response.status_code == 401:
            st.session_state.api_available["groq"] = False
            return False, "Groq API key is invalid or expired"
        elif response.status_code == 429:
            st.session_state.api_available["groq"] = False
            return False, "Groq API rate limit exceeded"
        else:
            st.session_state.api_available["groq"] = False
            return False, f"Groq API Error: {response.status_code}"
            
    except Exception as e:
        st.session_state.api_available["groq"] = False
        return False, f"Groq request failed: {str(e)}"

def get_llm_response(prompt: str, language: str) -> Tuple[str, str]:
    """Get fast LLM response with better fallback handling"""
    
    # Try Gemini first
    if st.session_state.api_available["gemini"]:
        success, response = call_gemini(prompt)
        if success:
            return response, "Gemini"
    
    # Try Groq fallback if enabled and available
    if ENABLE_FALLBACK and st.session_state.api_available["groq"]:
        success, response = call_groq(prompt)
        if success:
            return response, "Groq"
    
    # If both APIs failed, provide a helpful response using local data only
    return generate_local_response(prompt, language), "Local Database"

def generate_local_response(prompt: str, language: str) -> str:
    """Generate response using only local data when APIs are unavailable"""
    # Extract the main query from the prompt
    query_match = re.search(r"Legal Query:\s*(.+)", prompt)
    user_query = query_match.group(1) if query_match else "the legal question"
    
    # Simple rule-based responses for common queries in different languages
    query_lower = user_query.lower()
    
    # Responses for drink and drive in different languages
    if any(word in query_lower for word in ['drink', 'drunk', 'dui', 'dwi', 'alcohol', 'drive', 'शराब', 'गाडी', 'ड्राइव', 'दारू', 'गाडी']):
        responses = {
            "english": """**Drunk Driving Penalties Analysis**

**Applicable Laws:**
- California Vehicle Code § 23152: Driving under the influence
- Indian Motor Vehicles Act § 185: Driving by drunken person

**Penalties Summary:**

*California:*
- First offense: 96 hours to 6 months jail, $390-$1000 fine, 6-month license suspension
- Second offense: 90 days to 1 year jail, $390-$1000 fine, 2-year license suspension

*India:*
- First offense: Imprisonment up to 6 months and/or fine up to ₹10,000
- Second offense: Imprisonment up to 2 years and/or fine up to ₹15,000

**Additional Consequences:**
- Mandatory alcohol education programs
- Criminal record

**Note:** Penalties vary by jurisdiction. Always consult local legal counsel.

⚠️ *This analysis is based on local database only. API services are currently unavailable.*""",

            "hindi": """**ड्रिंक एंड ड्राइव दंड विश्लेषण**

**लागू कानून:**
- कैलिफोर्निया वाहन संहिता § 23152: शराब के प्रभाव में वाहन चलाना
- भारतीय मोटर वाहन अधिनियम § 185: नशे में वाहन चलाना

**दंड सारांश:**

*कैलिफोर्निया:*
- पहला अपराध: 96 घंटे से 6 महीने जेल, $390-$1000 जुर्माना, 6 महीने लाइसेंस निलंबन
- दूसरा अपराध: 90 दिन से 1 साल जेल, $390-$1000 जुर्माना, 2 साल लाइसेंस निलंबन

*भारत:*
- पहला अपराध: 6 महीने तक कारावास और/या 10,000 रुपये तक जुर्माना
- दूसरा अपराध: 2 साल तक कारावास और/या 15,000 रुपये तक जुर्माना

**अतिरिक्त परिणाम:**
- अनिवार्य शराब शिक्षा कार्यक्रम
- आपराधिक रिकॉर्ड

**नोट:** दंड क्षेत्राधिकार के अनुसार भिन्न होते हैं। हमेशा स्थानीय कानूनी सलाहकार से परामर्श लें।

⚠️ *यह विश्लेषण केवल स्थानीय डेटाबेस पर आधारित है। एपीआई सेवाएं वर्तमान में अनुपलब्ध हैं।*""",

            "marathi": """**ड्रिंक अँड ड्राईव्ह शिक्षा विश्लेषण**

**लागू कायदे:**
- कॅलिफोर्निया वाहन संहिता § 23152: दारूच्या परिणामाखाली वाहन चालवणे
- भारतीय मोटर वाहन कायदा § 185: दारू पिऊन वाहन चालवणे

**शिक्षा सारांश:**

*कॅलिफोर्निया:*
- पहिला गुन्हा: 96 तास ते 6 महिने तुरुंगवास, $390-$1000 दंड, 6 महिने परवाना निलंबन
- दुसरा गुन्हा: 90 दिवस ते 1 वर्ष तुरुंगवास, $390-$1000 दंड, 2 वर्ष परवाना निलंबन

*भारत:*
- पहिला गुन्हा: 6 महिन्यांपर्यंत तुरुंगवास आणि/किंवा 10,000 रुपयांपर्यंत दंड
- दुसरा गुन्हा: 2 वर्षांपर्यंत तुरुंगवास आणि/किंवा 15,000 रुपयांपर्यंत दंड

**अतिरिक्त परिणाम:**
- अनिवार्य दारू शिक्षण कार्यक्रम
- गुन्हेगारी रेकॉर्ड

**सूचना:** शिक्षा क्षेत्राधिकारानुसार बदलतात. नेहमी स्थानिक कायदा सल्लागारांचा सल्ला घ्या.

⚠️ *हे विश्लेषण फक्त स्थानिक डेटाबेसवर आधारित आहे. API सेवा सध्या अनुपलब्ध आहेत.*"""
        }
        return responses.get(language, responses["english"])

    else:
        # Generic response for other queries
        responses = {
            "english": f"""**Legal Analysis for: {user_query}**

Based on our local legal database, here are relevant findings:

**Key Legal Principles Identified:**
- Multiple jurisdictions have specific penalties for various offenses
- Penalties typically consider severity and prior offenses

**Recommendations:**
1. Consult with a qualified attorney in your jurisdiction
2. Review specific statute sections mentioned in our database

⚠️ *This analysis is based on local database only. API services are currently unavailable.*""",

            "hindi": f"""**कानूनी विश्लेषण: {user_query}**

हमारे स्थानीय कानूनी डेटाबेस के आधार पर, यहां प्रासंगिक निष्कर्ष दिए गए हैं:

**पहचाने गए मुख्य कानूनी सिद्धांत:**
- विभिन्न क्षेत्राधिकारों में विभिन्न अपराधों के लिए विशिष्ट दंड हैं
- दंड आमतौर पर गंभीरता और पिछले अपराधों को ध्यान में रखते हैं

**सिफारिशें:**
1. अपने क्षेत्राधिकार में एक योग्य वकील से परामर्श करें
2. हमारे डेटाबेस में उल्लिखित विशिष्ट धाराओं की समीक्षा करें

⚠️ *यह विश्लेषण केवल स्थानीय डेटाबेस पर आधारित है। एपीआई सेवाएं वर्तमान में अनुपलब्ध हैं।*""",

            "marathi": f"""**कायदेशीर विश्लेषण: {user_query}**

आमच्या स्थानिक कायदेशीर डेटाबेसवर आधारित, येथे संबंधित निष्कर्ष आहेत:

**ओळखलेली मुख्य कायदेशीर तत्त्वे:**
- विविध क्षेत्राधिकारांमध्ये विविध गुन्ह्यांसाठी विशिष्ट शिक्षा आहेत
- शिक्षा सामान्यत: गंभीरता आणि मागील गुन्हे लक्षात घेतात

**शिफारसी:**
1. आपल्या क्षेत्राधिकारातील पात्र वकिलांचा सल्ला घ्या
2. आमच्या डेटाबेसमध्ये नमूद केलेल्या विशिष्ट कलमांचे पुनरावलोकन करा

⚠️ *हे विश्लेषण फक्त स्थानिक डेटाबेसवर आधारित आहे. API सेवा सध्या अनुपलब्ध आहेत.*"""
        }
        return responses.get(language, responses["english"])

# UI Configuration
st.set_page_config(
    page_title="Law AI Agent - Multi-language",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .disclaimer {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 8px;
        margin-bottom: 1rem;
        color: #856404;
        font-size: 0.9rem;
    }
    .citation {
        background-color: #e7f3ff;
        border-left: 4px solid #1f77b4;
        padding: 8px;
        margin: 3px 0;
        font-size: 0.9rem;
    }
    .model-badge {
        background-color: #6c757d;
        color: white;
        padding: 2px 6px;
        border-radius: 10px;
        font-size: 0.7rem;
    }
    .gemini-badge {
        background-color: #4285f4;
    }
    .groq-badge {
        background-color: #00a67e;
    }
    .local-badge {
        background-color: #6c757d;
    }
    .language-selector {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize and build index automatically
initialize_session_state()

# Header
st.markdown('<div class="main-header">⚖️ Law AI Agent</div>', unsafe_allow_html=True)
st.markdown("""
<div class="disclaimer">
⚠️ <strong>Disclaimer:</strong> This tool provides informational assistance only and does not constitute legal advice. 
Always consult qualified legal professionals for legal matters.
</div>
""", unsafe_allow_html=True)

# Language Selector
st.markdown('<div class="language-selector">', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    if st.button(f"{LANGUAGES['english']['flag']} English", use_container_width=True):
        st.session_state.current_language = "english"
        st.rerun()
with col2:
    if st.button(f"{LANGUAGES['hindi']['flag']} हिन्दी", use_container_width=True):
        st.session_state.current_language = "hindi"
        st.rerun()
with col3:
    if st.button(f"{LANGUAGES['marathi']['flag']} मराठी", use_container_width=True):
        st.session_state.current_language = "marathi"
        st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

# Show current language
current_lang = LANGUAGES[st.session_state.current_language]
st.info(f"🌐 **Current Language:** {current_lang['flag']} {current_lang['name']}")

# Auto-build index on first load
if not st.session_state.index_built:
    with st.spinner("🚀 Initializing legal AI assistant..."):
        build_semantic_index()
    
    if st.session_state.index_built:
        st.success("✅ Legal database ready! Ask your question below.")
    else:
        st.error("❌ Failed to initialize legal database")

# Main Chat Interface
def main():
    # Language-specific UI texts
    ui_texts = {
        "english": {
            "title": "💬 Ask Legal Questions",
            "examples": "**Try these examples:** `DUI penalties` | `Drunk driving laws` | `Theft punishments`",
            "placeholder": "e.g., 'What are the penalties for drink and drive?'",
            "analyze": "🔍 Analyze",
            "clear": "🔄 Clear"
        },
        "hindi": {
            "title": "💬 कानूनी प्रश्न पूछें",
            "examples": "**उदाहरण आज़माएं:** `डीयूआई दंड` | `शराब पीकर गाडी चलाने के कानून` | `चोरी की सजा`",
            "placeholder": "उदा., 'शराब पीकर गाडी चलाने की क्या सजा है?'",
            "analyze": "🔍 विश्लेषण करें",
            "clear": "🔄 साफ करें"
        },
        "marathi": {
            "title": "💬 कायदेशीर प्रश्न विचारा",
            "examples": "**उदाहरणे वापरून पहा:** `DUI शिक्षा` | `दारू पिऊन गाडी चालवण्याचे कायदे` | `चोरीची शिक्षा`",
            "placeholder": "उदा., 'दारू पिऊन गाडी चालवल्यास काय शिक्षा आहे?'",
            "analyze": "🔍 विश्लेषण करा",
            "clear": "🔄 साफ करा"
        }
    }
    
    ui = ui_texts[st.session_state.current_language]
    
    st.header(ui["title"])
    st.markdown(ui["examples"])
    
    user_query = st.text_input(
        "Enter your legal question:",
        placeholder=ui["placeholder"],
        key="query_input"
    )
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        analyze_btn = st.button(ui["analyze"], type="primary", use_container_width=True)
    
    with col2:
        if st.button(ui["clear"], use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    
    # Process query immediately when button is clicked
    if analyze_btn and user_query:
        if not st.session_state.index_built:
            st.error("Legal database not ready. Please refresh the page.")
            return
        
        # Auto-detect language if not set manually
        if user_query and not any(c in user_query for c in ['\u0900-\u097F']):  # If no Devanagari chars
            detected_lang = detect_language(user_query)
            if detected_lang != st.session_state.current_language:
                st.session_state.current_language = detected_lang
                st.info(f"🌐 Auto-detected language: {LANGUAGES[detected_lang]['name']}")
        
        # Create prompt from user query
        relevant_docs = semantic_search(user_query)
        docs_text = ""
        for i, (score, doc) in enumerate(relevant_docs):
            docs_text += f"[Doc {i+1}: {doc.get('source')}] {doc['text']}\n\n"
        
        # Add language instruction to prompt
        language_instruction = get_language_prompt_instruction(st.session_state.current_language)
        prompt = f"Legal Query: {user_query}\n\nRelevant Legal Documents:\n{docs_text}\n\n{language_instruction}\n\nProvide concise legal analysis focusing on applicable laws and penalties."
        
        # Get response
        with st.spinner("🤔 Analyzing..."):
            response, model_used = get_llm_response(prompt, st.session_state.current_language)
            citations = extract_citations(response)
            
            # Add to chat history
            st.session_state.chat_history.insert(0, {
                "query": user_query,
                "response": response,
                "model": model_used,
                "citations": citations,
                "docs": relevant_docs,
                "language": st.session_state.current_language
            })
        
        # Clear input after processing
        st.rerun()
    
    # Display chat history (newest first)
    for i, chat in enumerate(st.session_state.chat_history):
        with st.container():
            st.markdown(f"**Q:** {chat['query']}")
            
            badge_class = {
                "Gemini": "gemini-badge",
                "Groq": "groq-badge", 
                "Local Database": "local-badge"
            }.get(chat['model'], "model-badge")
            
            badge = f'<span class="model-badge {badge_class}">{chat["model"]}</span>'
            
            st.markdown(f'**AI Response** {badge}')
            st.markdown(chat['response'])
            
            if chat['citations']:
                expander_text = {
                    "english": "📚 Legal Citations Found",
                    "hindi": "📚 कानूनी उद्धरण मिले",
                    "marathi": "📚 कायदेशीर उद्धरणे सापडली"
                }.get(chat.get('language', 'english'), "📚 Legal Citations Found")
                
                with st.expander(expander_text):
                    for citation in chat['citations']:
                        st.markdown(f'<div class="citation">{citation}</div>', unsafe_allow_html=True)
            
            if chat['docs']:
                expander_text = {
                    "english": "📄 Relevant Laws Used",
                    "hindi": "📄 प्रासंगिक कानून इस्तेमाल किए गए",
                    "marathi": "📄 संबंधित कायदे वापरले"
                }.get(chat.get('language', 'english'), "📄 Relevant Laws Used")
                
                with st.expander(expander_text):
                    for score, doc in chat['docs']:
                        st.write(f"**{doc.get('source')}** (Relevance: {score:.2f})")
                        st.write(f"*{doc['jurisdiction']}*")
                        st.write(doc['text'][:150] + "..." if len(doc['text']) > 150 else doc['text'])
            
            st.markdown("---")

def extract_citations(text: str) -> List[str]:
    """Extract legal citations from text"""
    patterns = [
        r'Section\s+\d+[A-Z]*',
        r'\d+\s+U\.S\.C\.\s+\d+',
        r'[A-Z][a-z]+\s+Code\s+Section\s+\d+',
        r'Penal Code\s+§?\s*\d+',
        r'IPC\s+Section\s+\d+',
        r'Indian Penal Code\s+Section\s+\d+',
        r'Motor Vehicles Act\s+§?\s*\d+',
        r'धारा\s+\d+',
        r'कलम\s+\d+'
    ]
    
    citations = []
    for pattern in patterns:
        citations.extend(re.findall(pattern, text, re.IGNORECASE))
    
    return list(set(citations))

# Footer with multi-language support
footer_texts = {
    "english": "AI Legal Assistant • Not Legal Advice • Supporting English, Hindi & Marathi",
    "hindi": "AI कानूनी सहायक • कानूनी सलाह नहीं • अंग्रेजी, हिन्दी और मराठी का समर्थन",
    "marathi": "AI कायदेशीर सहाय्यक • कायदेशीर सल्ला नाही • इंग्रजी, हिन्दी आणि मराठी समर्थन"
}

current_footer = footer_texts[st.session_state.current_language]

st.markdown("---")
st.markdown(f"""
<div style='text-align: center; font-size: 0.8rem; color: #666;'>
    <p><em>{current_footer}</em></p>
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()