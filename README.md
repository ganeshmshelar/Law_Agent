# ⚖️ Law AI Agent - Multi-language Legal Assistant

A Streamlit-based AI legal assistant that provides legal information support in English, Hindi, and Marathi with automatic language detection and fast response capabilities.

## 🌟 Features

- **Multi-language Support**: Full support for English, Hindi, and Marathi
- **Automatic Language Detection**: Intelligently detects input language
- **Fast Semantic Search**: FAISS-powered vector search for relevant legal documents
- **Dual AI Backend**: Integration with Google Gemini and Groq APIs with fallback
- **Comprehensive Legal Database**: Pre-loaded sample legal data across multiple jurisdictions
- **Real-time Analysis**: Quick legal information and penalty analysis
- **Citation Extraction**: Automatic identification of legal citations

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- API keys for Gemini and/or Groq (optional, has local fallback)

### Installation

1. **Clone or download the project files**

2. **Install required packages:**
```bash
pip install streamlit sentence-transformers faiss-cpu python-dotenv requests pandas numpy
```

3. **Set up environment variables:**
Create a `.env` file in the project directory:
```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-1.5-flash
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama3-70b-8192
```

4. **Run the application:**
```bash
streamlit run app.py
```

## 📋 Requirements

The application requires the following Python packages:

```txt
streamlit
sentence-transformers
faiss-cpu
python-dotenv
requests
pandas
numpy
```

Install all requirements with:
```bash
pip install -r requirements.txt
```

## 🛠️ Configuration

### API Keys (Optional)

- **Google Gemini**: Get API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
- **Groq**: Get API key from [Groq Cloud](https://console.groq.com/keys)

### Model Settings

- **Embedding Model**: `all-MiniLM-L6-v2` (default for semantic search)
- **LLM Models**: Configurable via environment variables
- **Search Settings**: Top K=3, Similarity Threshold=0.7

## 💡 Usage

1. **Select Language**: Choose between English, Hindi, or Marathi
2. **Ask Questions**: Type legal questions in natural language
3. **Get Analysis**: Receive AI-powered legal information with citations
4. **Review Sources**: Expand sections to see relevant laws used

### Example Queries

**English:**
- "What are the penalties for drunk driving?"
- "Murder punishment in India"
- "Theft laws and penalties"

**Hindi:**
- "शराब पीकर गाडी चलाने की सजा"
- "भारत में हत्या की सजा"
- "चोरी के कानून और दंड"

**Marathi:**
- "दारू पिऊन गाडी चालवल्याची शिक्षा"
- "भारतात खुनाची शिक्षा"
- "चोरीचे कायदे आणि शिक्षा"

## 🗂️ Project Structure

```
law-ai-agent/
├── app.py                 # Main Streamlit application
├── .env                  # Environment variables (create this)
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🔧 Technical Details

### Architecture

- **Frontend**: Streamlit web interface
- **Search Engine**: FAISS vector similarity search
- **AI Models**: Google Gemini + Groq LLM APIs
- **Language Processing**: Custom detection and translation prompts
- **Data Storage**: In-memory session state with pre-loaded legal data

### Key Components

1. **Language Detection**: Regex-based Unicode character analysis
2. **Semantic Search**: Sentence transformers + FAISS index
3. **Multi-LLM Integration**: Fallback mechanism between APIs
4. **Local Database**: Comprehensive sample legal data in 3 languages
5. **Response Generation**: Context-aware legal analysis

## ⚠️ Important Disclaimer

**This application provides informational assistance only and does not constitute legal advice.** 

- Always consult qualified legal professionals for legal matters
- The information provided may not be complete or up-to-date
- Legal penalties and procedures vary by jurisdiction
- This tool is for educational and informational purposes only

## 🎯 Supported Legal Areas

- Criminal law (murder, theft, etc.)
- Traffic violations (DUI/drunk driving)
- General legal principles
- Multi-jurisdictional comparisons (India, California examples)

## 🔄 Fallback System

The application includes robust fallback mechanisms:

1. **Primary**: Google Gemini API
2. **Secondary**: Groq API  
3. **Tertiary**: Local database with rule-based responses

## 🌐 Language Support Details

- **English**: Full legal terminology support
- **Hindi** (हिन्दी): Devanagari script with legal terminology
- **Marathi** (मराठी): Devanagari script with regional legal terms

## 📞 Support

For issues or questions:
1. Check that all dependencies are installed
2. Verify API keys in `.env` file
3. Ensure internet connectivity for API calls
4. Check browser console for any errors

## 📄 License

This project is for educational and informational purposes. Please ensure compliance with API terms of service and local regulations when deploying.

---

**Note**: This application is designed to demonstrate AI capabilities in legal information retrieval and should not be used as a substitute for professional legal counsel.
