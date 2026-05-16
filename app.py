import os
import json
import logging
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
import PyPDF2
import io
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ResumeAI API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- AI Clients Setup ---

# 1. Groq Setup (Fastest)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = None
if GROQ_API_KEY:
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        logger.info("Groq AI initialized.")
    except Exception as e:
        logger.error(f"Failed to init Groq: {e}")

# 2. Gemini Setup (Powerful)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_model = None
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        logger.info("Gemini AI initialized.")
    except Exception as e:
        logger.error(f"Failed to init Gemini: {e}")

# --- Helper Functions ---

def extract_text_from_pdf(file_content: bytes) -> str:
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        logger.error(f"Error extracting PDF text: {e}")
        return ""

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def calculate_score_tfidf(resume_text: str, jd_text: str) -> float:
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return float(similarity[0][0]) * 100

async def get_ai_analysis(resume_text: str, jd_text: str):
    prompt = f"""
    You are an expert recruiter. Analyze the following resume against the job description.
    
    JOB DESCRIPTION:
    {jd_text[:2000]}
    
    RESUME:
    {resume_text[:4000]}
    
    Return a JSON object with:
    1. "score": (0-100 based on fit)
    2. "matched_skills": [list of 5 key skills matched]
    3. "missing_skills": [list of 3 key gaps]
    4. "summary": (2 sentence professional assessment)
    
    Return ONLY raw JSON, no markdown formatting.
    """

    # Try Groq first
    if groq_client:
        try:
            logger.info("Using Groq for analysis...")
            chat_completion = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                response_format={"type": "json_object"}
            )
            return json.loads(chat_completion.choices[0].message.content)
        except Exception as e:
            logger.error(f"Groq error: {e}")

    # Fallback to Gemini
    if gemini_model:
        try:
            logger.info("Using Gemini for analysis...")
            response = gemini_model.generate_content(prompt)
            match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception as e:
            logger.error(f"Gemini error: {e}")

    return None

# --- Routes ---

@app.post("/api/screen")
async def screen_resumes(
    jd: str = Form(...),
    files: List[UploadFile] = File(...)
):
    results = []
    jd_clean = clean_text(jd)

    for file in files:
        content = await file.read()
        if file.filename.lower().endswith('.pdf'):
            text = extract_text_from_pdf(content)
        else:
            text = content.decode('utf-8', errors='ignore')
        
        text_clean = clean_text(text)
        
        ai_data = await get_ai_analysis(text_clean, jd_clean)
        
        if ai_data:
            result = {
                "name": file.filename,
                "score": ai_data.get("score", 0),
                "matched": ai_data.get("matched_skills", []),
                "missing": ai_data.get("missing_skills", []),
                "analysis": ai_data.get("summary", ""),
                "method": "Groq/Gemini AI"
            }
        else:
            score = calculate_score_tfidf(text_clean, jd_clean)
            result = {
                "name": file.filename,
                "score": round(score, 1),
                "matched": [],
                "missing": [],
                "analysis": "TF-IDF similarity analysis (AI APIs unavailable).",
                "method": "TF-IDF"
            }
        
        results.append(result)

    results.sort(key=lambda x: x['score'], reverse=True)
    return {"results": results}

@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 ResumeAI Server with Groq Support")
    print("👉 Access: http://localhost:8000")
    print("="*50 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
