import os
import json
import logging
import asyncio
import re
import io
from typing import List, Optional, Dict
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import PyPDF2
import google.generativeai as genai
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ResumeAI Pro - Advanced Engine")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- AI Clients Setup ---

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = None
if GROQ_API_KEY:
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
    except Exception as e:
        logger.error(f"Failed to init Groq: {e}")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_model = None
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        logger.error(f"Failed to init Gemini: {e}")

# --- Helper Functions ---

def extract_text_from_pdf(file_content: bytes) -> str:
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text
    except Exception as e:
        logger.error(f"Error extracting PDF text: {e}")
        return ""

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

async def call_groq_analysis(prompt: str) -> Optional[Dict]:
    if not groq_client: return None
    try:
        loop = asyncio.get_event_loop()
        # Use run_in_executor for synchronous Groq client
        chat_completion = await loop.run_in_executor(
            None, 
            lambda: groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                response_format={"type": "json_object"}
            )
        )
        return json.loads(chat_completion.choices[0].message.content)
    except Exception as e:
        logger.error(f"Groq analysis error: {e}")
        return None

async def call_gemini_analysis(prompt: str) -> Optional[Dict]:
    if not gemini_model: return None
    try:
        response = await asyncio.to_thread(gemini_model.generate_content, prompt)
        match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if match:
            return json.loads(match.group())
        return None
    except Exception as e:
        logger.error(f"Gemini analysis error: {e}")
        return None

def calculate_score_tfidf(resume_text: str, jd_text: str) -> float:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        return float(similarity[0][0]) * 100
    except Exception:
        return 0.0

async def get_consensus_analysis(resume_text: str, jd_text: str):
    prompt = f"""
    You are an AI Recruitment Engine (ResumeAI Pro). Analyze the following resume against the job description with high precision.
    
    JOB DESCRIPTION:
    {jd_text[:2500]}
    
    RESUME:
    {resume_text[:5000]}
    
    Return a STRICT JSON object with the following structure:
    {{
      "parsing": {{
        "name": "full name",
        "contact": "email/phone",
        "education": "highest degree and institution",
        "top_experience": "most relevant last role"
      }},
      "score": (0-100 based on exact skills, seniority, and industry fit),
      "ats_score": (0-100 based on formatting, keyword density, and readability for machines),
      "matched_skills": ["skill1", "skill2", "skill3", "skill4", "skill5"],
      "missing_skills": ["gap1", "gap2", "gap3"],
      "bias_check": "Flag any discriminatory language in JD or Resume, or 'Clean'",
      "improvement_tips": ["tip1", "tip2", "tip3"],
      "summary": "2 sentence expert assessment"
    }}
    
    Return ONLY raw JSON.
    """

    # Run AI in parallel
    results = await asyncio.gather(
        call_groq_analysis(prompt),
        call_gemini_analysis(prompt),
        return_exceptions=True
    )

    valid_results = [r for r in results if isinstance(r, dict) and r]
    
    if valid_results:
        # Merge / Consensus logic
        avg_score = sum(r.get("score", 0) for r in valid_results) / len(valid_results)
        avg_ats = sum(r.get("ats_score", 0) for r in valid_results) / len(valid_results)
        
        final_data = valid_results[0].copy()
        final_data["score"] = round(avg_score, 1)
        final_data["ats_score"] = round(avg_ats, 1)
        final_data["method"] = f"Consensus ({len(valid_results)} models)"
        return final_data

    # --- FALLBACK: TF-IDF if AI is offline/missing keys ---
    logger.warning("AI models unavailable. Falling back to TF-IDF.")
    fallback_score = calculate_score_tfidf(resume_text, jd_text)
    return {
        "parsing": {"name": "Candidate", "contact": "N/A", "education": "N/A", "top_experience": "N/A"},
        "score": round(fallback_score, 1),
        "ats_score": 50.0,
        "matched_skills": [],
        "missing_skills": [],
        "bias_check": "Clean (AI Offline)",
        "improvement_tips": ["Connect AI API keys for deep analysis"],
        "summary": "Basic similarity analysis (Fallback mode). AI Consensus engine is currently offline.",
        "method": "TF-IDF Fallback"
    }

# --- Routes ---

@app.post("/api/screen")
@app.post("/screen")  # Alias for Vercel pathing
async def screen_resumes(
    jd: str = Form(...),
    files: List[UploadFile] = File(...)
):
    results = []
    jd_clean = clean_text(jd)

    async def process_file(file):
        content = await file.read()
        if file.filename.lower().endswith('.pdf'):
            text = extract_text_from_pdf(content)
        else:
            text = content.decode('utf-8', errors='ignore')
        
        text_clean = clean_text(text)
        if not text_clean: return None

        # Add 8s timeout to prevent Vercel 10s kill
        try:
            return await asyncio.wait_for(get_consensus_analysis(text_clean, jd_clean), timeout=8.0)
        except asyncio.TimeoutError:
            logger.warning(f"Timeout for {file.filename}, using fallback")
            # Return a simple fallback directly to avoid another nested call
            score = calculate_score_tfidf(text_clean, jd_clean)
            return {
                "parsing": {"name": file.filename, "contact": "N/A", "education": "N/A", "top_experience": "N/A"},
                "score": round(score, 1),
                "ats_score": 50.0,
                "matched_skills": [],
                "missing_skills": [],
                "bias_check": "Timeout",
                "summary": "AI took too long. Showing basic similarity analysis.",
                "method": "Timeout Fallback"
            }

    file_tasks = [process_file(f) for f in files]
    analyses = await asyncio.gather(*file_tasks)

    for i, analysis in enumerate(analyses):
        if analysis:
            results.append({
                "name": files[i].filename,
                **analysis
            })

    results.sort(key=lambda x: x['score'], reverse=True)
    return {"results": results}

@app.post("/api/predict-fit")
@app.post("/predict-fit")  # Alias for Vercel pathing
async def predict_fit(
    jd: str = Form(...),
    file: UploadFile = File(...)
):
    """Instant fit predictor for a single JD and an uploaded resume file."""
    content = await file.read()
    if file.filename.lower().endswith('.pdf'):
        text = extract_text_from_pdf(content)
    else:
        text = content.decode('utf-8', errors='ignore')
    
    text_clean = clean_text(text)
    if not text_clean:
        return JSONResponse({"error": "Empty resume file"}, status_code=400)

    try:
        analysis = await asyncio.wait_for(get_consensus_analysis(text_clean, clean_text(jd)), timeout=8.5)
    except asyncio.TimeoutError:
        score = calculate_score_tfidf(text_clean, clean_text(jd))
        analysis = {
            "parsing": {"name": file.filename, "contact": "N/A", "education": "N/A", "top_experience": "N/A"},
            "score": round(score, 1),
            "ats_score": 50.0,
            "matched_skills": [],
            "missing_skills": [],
            "bias_check": "Timeout",
            "summary": "The AI analysis timed out. Falling back to basic keyword similarity.",
            "method": "Timeout Fallback",
            "improvement_tips": ["Try a shorter job description or check back later."]
        }
    
    if not analysis:
        return JSONResponse({"error": "Analysis failed"}, status_code=500)
    
    analysis["name"] = file.filename
    return analysis
