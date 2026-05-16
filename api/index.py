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

    # Run in parallel
    results = await asyncio.gather(
        call_groq_analysis(prompt),
        call_gemini_analysis(prompt),
        return_exceptions=True
    )

    valid_results = [r for r in results if isinstance(r, dict) and r]
    
    if not valid_results:
        return None

    # Merge / Consensus logic
    # We average the scores for higher accuracy
    avg_score = sum(r.get("score", 0) for r in valid_results) / len(valid_results)
    avg_ats = sum(r.get("ats_score", 0) for r in valid_results) / len(valid_results)
    
    # Use the first valid result for structured data but override scores
    final_data = valid_results[0].copy()
    final_data["score"] = round(avg_score, 1)
    final_data["ats_score"] = round(avg_ats, 1)
    final_data["method"] = f"Consensus ({len(valid_results)} models)"
    
    return final_data

# --- Routes ---

@app.post("/api/screen")
async def screen_resumes(
    jd: str = Form(...),
    files: List[UploadFile] = File(...)
):
    results = []
    jd_clean = clean_text(jd)

    # Process files concurrently for faster results
    async def process_file(file):
        content = await file.read()
        if file.filename.lower().endswith('.pdf'):
            text = extract_text_from_pdf(content)
        else:
            text = content.decode('utf-8', errors='ignore')
        
        text_clean = clean_text(text)
        if not text_clean: return None

        return await get_consensus_analysis(text_clean, jd_clean)

    # Note: For many files, we might want to limit concurrency
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
async def predict_fit(
    jd: str = Form(...),
    resume_text: str = Form(...)
):
    """Instant fit predictor for a single JD and single resume text."""
    analysis = await get_consensus_analysis(clean_text(resume_text), clean_text(jd))
    if not analysis:
        return JSONResponse({"error": "Analysis failed"}, status_code=500)
    return analysis
