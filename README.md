# 🎯 ResumeAI — Production-Grade AI Resume Screening

> Elevate your hiring with AI-powered resume ranking, 3D matched analytics, and Apple-inspired precision.

---

## ✨ Features

- 💎 **Apple Glass Design**: Premium, minimalist white background with glassmorphism components.
- 🧊 **3D Visualization**: Interactive Three.js icosahedron background and GSAP-powered motion design.
- ⚡ **FastAPI Backend**: Production-grade Python backend with asynchronous file processing.
- 🚀 **Groq & Gemini Support**: Ultra-fast screening using Groq (Llama 3.1) with Gemini 1.5 Flash fallback.
- 📄 **Robust Parsing**: Deep PDF text extraction via PyPDF2.
- 🪄 **Interactive UI**: 3D tilt-responsive candidate cards and smooth staggered animations.

---

## 📁 Project Structure

```
Resume-screening/
├── app.py              ← FastAPI backend (Entry point)
├── index.html          ← Premium Frontend (Three.js + GSAP)
├── requirements.txt    ← Production dependencies
├── .env                ← API Keys (Gemini)
├── .gitignore          ← Git protection
└── README.md
```

---

## ⚙️ Setup & Deployment

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
1. Create a `.env` file (copy `.env.example`).
2. Add your **GEMINI_API_KEY** from [Google AI Studio](https://aistudio.google.com/).

### 3. Run the application
```bash
python app.py
```
Open **[http://localhost:8000](http://localhost:8000)** in your browser.

---

## 🛠️ How I Built This

### 1. The Vision: Apple-Inspired Design
I wanted to create more than just a tool; I wanted to build an **experience**. Inspired by Apple's minimalist aesthetic, I implemented:
- **Glassmorphism**: Using high-blur overlays and subtle borders to create depth.
- **3D Interaction**: Integrated **Three.js** to render a live, interactive icosahedron background that responds to user presence.
- **Fluid Motion**: Used **GSAP** (GreenSock) to ensure every button click and result card feels smooth and premium.

### 2. The Engine: Ultra-Fast AI
Speed is critical in recruitment. I chose **FastAPI** for its high performance and integrated two powerful AI layers:
- **Groq (Primary)**: Using the Llama 3.3 70B model via Groq’s LPU, I achieved near-instant screening results.
- **Gemini (Fallback)**: Integrated Google’s Gemini 1.5 Flash as a robust fallback to ensure the system is always intelligent.

### 3. The Logic: Precision Ranking
The core logic involves three main stages:
- **Deep Parsing**: I used `PyPDF2` to extract clean text from complex resume layouts.
- **NLP Analysis**: The AI analyzes the job description against the resume to identify hidden skills and critical gaps.
- **Scoring Algorithm**: A custom-weighted scoring system that evaluates not just keywords, but the overall professional fit.

---

## 🎨 Tech Stack

- **Frontend**: HTML5, Vanilla CSS, JavaScript.
- **Motion**: [Three.js](https://threejs.org/) (3D), [GSAP](https://greensock.com/gsap/) (Animations).
- **Backend**: [FastAPI](https://fastapi.tiangolo.com/), [Uvicorn](https://www.uvicorn.org/).
- **AI/NLP**: [Google Gemini AI](https://deepmind.google/technologies/gemini/), [Scikit-learn](https://scikit-learn.org/) (TF-IDF).
- **File Processing**: PyPDF2.

---

---

Built with ❤️ by me
