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

## 🎨 Tech Stack

- **Frontend**: HTML5, Vanilla CSS, JavaScript.
- **Motion**: [Three.js](https://threejs.org/) (3D), [GSAP](https://greensock.com/gsap/) (Animations).
- **Backend**: [FastAPI](https://fastapi.tiangolo.com/), [Uvicorn](https://www.uvicorn.org/).
- **AI/NLP**: [Google Gemini AI](https://deepmind.google/technologies/gemini/), [Scikit-learn](https://scikit-learn.org/) (TF-IDF).
- **File Processing**: PyPDF2.

---

## 📊 Deployment Guide

- **Vercel/Netlify**: Deploy the frontend easily.
- **Heroku/Render**: Deploy the FastAPI backend.
- **Docker**: The app is structured for easy containerization.

---

Built with ❤️ by the Antigravity Team
