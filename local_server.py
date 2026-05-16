import uvicorn
import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from api.index import app as api_app

# Create a wrapper app for local development
local_app = FastAPI()

# Use the api_app directly and add the index route to it for local testing
@api_app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 ResumeAI Pro - Local Development Server")
    print("👉 Access: http://localhost:8000")
    print("="*50 + "\n")
    uvicorn.run(api_app, host="127.0.0.1", port=8000)
