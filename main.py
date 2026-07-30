import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai

app = FastAPI(title="QA Jigs API")

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

class PromptRequest(BaseModel):
    prompt: str

@app.get("/")
def read_root():
    return {"status": "online", "message": "QA Jigs API is running successfully!"}

@app.post("/generate")
def generate_text(request: PromptRequest):
    try:
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=request.prompt,
        )
        return {"response": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))