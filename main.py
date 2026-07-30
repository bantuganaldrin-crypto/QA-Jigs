import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai

app = FastAPI(title="QA Jigs API")

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

class PromptRequest(BaseModel):
    prompt: str

class QARequest(BaseModel):
    article_text: str

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

@app.post("/qa-check")
def qa_check_article(request: QARequest):
    try:
        qa_prompt = f"""
        Act as a strict Managing Editor and Quality Assurance (QA) expert. 
        Evaluate the following news article and provide:
        1. A **QA Score** out of 100 based on clarity, structure, grammar, and journalistic standards.
        2. A clear breakdown of any **Writer Mistakes** (such as redundancies, passive voice, structural flaws, or clarity issues).

        Article text:
        {request.article_text}
        """
        
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=qa_prompt,
        )
        return {"qa_evaluation": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))