import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from google import genai
from typing import List

app = FastAPI(title="QA Jigs API")

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

class PromptRequest(BaseModel):
    prompt: str

class QARequest(BaseModel):
    article_text: str

# Detailed scoring rubric model
class QAScoreBreakdown(BaseModel):
    grammar_and_syntax: int = Field(..., description="Score out of 100 for grammar, spelling, and sentence correctness")
    structural_flow: int = Field(..., description="Score out of 100 for inverted pyramid structure and transitions")
    journalistic_standards: int = Field(..., description="Score out of 100 for objectivity, quotes handling, and lack of redundancy")
    formatting_and_precision: int = Field(..., description="Score out of 100 for proper naming, typos, and clean presentation")

class EditorialFeedback(BaseModel):
    overall_qa_score: int = Field(..., description="The definitive overall weighted score out of 100")
    status: str = Field(..., description="APPROVED, REVISED, or REJECTED")
    score_breakdown: QAScoreBreakdown
    summary_assessment: str
    grammar_and_syntax_mistakes: List[str]
    structural_and_flow_flaws: List[str]
    journalistic_style_violations: List[str]
    actionable_fixes: List[str]

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

@app.post("/qa-check", response_model=EditorialFeedback)
def qa_check_article(request: QARequest):
    try:
        qa_prompt = f"""
        Act as a strict Managing Editor and Quality Assurance (QA) expert for official press releases.
        Evaluate the article below and return a structured assessment covering:
        1. overall_qa_score (Definitive weighted score out of 100)
        2. status (APPROVED / REVISED / REJECTED)
        3. score_breakdown (Provide precise integer scores out of 100 for grammar_and_syntax, structural_flow, journalistic_standards, and formatting_and_precision)
        4. summary_assessment (Short paragraph overview)
        5. grammar_and_syntax_mistakes (List typos or duplicate sentences)
        6. structural_and_flow_flaws (List issues like name dumps or poor transitions)
        7. journalistic_style_violations (List un-translated local quotes, redundancies, or jargon overuse)
        8. actionable_fixes (List precise steps the writer must take to fix the draft)

        Article text:
        {request.article_text}
        """
        
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=qa_prompt,
            config={
                'response_mime_type': 'application/json',
                'response_schema': EditorialFeedback,
            },
        )
        
        import json
        return json.loads(response.text)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))