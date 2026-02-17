from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import time
import random

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextRequest(BaseModel):
    text: str

@app.post("/predict")
def predict_text(request: TextRequest):
    time.sleep(1) # אנימציית טעינה
    text = request.text.lower()
    
    # הגרלת טווח מלא של ערכים (0 עד 100) לסיכוי שזה AI
    if "ai" in text or "robot" in text:
        ai_prob = random.randint(40, 100) # "פוש" קטן לכיוון ה-AI אם המילה מופיעה
    else:
        ai_prob = random.randint(0, 100) # טווח מלא לחלוטין
        
    # הלוגיקה החדשה: מי שניצח מקבל את אחוז הביטחון שלו
    if ai_prob >= 50:
        return {"prediction": "AI", "confidence": ai_prob}
    else:
        # אם הסיכוי ל-AI הוא למשל 20%, זה אומר שהסיכוי לאדם הוא 80%
        human_prob = 100 - ai_prob
        return {"prediction": "Human", "confidence": human_prob}