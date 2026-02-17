from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import time

app = FastAPI()

# הרשאה לאתר לדבר עם השרת
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
    time.sleep(1) # השהייה קלה בשביל האנימציה באתר
    
    text = request.text
    # לוגיקה מדומה זמנית - נחבר את המודל האמיתי בהמשך
    if "רובוט" in text or "AI" in text:
        return {"prediction": "AI 🤖"}
    else:
        return {"prediction": "בן אדם 🧑‍🦱"}