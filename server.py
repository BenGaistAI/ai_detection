from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
import joblib
import nltk
from nltk.tokenize import word_tokenize
import traceback # ספרייה חדשה שמדפיסה שגיאות מפורטות

# הורדת כלי החיתוך
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except:
    pass

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading AI Models... Please wait ⏳")
try:
    w2v_model = joblib.load("w2v_model.pkl")
    scaler = joblib.load("scaler.pkl")
    clf_w2v = joblib.load("clf_w2v.pkl")
    print("Models loaded successfully! 🚀")
except Exception as e:
    print(f"🚨 Error loading models: {e}")

class TextRequest(BaseModel):
    text: str

@app.post("/predict")
def predict_text(request: TextRequest):
    text = request.text
    
    try:
        tokens = word_tokenize(text.lower())

        vectors = [
            w2v_model.wv[word]
            for word in tokens
            if word in w2v_model.wv
        ]

        if vectors:
            doc_vec = np.mean(vectors, axis=0)
        else:
            doc_vec = np.zeros(w2v_model.vector_size)

        doc_vec = doc_vec.reshape(1, -1)
        col_names = [f"w2v_{i}" for i in range(w2v_model.vector_size)]
        doc_vec_df = pd.DataFrame(doc_vec, columns=col_names)

        doc_vec_scaled = scaler.transform(doc_vec_df)
        probs = clf_w2v.predict_proba(doc_vec_scaled)[0]
        
        human_prob = probs[0]
        ai_prob = probs[1]

        if ai_prob >= 0.5:
            confidence = int(ai_prob * 100)
            return {"prediction": "AI", "confidence": confidence}
        else:
            confidence = int(human_prob * 100)
            return {"prediction": "Human", "confidence": confidence}

    except Exception as e:
        # פה הקסם: אם המודל קורס, אנחנו נראה את זה מיד!
        print("\n" + "🔥"*20)
        print("🚨 CRITICAL ERROR DURING PREDICTION 🚨")
        print(f"Error Message: {str(e)}")
        print("Detailed Traceback:")
        traceback.print_exc()
        print("🔥"*20 + "\n")
        return {"prediction": "Human", "confidence": 0}