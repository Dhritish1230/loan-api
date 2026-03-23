from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("loan_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

@app.get("/")
def home():
    return {"message": "Loan Prediction API Running"}

@app.post("/predict")
def predict(data: dict):
    
    df = pd.DataFrame([data])
    X = preprocessor.transform(df)
    
    prob = model.predict_proba(X)[:,1][0]
    
    return {
        "conversion_probability": float(prob),
        "prediction": "Likely" if prob > 0.1 else "Not Likely"
    }