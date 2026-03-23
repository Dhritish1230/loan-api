from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# Load everything once (fast + efficient)
model = joblib.load("loan_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

@app.get("/")
def home():
    return {"message": "Loan Prediction API Running"}

@app.post("/predict")
def predict(data: dict):
    try:
        # Create full input with ALL columns
        input_data = {col: None for col in feature_columns}
        
        # Update with incoming request
        input_data.update(data)

        df = pd.DataFrame([input_data])

        # Ensure correct column order
        df = df[feature_columns]

        prob = model.predict_proba(df)[:,1][0]

        return {
            "conversion_probability": float(prob),
            "prediction": "Likely to Convert" if prob > 0.1 else "Not Likely"
        }

    except Exception as e:
        return {"error": str(e)}