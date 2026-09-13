from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import mlflow
import os


app=FastAPI()

mlflow.set_tracking_uri(
    os.getenv("MLFLOW_TRACKING_URI","sqlite:///../mlflow_project/mlflow.db")
)
model=mlflow.sklearn.load_model(
    "/mlflow_project/mlruns/0/models/m-0e84bbe940a04000884e4c1bf8cb5544/artifacts"
)

class IrisData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
class PredictionResponse(BaseModel):
    prediction: int
    probability: float

@app.get("/")
def home():
   return {"message": "ML API is running successfully", "version": "1.0"}
@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict",response_model=PredictionResponse)
def predict(data:IrisData):
    features = np.array([[
                    data.sepal_length,
                    data.sepal_width,
                    data.petal_length,
                    data.petal_width
    ]])
    prediction = model.predict(features)
    probabilities = model.predict_proba(features)
    predicted_class=int(prediction[0])
    predicted_probability=probabilities[0][predicted_class]

    return {
        "prediction":predicted_class,
        "probability": float(predicted_probability)
    }



