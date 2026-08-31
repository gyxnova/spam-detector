from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field

from src.models.predict_model import predict as run_predict

app = FastAPI(title="Spam Classifier API", description="A simple API for classifying messages as spam or ham.", version="1.0.0")

class MessageInput(BaseModel):
    text: str = Field(...,min_length=1,max_length=10000)

@app.get('/health')
def health_check():
    """Health check endpoint to verify that the API is running."""
    return {"status": "ok"}

@app.post('/predict')
def predict_endpoint(payload: MessageInput):
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")
    result = run_predict(payload.text)  
    return result  