from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field

from src.models.predict_model import predict as run_predict
from src.models.explain import explain as run_explain

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

@app.post("/explain")
def explain_endpoint(payload: MessageInput):
    if not payload.text.strip():
        raise HTTPException(400, "text cannot be empty")

    result = run_predict(payload.text)
    try:
        explanation = run_explain(payload.text, result["label"], result["confidence"])
    except RuntimeError as e:
        raise HTTPException(503, str(e))

    return {**result, "explanation": explanation}