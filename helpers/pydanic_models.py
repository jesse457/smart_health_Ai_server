from pydantic import BaseModel, Field # <-- Imported BaseModel and Field
from typing import Any, List, Optional
# --- Pydantic Models ---
# (Ensure HealthData model is correctly defined/imported from data_processor)

class TaskStatusResponse(BaseModel):
    task_id: str
    user_id: Optional[str] = None
    timestamp: Optional[str] = None # ISO format string
    date: Optional[str] = None # ISO format string
    data_points: Optional[int] = None
    status: str
    score: Optional[float] = None
    model_info: Optional[str] = None
    error: Optional[str] = None
    completed_at: Optional[str] = None # ISO format string
    
class DataExplanationResponse(BaseModel):
    user_id: str
    name: str
    data_summary_used: str
    explanation: str
    
class ChatMessage(BaseModel):
    message: str
    history: list[dict] = [] # To maintain conversation history
    
class TrainResponse(BaseModel):
    message: str
    task_id: str
    status: TaskStatusResponse # Use the detailed status model

class PredictionResponse(BaseModel):
    # Define expected prediction output structure
    predicted_value: Any # Or be more specific, e.g., str, float, List[str]

class Recommendation(BaseModel):
    title: str = Field(..., description="Short title for the recommendation", examples=["Mindful Movement Break"])
    description: str = Field(..., description="Detailed explanation of the recommendation", examples=["Incorporate a 5-10 minute walk or stretching session after lunch to aid digestion and boost afternoon focus."])
    category: Optional[str] = Field(None, description="e.g., Diet, Exercise, Sleep, Stress Management, Monitoring", examples=["Exercise"])

class RecommendationResponse(BaseModel):
    user_id: str = Field(..., description="User ID")
    disclaimer: str = "These recommendations are AI-generated based on your data and are not a substitute for professional medical advice. Consult a healthcare provider for any health concerns."
    recommendations: List[Recommendation] = Field(..., description="List of recommendations")
    data_summary_used: str = Field(..., description="Summary of data used for recommendations")


# --- New Pydantic Model for Diagnosis (Conceptual) ---
# Although the endpoint streams text, this model helps define the *intended* structure
# if the streamed data were assembled. The disclaimer is crucial.
class DiagnosisResponse(BaseModel):
    user_id: str
    diagnosis_suggestion: Optional[str] = Field(None, description="Potential AI-generated diagnosis suggestion based on the data.")
    disclaimer: str = "This is an AI-generated suggestion based ONLY on the provided data summary and is NOT a substitute for professional medical advice or diagnosis. It may be incomplete or inaccurate. Consult a qualified healthcare provider for any health concerns or before making any decisions related to your health or treatment."
    data_summary_used: Optional[str] = Field(None, description="Brief summary of the data used for diagnosis generation.")
