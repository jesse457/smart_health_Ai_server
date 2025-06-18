import json
import logging
import asyncio
import uuid
import os # <-- Added os import
import pandas as pd
from dotenv import load_dotenv
from fastapi import BackgroundTasks, Body, FastAPI, HTTPException, Depends,status,Query,WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field # <-- Imported BaseModel and Field
from datetime import datetime
from typing import Any, Dict, List, Optional,AsyncGenerator
import regex as re # Keep regex if needed elsewhere, not used in diagnosis endpoint itself
from google import genai
from fastapi.responses import StreamingResponse
from sqlalchemy import true
from helpers.functions import log_api_error,log_api_request,gemini_diagnosis_stream,summarize_health_data,train_model_background
from helpers.pydanic_models import TrainResponse,TaskStatusResponse,PredictionResponse,Recommendation,RecommendationResponse,DataExplanationResponse
from helpers.notifications import send_expo_notification_to_user
# Import Google Generative AI library
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Import your DataProcessor and HealthData model
# Make sure this path is correct for your project structure
from data_processor import DataProcessor, HealthData

# --- Logging Configuration ---
log_file = "C:\\Users\\Okiwa Jesse\\Desktop\\logs\\app_logs.jsonl"
# Use a custom formatter for JSON logs
class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            # Add other standard fields if needed
            "logger_name": record.name,
            "pathname": record.pathname,
            "lineno": record.lineno,
        }
        # Add extra fields from the log call
        if hasattr(record, 'extra_data'):
            log_entry.update(record.extra_data)
        # Add exception info if present
        if record.exc_info:
            # Ensure traceback formatting doesn't break JSON
            try:
                exc_text = self.formatException(record.exc_info)
                # Limit length if necessary
                log_entry['exception'] = exc_text[:2000] # Example limit
            except Exception:
                log_entry['exception'] = "Error formatting exception info"
            # Optionally add full traceback separately if needed and managed
            # try:
            #     tb_text = traceback.format_exc()
            #     log_entry['traceback'] = tb_text[:4000] # Example limit
            # except Exception:
            #     log_entry['traceback'] = "Error formatting traceback"

        return json.dumps(log_entry)

# Configure root logger
logger = logging.getLogger() # Get root logger
logger.setLevel(logging.INFO)

# Remove existing handlers if any (to avoid duplicate logs)
for handler in logger.handlers[:]:
    logger.removeHandler(handler)
# Gemini Model Configuration
MODEL_NAME = "gemini-2.0-flash" # Recommended for fast chat/streaming
generation_config = {
    "temperature": 0.7,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 8024,
}



# File Handler with JSON Formatter
try:
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(JsonFormatter())
    logger.addHandler(file_handler)
except Exception as e:
    print(f"Error setting up file logging: {e}") # Log config error to console

# Stream Handler (Console) with basic formatter for readability
stream_handler_basic = logging.StreamHandler()
basic_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler_basic.setFormatter(basic_formatter)
logger.addHandler(stream_handler_basic)

# --- Environment Variables & API Key ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.critical("GEMINI_API_KEY not found in environment variables.")
    # Decide if the app should exit or run without recommendation/diagnosis feature
    # exit("Critical Error: GEMINI_API_KEY missing.") # Option to exit
    print("Warning: GEMINI_API_KEY not found. Recommendation and Diagnosis endpoints will be disabled.")
else:
    try:
        model = genai.Client(api_key=GEMINI_API_KEY)
        print("Gemini API configured successfully.")
        logger.info("Gemini API configured successfully.")
    except Exception as e:
        logger.error("Failed to configure Gemini API", extra={'error': str(e)}, exc_info=True)
        GEMINI_API_KEY = None # Disable feature if config fails

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Health Data API",
    description="API for processing, analyzing health data, providing recommendations, and potential diagnoses.",
    version="1.0.0" # <-- Updated version for new feature
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins for simplicity, restrict in production
    allow_credentials=True,
    allow_methods=["*"], # Allow all methods
    allow_headers=["*"], # Allow all headers
)


# --- Global Variables & Initialization ---
try:
    data_preprocessor = DataProcessor() # Ensure DataProcessor initializes correctly
except Exception as e:
    logger.critical(f"Failed to initialize DataProcessor: {e}", exc_info=True)
    # Depending on severity, you might want to exit the application
    # exit("Critical Error: DataProcessor initialization failed.")
    data_preprocessor = None # Or set to None and handle in endpoints

# In-memory task store (Replace with Redis/DB/Celery for production scalability)
task_status: Dict[str, Dict[str, Any]] = {}
MIN_DATA_POINTS = 20 # For training
MIN_DIAGNOSIS_POINTS = 10 # Minimum points for diagnosis attempt

# --- Helper Functions ---
# --- Dependency Check for Endpoints ---
async def get_data_processor() -> DataProcessor:
    """Dependency to ensure DataProcessor is initialized."""
    if data_preprocessor is None:
        logger.error("DataProcessor not available.")
        raise HTTPException(status_code=503, detail="Service dependency (DataProcessor) not available.")
    return data_preprocessor

# --- API Endpoints ---
@app.post('/train_model',
    response_model=TrainResponse,
    status_code=202, # Use 202 Accepted for async tasks
    tags=["Training"],
    summary="Submit data and trigger model training")
async def train_model(
    health_data: HealthData,
    background_tasks: BackgroundTasks,
    dp: DataProcessor = Depends(get_data_processor) # Use dependency injection
) -> TrainResponse:
    """
    Receives health data, saves it, and initiates model training in the background
    if sufficient data exists for the user. Returns task ID and status.
    """
    task_id = str(uuid.uuid4())
    log_api_request("train_model", health_data.user_id, task_id)

    # Initialize task status early
    task_status[task_id] = {
        "user_id": health_data.user_id,
        "task_id": task_id,
        "status": "pending_data_check",
        "timestamp": datetime.now().isoformat(),
        "date": datetime.now().strftime("%Y-%m-%d"),
    }

    try:
        # 1. Save the newly submitted data point
        dp.save_data(health_data) # Use injected dp
        logger.info(f"Data point saved for user {health_data.user_id}", extra={'extra_data': {"user_id": health_data.user_id, "task_id": task_id}})

        # 2. Load *all* historical data for the user to check count
        # Ensure load_data_by_user_id is implemented in your DataProcessor
        data = dp.load_data_by_user_id(health_data.user_id,health_data.name)
        print(data)
        data_points_count = len(data)# Count data points, handle None case
        print(f"Data points count for user {health_data.user_id}: {data_points_count}")
        task_status[task_id]["data_points"] = data_points_count

        # 3. Decide whether to queue training
        if data is not None or data.empty and data_points_count >= MIN_DATA_POINTS:
            task_status[task_id]["status"] = "queued"
            background_tasks.add_task(
                train_model_background,
                task_id,
                data.copy(), # Pass a copy to avoid issues if dataframe is modified elsewhere
                health_data.user_id,
                health_data.name,
               
                task_status,
                data_preprocessor
            )
            message = f"Model training initiated. {data_points_count} data points found."
            logger.info(message, extra={'extra_data': {"user_id": health_data.user_id, "task_id": task_id}})
            # Status code remains 202 (Accepted) as task is queued
        else:
            task_status[task_id]["status"] = "insufficient_data"
            message = f"Insufficient data ({data_points_count}/{MIN_DATA_POINTS}) for model training. Data saved."
            logger.warning(message, extra={'extra_data': {"user_id": health_data.user_id, "task_id": task_id, "data_points": data_points_count}})
            # Change status code to 200 OK if not training, as the data save was successful.
            # This doesn't work directly in return, FastAPI handles status code based on decorator

        # Return the standardized response - FastAPI uses the decorator status_code (202)
        # The 'status' field within the response clarifies the outcome.
        return TrainResponse(
            message=message,
            task_id=task_id,
            # Construct the status object matching TaskStatusResponse fields
            status=TaskStatusResponse(
                task_id=task_id,
                user_id=task_status[task_id].get("user_id"),
                timestamp=task_status[task_id].get("timestamp"),
                date=task_status[task_id].get("date"),
                data_points=task_status[task_id].get("data_points"),
                status=task_status[task_id].get("status", "unknown"),
                score=task_status[task_id].get("score"),
                model_info=task_status[task_id].get("model_info"),
                error=task_status[task_id].get("error"),
                completed_at=task_status[task_id].get("completed_at"),
            )
        )

    except Exception as e:
        log_api_error("train_model", e, health_data.user_id, task_id)
        # Update task status to failed if an error occurs before queueing
        if task_id in task_status and task_status[task_id]["status"] not in ["queued", "processing", "completed", "failed"]:
            task_status[task_id]["status"] = "failed"
            task_status[task_id]["error"] = str(e)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during data submission/training check: {str(e)}")


@app.get('/task/{task_id}',
    response_model=TaskStatusResponse,
    tags=["Training"],
    summary="Get background task status")
async def get_task_status(task_id: str) -> TaskStatusResponse:
    """Get the status and results of a background model training task."""
    log_api_request("get_task_status", task_id=task_id)

    if task_id not in task_status:
        logger.warning(f"Task ID not found: {task_id}", extra={'extra_data': {"task_id": task_id, "event": "task_not_found"}})
        raise HTTPException(status_code=404, detail="Task not found")

    status_data = task_status[task_id]
    logger.info(f"Returning status for task {task_id}", extra={'extra_data': {"task_id": task_id, "status": status_data.get('status')}})

    # Map the dictionary to the Pydantic model, handling potential missing keys gracefully
    return TaskStatusResponse(
        task_id=task_id,
        user_id=status_data.get("user_id"),
        timestamp=status_data.get("timestamp"),
        date=status_data.get("date"),
        data_points=status_data.get("data_points"),
        status=status_data.get("status", "unknown"), # Default to unknown if missing
        score=status_data.get("score"),
        model_info=status_data.get("model_info"),
        error=status_data.get("error"),
        completed_at=status_data.get("completed_at")
    )

@app.post(
    '/predict',
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Predict anomalies for submitted data"
)
async def predict_anomalies(
  health_data: HealthData,
    background_tasks: BackgroundTasks, # <--- MOVED HERE: Non-default arguments come before default arguments
    dp: DataProcessor = Depends(get_data_processor)
) -> PredictionResponse:
    """
    Predict anomalies based on user's latest health data submission using
    the corresponding trained model.
    Sends a push notification if the data is an anomaly.
    """
    log_api_request("predict_anomalies", health_data.user_id)
    
    try:
        # Ensure predict_anomalies is implemented in DataProcessor
        # This function should ideally return the predicted value AND a boolean indicating anomaly status
        # For this example, let's assume predict_anomalies returns the score,
        # and you have a threshold to determine if it's an anomaly.
        predicted_score = dp.predict_anomalies(health_data)

        # --- Anomaly Detection Logic (Example) ---
        # You need to define how you determine if 'predicted_score' indicates an anomaly.
        # This threshold should be part of your model's logic or configuration.
        ANOMALY_THRESHOLD = 0.8 # Example: If score is above 0.8, it's an anomaly.
        is_anomaly = true

        if is_anomaly:
            notification_title = "Anomaly Detected!"
            notification_body = (
                f"Anomaly score: {predicted_score['anomaly_score']}. "
                "Please review your patient health data."
            )
            notification_data = {
                "type": "anomaly_alert",
                "user_id": health_data.user_id,
                "score": predicted_score['anomaly_score'],
               # Assuming dp might have a timestamp or get it here
            }
            
            # Use background_tasks to send the notification asynchronously
            # This prevents the API response from waiting for the notification to be sent.
            background_tasks.add_task(
                send_expo_notification_to_user,
                user_id=health_data.user_id,
                title=notification_title,
                body=notification_body,
                data=notification_data
            )
            logger.warning(f"Anomaly detected for user {health_data.user_id}. Notification scheduled.",
                           extra={'extra_data': {"user_id": health_data.user_id, "prediction": predicted_score, "is_anomaly": True}})
        else:
            logger.info(f"Prediction successful for user {health_data.user_id}. No anomaly detected.",
                        extra={'extra_data': {"user_id": health_data.user_id, "prediction": predicted_score, "is_anomaly": False}})
        
        return PredictionResponse(predicted_value=predicted_score, is_anomaly=is_anomaly)

    except FileNotFoundError as e:
        log_api_error("predict_anomalies", e, health_data.user_id, status_code=404)
        raise HTTPException(
            status_code=404,
            detail=f"Model or data not found for user {health_data.user_id}. Ensure the model has been trained successfully."
        )
    except NotImplementedError as e:
        log_api_error("predict_anomalies", e, health_data.user_id, status_code=501)
        raise HTTPException(
            status_code=501,
            detail="Prediction functionality is not implemented."
        )
    except Exception as e:
        log_api_error("predict_anomalies", e, health_data.user_id)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# --- Recommendation Endpoint ---
@app.get("/recommendations/{name}/{user_id}",
    response_model=RecommendationResponse,
    summary="Get AI-generated health recommendations",
    tags=["Recommendations & Diagnosis"])
async def get_recommendations(
    name: str,
    user_id: str,
    dp: DataProcessor = Depends(get_data_processor)
) -> RecommendationResponse:
    """
    Generates personalized health recommendations for a user
    based on their historical data.
    """
    log_api_request("get_recommendations", user_id=user_id) # Include name in logs

    if not GEMINI_API_KEY:
        logger.error("Gemini API key not configured, cannot provide recommendations.", extra={'extra_data': {"user_id": user_id, "name": name, "event": "gemini_key_missing"}})
        raise HTTPException(status_code=503, detail="Recommendation service unavailable: AI provider not configured.")

    try:
        # 1. Load historical data using both user_id and name
        data = dp.load_data_by_user_id(user_id, name=name)

        if data is None or data.empty:
            logger.warning(f"No historical data found for user {user_id} (name: {name}) for recommendations.", extra={'extra_data': {"user_id": user_id, "name": name, "event": "no_data_found"}})
            raise HTTPException(status_code=404, detail=f"No historical data found for user {user_id} (name: {name}).")

        min_reco_points = 5
        if len(data) < min_reco_points:
            logger.warning(f"Insufficient data ({len(data)}/{min_reco_points} points) for recommendations for user {user_id} (name: {name}).", extra={'extra_data': {"user_id": user_id, "name": name, "event": "insufficient_data", "data_points": len(data)}})
            # Proceeding, but LLM might state insufficiency

        # 2. Summarize data
        data_summary = summarize_health_data(data)
        logger.info(f"Generated data summary for user {user_id} (name: {name}) recommendations", extra={'extra_data': {"user_id": user_id, "name": name}})

        # 3. Construct Prompt for Recommendations
        prompt =f"""
     You are an AI health assistant providing personalized, actionable recommendations for a non-technical person. Your goal is to offer clear and simple advice.

Analyze the following health data summary for user ID '{{user_id}}' with name '{{name}}':
--- START DATA SUMMARY ---
{{data_summary}}
--- END DATA SUMMARY ---

Based only on this summary, generate 20 concise, actionable health recommendations focusing on areas like diet, exercise, sleep, stress management, or monitoring.

IMPORTANT INSTRUCTIONS:

Format: Respond ONLY with a JSON list of recommendations. Each object should have "title", "description"-> which should be very detailed, and optionally "category" (e.g., Diet, Exercise, Sleep, Stress Management, Monitoring).
DO NOT: Provide medical diagnoses, treatment plans, or medication advice.
DO NOT include any introductory text, explanations, or disclaimers outside the JSON list itself. The API will handle the disclaimer.
If data is insufficient for meaningful recommendations, return an empty JSON list [].
JSON Recommendation List:
        """

        # 4. Call Gemini API (non-streaming for recommendations)
        try:
            logger.info(f"Sending request to Gemini for user {user_id} (name: {name}) recommendations", extra={'extra_data': {"user_id": user_id, "name": name, "event": "gemini_request_start"}})
             # Or gemini-pro
            formatted_prompt = prompt.format(user_id=user_id, name=name, data_summary=data_summary)
            response =  model.models.generate_content(contents=formatted_prompt,model=MODEL_NAME) # Non-streaming call

            # 5. Parse Gemini Response
            raw_response_text = response.text.strip()
            print(raw_response_text)
            logger.info(f"Received raw response from Gemini for {user_id} (name: {name}): {raw_response_text}", extra={'extra_data': {"user_id": user_id, "name": name, "event": "gemini_response_received"}})

            # Attempt to find and parse the JSON list
            recommendations_list = []
            json_match = re.search(r'\[\s*\{.*\}\s*\]', raw_response_text, re.DOTALL) # Look for list [...] containing objects {...}

            if json_match:
                json_str = json_match.group(0)
                try:
                    parsed_list = json.loads(json_str)
                    # Validate structure (simple check)
                    if isinstance(parsed_list, list):
                        for item in parsed_list:
                            if isinstance(item, dict) and "title" in item and "description" in item:
                                recommendations_list.append(Recommendation(**item)) # Use Pydantic model for validation
                            else:
                                logger.warning(f"Gemini response for {user_id} (name: {name}) contained an item that was not a dict with 'title' and 'description': {item}", extra={'extra_data': {"user_id": user_id, "name": name, "item": item}})
                    else:
                        logger.warning(f"Gemini response for {user_id} (name: {name}) recommendations was not a valid JSON list.", extra={'extra_data': {"user_id": user_id, "name": name, "raw_response": raw_response_text}})
                except json.JSONDecodeError as json_e:
                    logger.error(f"Failed to parse JSON recommendations for user {user_id} (name: {name}). Error: {json_e}. Raw text: {raw_response_text}", extra={'extra_data': {"user_id": user_id, "name": name, "raw_response": raw_response_text}})
                    # Fallback or raise error? Return empty list for now.
            else:
                logger.warning(f"Could not find a valid JSON list in Gemini response for {user_id} (name: {name}) recommendations. Raw text: {raw_response_text}", extra={'extra_data': {"user_id": user_id, "name": name, "raw_response": raw_response_text}})


            logger.info(f"Successfully parsed {len(recommendations_list)} recommendations for user {user_id} (name: {name})", extra={'extra_data': {"user_id": user_id, "name": name, "recommendation_count": len(recommendations_list)}})

            return RecommendationResponse(
                user_id=user_id,
                recommendations=recommendations_list,
                data_summary_used=data_summary
            )

        except Exception as gemini_error:
            log_api_error("get_recommendations (Gemini Call)", gemini_error, user_id)
            raise HTTPException(status_code=502, detail=f"AI service error during recommendations: {str(gemini_error)}")

    except HTTPException as http_exc:
        raise http_exc # Re-raise known HTTP exceptions
    except NotImplementedError as e:
        log_api_error("get_recommendations", e, user_id, status_code=501)
        raise HTTPException(status_code=501, detail=str(e))
    except Exception as e:
        log_api_error("get_recommendations", e, user_id)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during recommendations: {str(e)}")

@app.get("/data/explanation/{name}/{user_id}",
            response_model=DataExplanationResponse,
            summary="Get AI-generated explanation and summary of patient health data for caregivers",
            tags=["Data Explanation & Summary"])
async def get_data_explanation_and_summary(
    name: str,
    user_id: str,
    dp: DataProcessor = Depends(get_data_processor)
) -> DataExplanationResponse:
    """
    Generates an easy-to-understand explanation and summary of a patient's
    historical health data for caregivers.
    """
    log_api_request("get_data_explanation_and_summary", user_id=user_id)

    if not GEMINI_API_KEY:
        logger.error("Gemini API key not configured, cannot provide data explanation.", extra={'extra_data': {"user_id": user_id, "name": name, "event": "gemini_key_missing"}})
        raise HTTPException(status_code=503, detail="Data explanation service unavailable: AI provider not configured.")

    try:
        # 1. Load historical data using both user_id and name
        data = dp.load_data_by_user_id(user_id, name=name)

        if data is None or data.empty: # Check if data is empty or None
            logger.warning(f"No historical data found for patient {user_id} (name: {name}) for explanation.", extra={'extra_data': {"user_id": user_id, "name": name, "event": "no_data_found"}})
            raise HTTPException(status_code=404, detail=f"No historical data found for patient {user_id} (name: {name}).")

        min_explanation_points = 5 # Minimum data points to provide a meaningful explanation
        if len(data) < min_explanation_points:
            logger.warning(f"Insufficient data ({len(data)}/{min_explanation_points} points) for a comprehensive explanation for patient {user_id} (name: {name}).", extra={'extra_data': {"user_id": user_id, "name": name, "event": "insufficient_data", "data_points": len(data)}})
            # The prompt will handle the "insufficient data" scenario within the explanation.

        # 2. Summarize data
        data_summary = summarize_health_data(data)
        logger.info(f"Generated data summary for patient {user_id} (name: {name}) explanation", extra={'extra_data': {"user_id": user_id, "name": name}})

        # 3. Construct Prompt for Explanation and Summary
        prompt = f"""
You are an AI health assistant explaining health data to a caregiver about their patient. Your goal is to provide a clear, concise, and easy-to-understand explanation and summary of the patient's health data.
Analyze the following health data summary for patient ID '{user_id}' with name '{name}':

--- START PATIENT DATA SUMMARY ---

{data_summary}

--- END PATIENT DATA SUMMARY ---

Based ONLY on this summary, explain and summarize the key health trends, notable values, and overall health status of the patient in the simplest possible terms for a caregiver. Focus on what's important for the caregiver to understand about their patient's health at a glance, and how this data might inform their caregiving.

DO NOT: Provide medical diagnoses, treatment plans, or medication advice.
DO NOT include any introductory text, explanations, or disclaimers. Provide only the explanation and summary text.

If data is insufficient for a meaningful explanation, state briefly that the data is limited and an in-depth explanation for the patient cannot be provided yet.
"""
        # 4. Call Gemini API (non-streaming)
        try:
            logger.info(f"Sending request to Gemini for patient {user_id} (name: {name}) explanation", extra={'extra_data': {"user_id": user_id, "name": name, "event": "gemini_request_start"}})
            formatted_prompt = prompt.format(user_id=user_id, name=name, data_summary=data_summary)
            response = model.models.generate_content(contents=formatted_prompt, model=MODEL_NAME) # Non-streaming call

            # 5. Parse Gemini Response (expecting plain text)
            explanation_text = response.text.strip()
            print(explanation_text) # For debugging in console
            logger.info(f"Received raw explanation from Gemini for {user_id} (name: {name}): {explanation_text}", extra={'extra_data': {"user_id": user_id, "name": name, "event": "gemini_response_received"}})

            logger.info(f"Successfully generated explanation for patient {user_id} (name: {name})", extra={'extra_data': {"user_id": user_id, "name": name}})

            return DataExplanationResponse(
                user_id=user_id,
                name=name,
                data_summary_used=data_summary,
                explanation=explanation_text
            )

        except Exception as gemini_error:
            log_api_error("get_data_explanation_and_summary (Gemini Call)", gemini_error, user_id)
            raise HTTPException(status_code=502, detail=f"AI service error during data explanation: {str(gemini_error)}")

    except HTTPException as http_exc:
        raise http_exc # Re-raise known HTTP exceptions
    except NotImplementedError as e:
        log_api_error("get_data_explanation_and_summary", e, user_id, status_code=501)
        raise HTTPException(status_code=501, detail=str(e))
    except Exception as e:
        log_api_error("get_data_explanation_and_summary", e, user_id)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during data explanation: {str(e)}")

# --- NEW Diagnosis Endpoint (Streaming) ---
@app.websocket("/ws/diagnosis/{user_id}/{name}")
async def websocket_diagnosis_stream(
    websocket: WebSocket,
    user_id: str,
    name: str,
    symptoms: str = None, # Optional: if you want to pass initial symptoms via URL query param
    dp: DataProcessor = Depends(get_data_processor)
):
    await websocket.accept()
    full_response_for_log = ""
    chunk_count = 0

    try:
        logger.info(
            f"Starting WebSocket diagnosis stream for user {user_id} (name: {name})",
            extra={'extra_data': {"user_id": user_id, "name": name, "event": "websocket_diagnosis_stream_start"}}
        )

        if not GEMINI_API_KEY:
            error_msg = "AI provider not configured."
            await websocket.send_text(json.dumps({"type": "error", "message": error_msg}))
            logger.error(f"WebSocket: Gemini API key not configured for user {user_id}", extra={'extra_data': {"user_id": user_id, "name": name, "event": "gemini_key_missing"}})
            return # Close connection

        # 1. Load historical data
        data = dp.load_data_by_user_id(user_id, name=name)

        if data is None or data.empty:
            no_data_msg = f"No historical data found for user {user_id} (name: {name})."
            await websocket.send_text(json.dumps({"type": "error", "message": no_data_msg}))
            logger.warning(f"WebSocket: {no_data_msg}", extra={'extra_data': {"user_id": user_id, "name": name, "event": "no_data_found"}})
            return # Close connection

        # 2. Summarize data
        data_summary = summarize_health_data(data)
        logger.info(f"WebSocket: Generated data summary for user {user_id} (name: {name}) diagnosis", extra={'extra_data': {"user_id": user_id, "name": name}})

        # Send initial disclaimer
        disclaimer_text = "AI-generated diagnoses are for informational purposes only and should not replace professional medical advice. Always consult a healthcare professional for any health concerns."
        await websocket.send_text(json.dumps({"type": "disclaimer", "content": disclaimer_text}))

        # 3. Construct Prompt for Diagnosis
        prompt = f"""
You are an AI health assistant specializing in providing diagnostic insights based on provided data and symptoms. Your goal is to offer clear, concise, and actionable diagnostic information.

Analyze the following health data summary for user ID '{user_id}' with name '{name}':
--- START DATA SUMMARY ---
{data_summary}
--- END DATA SUMMARY ---

{f"The user also reported the following symptoms: {symptoms}" if symptoms else ""}

Based only on this summary (and optional symptoms), provide a general health assessment and potential areas of concern.
DO NOT: Provide definitive medical diagnoses, prescribe treatments, or recommend specific medications.
Focus on: Identifying patterns, highlighting potential risks, and suggesting general next steps like consulting a doctor or monitoring certain metrics.
Format your response as continuous text.
"""
        # 4. Stream Gemini Diagnosis Chunks
        try:
            logger.info(f"WebSocket: Sending streaming request to Gemini for user {user_id} (name: {name}) diagnosis", extra={'extra_data': {"user_id": user_id, "name": name, "event": "gemini_stream_request_start"}})
            response_stream = model.models.generate_content_stream(contents=prompt, model=MODEL_NAME)

            for chunk in response_stream:
                if hasattr(chunk, 'text') and chunk.text:
                    full_response_for_log += chunk.text
                    # Send text chunks
                    await websocket.send_text(json.dumps({"type": "text", "content": chunk.text}))
                    chunk_count += 1
                elif chunk.prompt_feedback:
                    logger.warning(
                        f"WebSocket: Gemini safety feedback received for user {user_id}: {chunk.prompt_feedback}",
                        extra={'extra_data': {"user_id": user_id, "name": name, "event": "gemini_safety_feedback", "feedback": str(chunk.prompt_feedback)}}
                    )
                    if chunk.prompt_feedback.block_reason:
                        error_msg = f"AI response blocked due to safety concerns: {chunk.prompt_feedback.block_reason}"
                        await websocket.send_text(json.dumps({"type": "error", "message": error_msg}))
                        return # Stop streaming if content is blocked

        except Exception as gemini_stream_error:
            error_msg = f"AI service error during diagnosis streaming: {str(gemini_stream_error)}"
            await websocket.send_text(json.dumps({"type": "error", "message": error_msg}))
            log_api_error("websocket_diagnosis_stream (Gemini)", gemini_stream_error, user_id, name=name)

    except WebSocketDisconnect:
        logger.info(
            f"WebSocket: Client disconnected from diagnosis stream for user {user_id}.",
            extra={'extra_data': {"user_id": user_id, "name": name, "event": "websocket_disconnect", "chunk_count": chunk_count}}
        )
    except Exception as general_error:
        error_msg = f"An unexpected error occurred during diagnosis WebSocket: {str(general_error)}"
        try:
            await websocket.send_text(json.dumps({"type": "error", "message": error_msg}))
        except RuntimeError:
            pass # Client might have already disconnected
        log_api_error("websocket_diagnosis_stream (General)", general_error,user_id)
    finally:
        logger.info(
            f"WebSocket: Diagnosis stream finished for user {user_id}. Sent {chunk_count} chunks.",
            extra={'extra_data': {"user_id": user_id, "name": name, "event": "websocket_diagnosis_stream_complete", "chunk_count": chunk_count}}
        )
        logger.debug(
            f"WebSocket: Full Gemini diagnosis response (logged internally) for {user_id}: {full_response_for_log}",
            extra={'extra_data': {"user_id": user_id, "name": name, "full_response": full_response_for_log}}
        )
        # Send a final completion event
        try:
            await websocket.send_text(json.dumps({"type": "completion", "status": "success", "message": "Diagnosis stream completed."}))
            # You might want to close the connection explicitly here,
            # but WebSocketDisconnect usually handles it.
            # await websocket.close()
        except RuntimeError:
            pass # Client might have already disconnected
@app.get("/health", status_code=200, tags=["Health"])
async def health_check():
    """Basic health check endpoint."""
    # Add more checks: DataProcessor status, basic Gemini ping?
    health_status = {"status": "ok", "timestamp": datetime.now().isoformat(), "version": app.version}
    status_code = 200 # Default OK

    if data_preprocessor is None:
        health_status["data_processor"] = "unavailable"
        health_status["status"] = "degraded"
        status_code = 503
    else:
        # Add more specific check if possible, e.g., check data path existence
        health_status["data_processor"] = "available" # Assuming init means available

    if not GEMINI_API_KEY:
        health_status["gemini_service"] = "unavailable (API key missing)"
        health_status["status"] = "degraded"
        status_code = 503 # Service depends on Gemini
    else:
        health_status["gemini_service"] = "configured"
        # TODO: Consider adding a quick non-streaming test call to Gemini here
        # like listing models, but be mindful of latency/cost.
        # For now, 'configured' means the key exists.

   
    return health_status # Simpler approach


