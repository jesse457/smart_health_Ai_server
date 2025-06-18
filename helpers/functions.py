import json
import pandas as pd
from fastapi import  HTTPException
from datetime import datetime
from typing import  Dict, Optional,AsyncGenerator
from data_processor import DataProcessor
from helpers.pydanic_models import DiagnosisResponse
import logging

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

def log_api_request(endpoint: str, user_id: Optional[str] = None, task_id: Optional[str] = None, extra_data: Optional[Dict] = None):
    """Helper to log API request starts."""
    log_details = {"event": "api_request_start", "endpoint": endpoint}
    if user_id: log_details["user_id"] = user_id
    if task_id: log_details["task_id"] = task_id
    if extra_data: log_details.update(extra_data)
    # Use logger instance directly
    logger.info(f"Request received for {endpoint}", extra={'extra_data': log_details})

def log_api_error(endpoint: str, error: Exception, user_id: Optional[str] = None, task_id: Optional[str] = None, status_code: int = 500):
    """Helper to log API errors consistently."""
    log_details = {
        "event": "api_request_error",
        "endpoint": endpoint,
        "error": str(error),
        "error_type": type(error).__name__,
        "status_code": status_code,
    }
    if user_id: log_details["user_id"] = user_id
    if task_id: log_details["task_id"] = task_id

    logger.error(f"Error in {endpoint}: {str(error)}", exc_info=True, extra={'extra_data': log_details})


def summarize_health_data(df: pd.DataFrame) -> str:
    """Creates a textual summary of health data for the LLM prompt."""
    if df is None or df.empty:
        return "No historical data available."

    summary_parts = []
    num_records = len(df)

    # Ensure 'timestamp' column exists and is datetime type
    if 'timestamp' in df.columns and pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp']) # Ensure conversion
        min_date = df['timestamp'].min().strftime('%Y-%m-%d') if not df['timestamp'].isnull().all() else "N/A"
        max_date = df['timestamp'].max().strftime('%Y-%m-%d') if not df['timestamp'].isnull().all() else "N/A"
        summary_parts.append(f"Summary based on {num_records} records from {min_date} to {max_date}.")
    else:
        summary_parts.append(f"Summary based on {num_records} records. (Timestamp info missing or invalid).")


    # Example: Summarize numeric columns (adjust column names based on your HealthData)
    numeric_cols = df.select_dtypes(include=['number']).columns
    # Customize this list based on actual meaningful numeric columns in your HealthData
    relevant_numeric_cols = [col for col in numeric_cols if col not in ['user_id', 'name'] and df[col].nunique() > 1] # Filter out identifiers and constant columns

    if not relevant_numeric_cols:
        summary_parts.append("No relevant numeric health metrics found or metrics have no variation.")
    else:
        summary_parts.append("\nNumeric Metrics Summary:")
        for col in relevant_numeric_cols:
            # Check if column has non-null values before calculating stats
            if not df[col].isnull().all():
                try:
                    mean_val = df[col].mean()
                    min_val = df[col].min()
                    max_val = df[col].max()
                    std_val = df[col].std()
                    # Simple recent trend (optional, compare last value to mean or previous)
                    # trend = ""
                    # if len(df[col].dropna()) > 1:
                    #     last_val = df[col].dropna().iloc[-1]
                    #     if last_val > mean_val + 0.5 * std_val: trend = " (Recent: High)"
                    #     elif last_val < mean_val - 0.5 * std_val: trend = " (Recent: Low)"

                    summary_parts.append(f"- {col.replace('_', ' ').title()}: Avg {mean_val:.2f}, Range ({min_val:.2f}-{max_val:.2f}), Std Dev {std_val:.2f}") # Removed trend for simplicity
                except Exception as stat_error:
                    summary_parts.append(f"- {col.replace('_', ' ').title()}: Error calculating stats ({stat_error})")
            else:
                summary_parts.append(f"- {col.replace('_', ' ').title()}: No data recorded.")

    # Add summary for categorical data if relevant (e.g., stress levels, symptoms)
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    relevant_categorical_cols = [col for col in categorical_cols if col not in ['user_id', 'name', 'timestamp'] and df[col].nunique() > 1] # Example filter

    if relevant_categorical_cols:
       summary_parts.append("\nCategorical/Event Summary:")
       for col in relevant_categorical_cols:
           if not df[col].isnull().all():
               # Show top few categories or presence/absence
               top_categories = df[col].value_counts().head(3).to_dict()
               summary_parts.append(f"- {col.replace('_', ' ').title()}: Common entries: {top_categories}")
               # You might want to summarize counts or frequency over time if relevant
           else:
               summary_parts.append(f"- {col.replace('_', ' ').title()}: No data recorded.")


    return "\n".join(summary_parts)


# --- Background Task ---
async def train_model_background(task_id: str, data: pd.DataFrame, user_id: str, name: str,task_status,data_preprocessor) -> None:
    """Background task to train the model."""
    log_context = {"event": "background_task", "task_type": "train_model", "task_id": task_id, "user_id": user_id}
    logger.info(f"Starting model training", extra={'extra_data': log_context})

    if task_id not in task_status:
        logger.error(f"Task ID {task_id} not found in status dict at start of background task.", extra={'extra_data': log_context})
        return # Should not happen if queued correctly

    task_status[task_id]["status"] = "processing"
    try:
        # Assuming DataProcessor exists and is initialized correctly
        if data_preprocessor:
            # This might raise errors too (e.g., during preprocessing or model fitting)
            model, score = data_preprocessor.train_model(data, user_id, name)
            task_status[task_id].update({
                "status": "completed",
                "score": score,
                "model_info": str(model), # Consider a more structured representation if possible
                "completed_at": datetime.now().isoformat(),
            })
            logger.info(f"Model training completed successfully. Score: {score}", extra={'extra_data': {**log_context, "score": score}})
        else:
            raise RuntimeError("DataProcessor is not initialized.") # Fail task if DP is missing

    except Exception as e:
        log_api_error("train_model_background", e, user_id, task_id) # Use helper
        task_status[task_id].update({
            "status": "failed",
            "error": str(e), # Include error message in status
            "completed_at": datetime.now().isoformat(),
        })
async def gemini_diagnosis_stream(
    
    system_prompt: str,
    user_id: str,

    model,
    model_name
) -> AsyncGenerator[str, None]: # <--- Now yields strings
    """
    Asynchronous generator that streams Gemini AI diagnosis content as SSE events (strings).
    """
    full_response_for_log = ""
    try:
        logger.info(
            f"Starting Gemini stream request for user {user_id} diagnosis",
            extra={'extra_data': {"user_id": user_id, "event": "gemini_diagnosis_stream_start"}}
        )

        

        # Send the disclaimer as the first event
        disclaimer_text = DiagnosisResponse.model_fields['disclaimer'].default
        # Manual SSE formatting: event line + data line + double newline
        yield f"event: disclaimer\ndata: {json.dumps({'type': 'disclaimer', 'content': disclaimer_text})}\n\n"

        # Stream Gemini Diagnosis Chunks
        chunk_count = 0
        response_stream = model.models.generate_content_stream(contents=system_prompt,model=model_name)

        for chunk in response_stream:
            print(chunk.text,end="")
            if hasattr(chunk, 'text'):
                yield f"event: diagnosis_chunk\ndata: {json.dumps({'type': 'text', 'content': chunk.text})}\n\n"
                chunk_count += 1
            elif chunk.prompt_feedback:
                logger.warning(
                    f"Gemini safety feedback received for user {user_id}: {chunk.prompt_feedback}",
                    extra={'extra_data': {"user_id": user_id, "event": "gemini_safety_feedback", "feedback": str(chunk.prompt_feedback)}}
                )
                if chunk.prompt_feedback.block_reason:
                     # Manual SSE formatting for error due to safety
                     yield f"event: error\ndata: {json.dumps({'type': 'error', 'content': f"AI response blocked due to safety concerns: {chunk.prompt_feedback.block_reason}"})}\n\n"
                     break # Stop streaming if content is blocked

          

    except Exception as stream_error:
        log_api_error("gemini_diagnosis_stream", stream_error, user_id)
        error_event_data = {"type": "error", "message": f"AI service error during diagnosis streaming: {str(stream_error)}"}
        # Manual SSE formatting for generic stream error
        yield f"event: error\ndata: {json.dumps(error_event_data)}\n\n"
    finally:
        logger.info(
            f"Gemini stream processing finished for user {user_id}. Sent {chunk_count} chunks.",
            extra={'extra_data': {"user_id": user_id, "event": "gemini_diagnosis_stream_complete", "chunk_count": chunk_count}}
        )
        logger.debug(
            f"Full Gemini diagnosis response (logged internally) for {user_id}: {full_response_for_log}",
            extra={'extra_data': {"user_id": user_id, "full_response": full_response_for_log}}
        )
        # Send completion event
        yield f"event: completion\ndata: {json.dumps({'type': 'completion', 'status': 'success', 'message': 'Diagnosis stream completed.'})}\n\n"


