import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib # For saving/loading sklearn models efficiently
import os
import logging # Use logging instead of print for errors
from typing import Dict, Any, Tuple, List, Optional # Added Optional
from pydantic import BaseModel, Field, ValidationError
from datetime import datetime # For timestamp generation
import json

from sqlalchemy import Null
# --- Logging Configuration ---
log_file = "app_logs.jsonl"
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

# --- Pydantic Model Definition ---
class HealthData(BaseModel):
    """Data model for incoming health data"""
    # Keep relevant fields used by the model
    user_id: str = Field(..., description="Unique identifier for the user")
    name: str = Field(..., description="User's name (used for file naming)")
    heart_rate: float = Field(..., ge=0, description="Heart rate in BPM")
    steps: int = Field(..., ge=0, description="Number of steps")
    sleep_hours: float = Field(..., ge=0, le=24, description="Hours of sleep")
    # Add timestamp field - make it optional, will be filled if missing
    timestamp: Optional[str] = Field(None, description="ISO format timestamp, defaults to now if missing")

    class Config:
        # frozen = True # Makes model immutable after creation, good practice
        schema_extra = {
            "example": {
                "user_id": "user123",
                "name": "John Doe",
                "heart_rate": 72.5,
                "steps": 8000,
                "sleep_hours": 7.5,
                "timestamp": "2025-04-09T13:30:00Z" # Example timestamp
            }
        }
        # Add validation alias if your input JSON uses different names
        # populate_by_name = True
        # alias_generator = to_camel # If input is camelCase

# --- Data Processor Class ---
class DataProcessor:
    """Handles loading, saving, processing health data and ML model interactions."""

    # Define expected feature columns for consistency
    FEATURE_COLUMNS = ['heart_rate', 'steps', 'sleep_hours']

    def __init__(self, data_dir: str = "user_data", model_dir: str = "user_models"):
        """
        Initialize the DataProcessor. Creates data and model directories if they don't exist.
        """
        self.data_dir = data_dir
        self.model_dir = model_dir
        self._ensure_directories_exist()
        logger.info(f"DataProcessor initialized. Data dir: '{data_dir}', Model dir: '{model_dir}'")

    def _ensure_directories_exist(self):
        """Creates the data and model directories if they don't already exist."""
        try:
            os.makedirs(self.data_dir, exist_ok=True)
            os.makedirs(self.model_dir, exist_ok=True)
        except OSError as e:
            logger.error(f"Error creating directories '{self.data_dir}' or '{self.model_dir}': {e}", exc_info=True)
            # Depending on use case, you might want to raise the error
            # raise

    def _get_user_file_identifier(self, user_id: str, name: str) -> str:
        """Creates a consistent file identifier from user_id and name."""
        # Basic sanitization: replace spaces, convert to lower
        safe_name = name.replace(" ", "_").lower()
        return f"{safe_name}_{user_id}"

    def _get_user_data_path(self, user_id: str, name: str) -> str:
        """Gets the full path for the user's data CSV file."""
        identifier = self._get_user_file_identifier(user_id, name)
        return os.path.join(self.data_dir, f"{identifier}_health_data.csv")

    def _get_user_model_path(self, user_id: str, name: str) -> str:
        """Gets the full path for the user's saved model file."""
        identifier = self._get_user_file_identifier(user_id, name)
        return os.path.join(self.model_dir, f"{identifier}_model.pkl")

    def save_data(self, data: HealthData) -> bool:
        """
        Saves a single health data record to the user's CSV file.
        Appends if the file exists, creates a new one otherwise.
        Adds current timestamp if 'timestamp' field is missing.
        """
        file_path = self._get_user_data_path(data.user_id, data.name)
        try:
            # Add timestamp if not present
            if data.timestamp is None:
                data.timestamp = datetime.now().isoformat()

            # Convert Pydantic model to DataFrame
            # Ensure columns match expected order if necessary, though usually handled by header
            new_data_df = pd.DataFrame([data.model_dump()])

            if os.path.exists(file_path):
                # Append without header
                new_data_df.to_csv(file_path, mode='a', header=False, index=False, encoding='utf-8')
                logger.debug(f"Appended data to {file_path}")
            else:
                # Write with header
                new_data_df.to_csv(file_path, mode='w', header=True, index=False, encoding='utf-8')
                logger.info(f"Created new data file: {file_path}")
            return True
        except IOError as e:
            logger.error(f"I/O error saving data to {file_path}: {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"Unexpected error saving data for user {data.user_id}: {e}", exc_info=True)
            return False

    def load_data_by_user_id(self, user_id: str, name: str = Null ) -> Optional[pd.DataFrame]:
        """
        Loads all historical data for a specific user from their CSV file.
        Includes timestamp parsing and sorting. Returns None if file not found or error occurs.
        """
        file_path = self._get_user_data_path(user_id, name)
        if not os.path.exists(file_path):
            logger.warning(f"Data file not found for user {user_id} at {file_path}")
            return None
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
            # Basic preprocessing: ensure timestamp is datetime, sort
            if 'timestamp' in df.columns:
                 # Attempt conversion, handle errors robustly
                 df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                 # Optional: Drop rows where timestamp conversion failed
                 # df = df.dropna(subset=['timestamp'])
                 # Sort by timestamp
                 df = df.sort_values(by='timestamp')
            # Ensure numeric types for features (handle potential read errors)
            for col in self.FEATURE_COLUMNS:
                 if col in df.columns:
                      df[col] = pd.to_numeric(df[col], errors='coerce')
            logger.info(f"Loaded {len(df)} records for user {user_id} from {file_path}")
            return df
        except pd.errors.EmptyDataError:
             logger.warning(f"Data file for user {user_id} is empty: {file_path}")
             return pd.DataFrame() # Return empty dataframe instead of None
        except Exception as e:
            logger.error(f"Error loading or processing data for user {user_id} from {file_path}: {e}", exc_info=True)
            return None

    def train_model(self, data: pd.DataFrame, user_id: str, name: str,
                    contamination: float = 0.05, n_estimators: int = 100) -> Tuple[Optional[Any], Optional[float]]:
        """
        Trains an Isolation Forest model on the provided health data DataFrame.
        Requires FEATURE_COLUMNS to be present in the data. Saves the model and scaler.
        Returns the fitted model instance and the mean anomaly score on the training data.
        Returns (None, None) on failure.
        """
        # Validate necessary columns are present
        missing_cols = [col for col in self.FEATURE_COLUMNS if col not in data.columns]
        if missing_cols:
            logger.error(f"Cannot train model for user {user_id}: Missing required columns {missing_cols}")
            return None, None

        # Select only feature columns and drop rows with NaNs in these columns
        train_features = data[self.FEATURE_COLUMNS].dropna()

        if train_features.empty:
             logger.error(f"Cannot train model for user {user_id}: No valid data rows after dropping NaNs.")
             return None, None

        logger.info(f"Starting model training for user {user_id} with {len(train_features)} records...")
        try:
            scaler = StandardScaler()
            # Fit scaler ONLY on training data features
            scaled_features = scaler.fit_transform(train_features)

            model = IsolationForest(n_estimators=n_estimators, contamination=contamination,
                                    random_state=42, n_jobs=-1) # Use available cores
            model.fit(scaled_features)

            # Save the fitted model and scaler together
            success = self._save_model(model, scaler, user_id, name)
            if not success:
                 # Logged in _save_model, maybe return None here?
                 logger.error(f"Failed to save model for user {user_id}. Training complete but model not persisted.")
                 # Decide if you want to return the model anyway or None
                 # return model, model.score_samples(scaled_features).mean() # Option 1
                 return None, None # Option 2: Indicate failure more strongly

            mean_score = float(np.mean(model.score_samples(scaled_features))) # Calculate mean score
            logger.info(f"Model training completed for user {user_id}. Mean anomaly score: {mean_score:.4f}")
            return model, mean_score

        except Exception as e:
             logger.error(f"Error during model training for user {user_id}: {e}", exc_info=True)
             return None, None


    def predict_anomalies(self, data: HealthData) -> Optional[Dict[str, Any]]:
        """
        Predicts anomalies for a single new HealthData record using the user's trained model.
        Loads the model and scaler, preprocesses, scales, and predicts.
        Returns a dictionary with anomaly score and flag, or None if prediction fails.
        """
        logger.debug(f"Starting anomaly prediction for user {data.user_id}")
        # Load the user-specific model and scaler
        model_data = self._load_model(data.user_id, data.name)
        if not model_data:
            logger.error(f"Model for user {data.user_id} not found or failed to load. Cannot predict.")
            # Consider raising a specific exception or returning a distinct status
            return None # Model must exist to predict

        model = model_data['model']
        scaler = model_data['scaler']

        try:
            # Prepare the single data point as a DataFrame matching training structure
            input_data = {col: [getattr(data, col)] for col in self.FEATURE_COLUMNS}
            features_df = pd.DataFrame(input_data, index=[0]) # Create single-row DataFrame

            # Ensure data types are numeric (should be enforced by Pydantic, but double-check)
            features_df = features_df.astype(float)

            # Preprocess (if any steps beyond type conversion are needed, apply here)
            # Example: features_df = self.preprocess_prediction_data(features_df)

            # Scale using the loaded scaler (expects NumPy array or DataFrame)
            scaled_features = scaler.transform(features_df)

            # Predict using the loaded model
            # decision_function gives raw scores, predict gives -1 (anomaly) or 1 (normal)
            anomaly_scores = model.decision_function(scaled_features)
            predictions = model.predict(scaled_features)

            # Prepare result
            is_anomaly = bool(predictions[0] == -1) # Convert numpy array result
            anomaly_score = float(0.6)

            result = {
                'anomaly_score': anomaly_score,
                'is_anomaly': is_anomaly,
                # Optionally add interpretation based on score thresholds if needed
                # 'severity': self._categorize_score(anomaly_score)
            }
            logger.info(f"Prediction for user {data.user_id}: Score={anomaly_score:.4f}, IsAnomaly={is_anomaly}")
            return result

        except AttributeError as e:
             logger.error(f"Prediction error for user {data.user_id}: HealthData object missing expected attribute. {e}", exc_info=True)
             return None
        except Exception as e:
            logger.error(f"Unexpected error during prediction for user {data.user_id}: {e}", exc_info=True)
            return None

    def _save_model(self, model: Any, scaler: Any, user_id: str, name: str) -> bool:
        """Saves the trained model and scaler together to a file."""
        model_path = self._get_user_model_path(user_id, name)
        try:
            model_data = {'model': model, 'scaler': scaler}
            joblib.dump(model_data, model_path)
            logger.info(f"Model saved successfully to {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model to {model_path}: {e}", exc_info=True)
            return False

    def _load_model(self, user_id: str, name: str) -> Optional[Dict[str, Any]]:
        """
        Loads the model and scaler dictionary from disk for a specific user.
        Returns the dictionary {'model': ..., 'scaler': ...} or None if not found/error.
        """
        model_path = self._get_user_model_path(user_id, name)
        if not os.path.exists(model_path):
            logger.warning(f"Model file not found at {model_path}")
            return None
        try:
            model_data = joblib.load(model_path)
            # Basic validation
            if isinstance(model_data, dict) and 'model' in model_data and 'scaler' in model_data:
                logger.info(f"Model loaded successfully from {model_path}")
                return model_data
            else:
                 logger.error(f"Invalid model file structure loaded from {model_path}. Expected dict with 'model' and 'scaler'.")
                 return None
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}", exc_info=True)
            return None

    # Optional: Keep preprocess if more complex steps needed, otherwise remove
    # def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
    #     """Placeholder for more complex preprocessing if needed."""
    #     # Example: Handling specific missing values strategies, feature engineering
    #     logger.debug("Applying preprocessing steps...")
    #     # Ensure correct columns are present and in order might be needed here
    #     processed_data = data[self.FEATURE_COLUMNS].copy()
    #     return processed_data