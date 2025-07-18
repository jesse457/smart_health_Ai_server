# 🩺 Health Data API

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Google Gemini](https://img.shields.io/badge/Google%20Gemini-AI-orange?style=for-the-badge&logo=google&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-2.7.1-E92063?style=for-the-badge&logo=pydantic&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

---

## 📖 Table of Contents

*   [🌟 Introduction](#-introduction)
*   [✨ Features](#-features)
*   [🚀 Getting Started](#-getting-started)
    *   [Prerequisites](#prerequisites)
    *   [Installation](#installation)
    *   [Environment Variables](#environment-variables)
    *   [Running the Application](#running-the-application)
*   [💡 API Endpoints](#-api-endpoints)
    *   [Health Check](#health-check)
    *   [Train Model](#train-model)
    *   [Get Task Status](#get-task-status)
    *   [Predict Anomalies](#predict-anomalies)
    *   [Get Health Recommendations](#get-health-recommendations)
    *   [Get Data Explanation for Caregivers](#get-data-explanation-for-caregivers)
    *   [WebSocket Diagnosis Stream](#websocket-diagnosis-stream)
*   [📂 Project Structure](#-project-structure)
*   [⚙️ Configuration & Logging](#-configuration--logging)
*   [⚠️ Important Notes & Disclaimers](#️-important-notes--disclaimers)
*   [🤝 Contributing](#-contributing)
*   [📄 License](#-license)
*   [📞 Contact](#-contact)

---

## 🌟 Introduction

The **Health Data API** is a robust and intelligent backend service designed to process, analyze, and provide insights into patient health data. Leveraging FastAPI for high performance and Google's Gemini AI for advanced natural language understanding, this API offers functionalities ranging from anomaly detection and personalized health recommendations to AI-driven data explanations for caregivers and real-time diagnostic assistance.

This project aims to empower healthcare applications with smart data processing and AI capabilities, making health data more actionable and understandable.

## ✨ Features

*   **Health Data Ingestion:** Securely receive and store patient health data.
*   **Background Model Training:** Asynchronously train anomaly detection models for individual users when sufficient data is available.
*   **Anomaly Prediction:** Identify unusual patterns in incoming health data and trigger notifications (e.g., via Expo).
*   **AI-Powered Health Recommendations:** Generate personalized, actionable health recommendations using Google Gemini based on historical data.
*   **Caregiver Data Explanation:** Provide easy-to-understand summaries and explanations of patient health trends for non-technical caregivers.
*   **Real-time AI Diagnosis (WebSocket):** Stream AI-generated diagnostic insights based on health data and reported symptoms.
*   **Asynchronous Task Management:** Track the status of long-running background tasks.
*   **Structured Logging:** Custom JSON logging for better observability and error tracking.
*   **Scalable Architecture:** Built with FastAPI for high concurrency and performance.

## 🚀 Getting Started

Follow these steps to set up and run the Health Data API on your local machine.

### Prerequisites

*   **Python 3.9+**
*   **`pip`** (Python package installer)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/jesse457/smart_health_Ai_server.git
    cd health-data-api
    ```
  

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    # On Windows
    .venv\Scripts\activate
    # On macOS/Linux
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    Create a `requirements.txt` file in the root directory of your project with the following content:
    ```
    fastapi[standard]
    pandas
    python-dotenv
    google-genai
    pydantic
    scikit-learn
    regex
    # Add any other specific libraries your helpers use, e.g., sqlalchemy if fully utilized
    ```
    Then install:
    ```bash
    pip install -r requirements.txt
    ```

### Environment Variables

Create a `.env` file in the root directory of your project and add your Google Gemini API key:

```dotenv
GEMINI_API_KEY="YOUR_GOOGLE_GEMINI_API_KEY"
EXPO_SINGLE_TEST_TOKEN="YOUR_EXPO_TOKEN_FROM_DEVICE"
```

You can obtain a Gemini API key from the [Google AI Studio](https://ai.google.dev/gemini-api/docs/api-key).

### Running the Application

1.  **Start the FastAPI application using Uvicorn:**
    ```bash
    fastapi dev --host 0.0.0.0 --port 8000
    ```
    *   `dev`: Enables development mode in the server.
    *   `--host 0.0.0.0`: Makes the server accessible from other devices on your network.
    *   `--port 8000`: Runs the server on port 8000.

2.  **Access the API documentation:**
    Once the server is running, you can access the interactive API documentation (Swagger UI) at:
    `http://localhost:8000/docs`

## 💡 API Endpoints

Here's a summary of the available API endpoints:

### Health Check

*   **GET `/health`**
*   **Description:** Basic health check to verify the API's status and dependency availability (DataProcessor, Gemini API).
*   **Response:** `{"status": "ok", "timestamp": "...", "version": "...", "data_processor": "...", "gemini_service": "..."}`

### Train Model

*   **POST `/train_model`**
*   **Description:** Submits new health data for a user and triggers a background task to train an anomaly detection model if enough data points are accumulated.
*   **Request Body:** `HealthData` (e.g., `{"user_id": "user123", "name": "John Doe", "timestamp": "...", "blood_pressure": "...", "heart_rate": "...", ...}`).
*   **Response:** `TrainResponse` (e.g., `{"message": "Model training initiated.", "task_id": "...", "status": {"task_id": "...", "status": "queued", ...}}`)
*   **Status Code:** `202 Accepted` (for background task initiation)

### Get Task Status

*   **GET `/task/{task_id}`**
*   **Description:** Retrieves the current status of a background model training task.
*   **Path Parameters:** `task_id` (UUID of the task).
*   **Response:** `TaskStatusResponse` (e.g., `{"task_id": "...", "user_id": "...", "status": "completed", "score": 0.95, ...}`)

### Predict Anomalies

*   **POST `/predict`**
*   **Description:** Predicts anomalies for a user's latest health data submission using their trained model. An Expo push notification is sent if an anomaly is detected.
*   **Request Body:** `HealthData` (same as `/train_model`).
*   **Response:** `PredictionResponse` (e.g., `{"predicted_value": {"anomaly_score": 0.92}, "is_anomaly": true}`)

### Get Health Recommendations

*   **GET `/recommendations/{name}/{user_id}`**
*   **Description:** Generates personalized health recommendations using Google Gemini based on the user's historical health data.
*   **Path Parameters:** `name`, `user_id`.
*   **Response:** `RecommendationResponse` (e.g., `{"user_id": "...", "recommendations": [{"title": "...", "description": "...", "category": "..."}, ...], "data_summary_used": "..."}`)

### Get Data Explanation for Caregivers

*   **GET `/data/explanation/{name}/{user_id}`**
*   **Description:** Provides an easy-to-understand explanation and summary of a patient's historical health data, tailored for caregivers, using Google Gemini.
*   **Path Parameters:** `name`, `user_id`.
*   **Response:** `DataExplanationResponse` (e.g., `{"user_id": "...", "name": "...", "data_summary_used": "...", "explanation": "..."}`)

### WebSocket Diagnosis Stream

*   **WebSocket `/ws/diagnosis/{user_id}/{name}`**
*   **Description:** Establishes a real-time WebSocket connection to stream AI-generated diagnostic insights based on historical data and optional symptoms.
*   **Path Parameters:** `user_id`, `name`.
*   **Query Parameters (Optional):** `symptoms` (e.g., `ws://localhost:8000/ws/diagnosis/user123/John%20Doe?symptoms=fever,cough`).
*   **Messages:**
    *   `{"type": "disclaimer", "content": "..."}`: Initial disclaimer.
    *   `{"type": "text", "content": "..."}`: Streamed chunks of the AI's diagnosis.
    *   `{"type": "error", "message": "..."}`: Error messages.
    *   `{"type": "completion", "status": "success", "message": "..."}`: Indicates stream completion.

## 📂 Project Structure

```
.
├── .env                  # Environment variables (e.g., GEMINI_API_KEY)
├── .gitignore            # Files/directories to ignore in Git
├── main.py               # Main FastAPI application
├── data_processor.py     # Handles data loading, saving, and model interactions
├── helpers/              # Helper functions and Pydantic models
│   ├── __init__.py
│   ├── functions.py      # Utility functions (logging, data summarization, background tasks)
│   ├── pydanic_models.py # Pydantic models for request/response validation
│   └── notifications.py  # Logic for sending push notifications (e.g., Expo)
├── app_logs.jsonl        # JSON formatted application logs
├── user_data/            # Directory for storing raw patient health data (e.g., CSV, JSON per user)
├── user_models/          # Directory for storing trained anomaly detection models per user
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## ⚙️ Configuration & Logging

*   **Gemini API:** Configured via `GEMINI_API_KEY` environment variable. If not set, AI-dependent features (recommendations, diagnosis, explanation) will be disabled.
*   **Logging:** The application uses a custom JSON formatter for logging to `app_logs.jsonl` for structured, machine-readable logs. Console logging is also enabled for real-time monitoring during development.
*   **CORS:** Currently configured to allow all origins (`*`) for development purposes. **For production deployments, it is highly recommended to restrict `allow_origins` to your specific frontend domains.**

## ⚠️ Important Notes & Disclaimers

*   **AI Disclaimer:** The AI-generated recommendations, explanations, and diagnostic insights are for informational purposes only and **should not be considered medical advice, diagnosis, or treatment.** Always consult a qualified healthcare professional for any health concerns.
*   **Data Storage:** The current implementation uses local file storage (`user_data/`, `user_models/`) and an in-memory dictionary (`task_status`) for simplicity. **For production environments, it is strongly recommended to use a robust database (SQL/NoSQL) for data persistence and a dedicated task queue system (e.g., Celery with Redis/RabbitMQ) for background task management.**
*   **Security:** Ensure proper authentication and authorization mechanisms are implemented for production use cases, as this example focuses primarily on the core API logic.
*   **Minimum Data Points:** The API includes checks for minimum data points before initiating model training or providing AI insights to ensure meaningful results.

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improvements, new features, or bug fixes, please open an issue or submit a pull request.

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## 📞 Contact

For any questions or inquiries, please reach out to:

*   **Email:** [okiwajesse1@gmail.com]
*   **GitHub:** [https://github.com/jesse457]
```