import httpx
import json
import logging
from typing import Dict, List, Optional
import
# Set up logging for this module
logger = logging.getLogger(__name__)

# --- Configuration for Single Test Token ---
# IMPORTANT: Replace this with an actual Expo Push Token from your testing device/simulator.
# You'll get this token from your Expo app running on a device after calling
# Notifications.getExpoPushTokenAsync().
load_dotenv()
EXPO_SINGLE_TEST_TOKEN = os.getenv("EXPO_SINGLE_TEST_TOKEN")

# You can also use an environment variable for this:
# import os
# EXPO_SINGLE_TEST_TOKEN = os.getenv("EXPO_TEST_TOKEN", "ExpoPushToken[default_if_not_set]")


# --- Main Notification Sending Function (Modified) ---
async def send_expo_notification_to_user(
    user_id: str, # user_id will still be passed but ignored for token retrieval
    title: str,
    body: str,
    data: Optional[Dict] = None,
    sound: Optional[str] = "default",
    # Add other Expo notification fields if needed
) -> Dict:
    """
    Sends an Expo push notification to a SINGLE, hardcoded test device.
    FOR TESTING PURPOSES ONLY.

    Args:
        user_id (str): The ID of the user (ignored for token retrieval in this test version).
        title (str): The title of the notification.
        body (str): The main body text of the notification.
        data (Optional[Dict]): Custom data to be sent with the notification (accessible in app).
        sound (Optional[str]): The sound to play (e.g., "default", or "custom.wav").

    Returns:
        Dict: A dictionary containing the result from the Expo Push API.
    """
    logger.warning("Using a SINGLE, HARDCODED EXPO PUSH TOKEN for notification. This is for testing only and must be replaced in production!")

    # Use the single test token directly
    tokens = [EXPO_SINGLE_TEST_TOKEN]

    if not tokens or not EXPO_SINGLE_TEST_TOKEN:
        logger.error("EXPO_SINGLE_TEST_TOKEN is not set. Cannot send notification.")
        return {"status": "skipped", "reason": "test_token_not_set"}

    # 1. Prepare message for Expo API
    expo_messages = []
    # No need for a loop here as we only have one token, but keeping it as a list for Expo API consistency.
    expo_messages.append({
        "to": EXPO_SINGLE_TEST_TOKEN,
        "title": title,
        "body": body,
        "data": data,
        "sound": sound,
    })

    # 2. Send message to Expo Push API
    EXPO_PUSH_API_URL = "https://exp.host/--/api/v2/push/send"
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                EXPO_PUSH_API_URL,
                json=expo_messages, # Send the list containing our single message
                headers={
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip, deflate",
                    "Content-Type": "application/json",
                    # If you enabled "Enhanced Security for Push Notifications" in Expo,
                    # provide your Expo Access Token here. Store it securely (e.g., env var).
                 "Authorization": "Bearer cuNG1O3pLLaKIxuZlnvK6dZpyKCSpnMiGtm5mm7P",
                }
            )
            response.raise_for_status() # Raises an exception for 4xx/5xx responses
            
            expo_response_data = response.json()
            logger.info(f"Expo API raw response for test token: {expo_response_data}")

            # Process the Expo API response (tickets)
            # This is simplified as we expect only one ticket for our single token.
            for ticket in expo_response_data.get("data", []):
                if ticket.get("status") == "error":
                    error_details = ticket.get("details", {})
                    error_message = ticket.get("message", "Unknown error")
                    
                    if error_details.get("error") == "DeviceNotRegistered":
                        logger.error(
                            f"Expo Push: Test token '{ticket.get('to')}' is no longer registered. "
                            "You might need to get a new token from your device."
                        )
                    else:
                        logger.error(
                            f"Expo Push: Failed to send notification to test token '{ticket.get('to')}'. "
                            f"Error: {error_message} (Details: {error_details})"
                        )
                elif ticket.get("status") == "ok":
                    logger.debug(f"Expo Push: Notification sent successfully to test token '{ticket.get('id')}'.")
            
            return {"status": "success", "details": expo_response_data}

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error sending Expo notification to test token: {e.response.status_code} - {e.response.text}")
            return {"status": "error", "reason": "http_error", "detail": e.response.text}
        except httpx.RequestError as e:
            logger.error(f"Network error sending Expo notification to test token: {e}")
            return {"status": "error", "reason": "network_error", "detail": str(e)}
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error from Expo API for test token: {e}")
            return {"status": "error", "reason": "json_decode_error", "detail": str(e)}
        except Exception as e:
            logger.error(f"An unexpected error occurred while sending Expo notification to test token: {e}")
            return {"status": "error", "reason": "unexpected_error", "detail": str(e)}