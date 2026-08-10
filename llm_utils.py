import os
import time
from dotenv import load_dotenv
from google import genai

load_dotenv()


def call_llm(prompt, temperature=0.2, max_tokens=500):
    """
    Sends a prompt to Gemini and retries up to 3 times
    if the API call fails.

    Returns:
        str: Model response if successful.
        None: If all 3 attempts fail.
    """

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found in environment variables."
        )

    client = genai.Client(api_key=api_key)

    for attempt in range(1, 4):
        try:
            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt,
                config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                    "response_mime_type": "application/json"
                }
            )

            return response.text

        except Exception as e:
            print(f"Attempt {attempt}/3 failed: {e}")

            if attempt < 3:
                wait_time = 30 * attempt
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)

            else:
                print("All 3 attempts failed.")
                print("Moving on to the next request.")

    return None