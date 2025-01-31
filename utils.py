import os
import openai
import google.generativeai as genai
import time
from functools import wraps

def retry_with_backoff(retries=3, backoff_factor=2):
    """Decorator for retrying a function with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retry_count = 0
            while retry_count < retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retry_count += 1
                    if retry_count == retries:
                        raise e
                    wait_time = backoff_factor ** retry_count
                    time.sleep(wait_time)
            return func(*args, **kwargs)
        return wrapper
    return decorator 

genai.configure(api_key=os.environ["GEMINI_API_KEY"])


@retry_with_backoff(retries=10)
def get_gpt_response(messages, model, temperature, n=1, max_tokens=500, stop=None):
    client = openai.OpenAI()
    return client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        n=n,
        max_tokens=max_tokens,
        stop=stop,
    )


GEMINI_SAFETY_SETTINGS_BLOCK_NONE = {
    genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
    genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
    genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
    genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
}


@retry_with_backoff(retries=10)
def get_gemini_response(messages, model, temperature, n=1, max_tokens=500, stop=None):
    # Extract the system instruction and messages history
    system_instruction = None
    history = []

    for msg in messages:
        if msg["role"] == "system":
            system_instruction = msg["content"]
        elif msg["role"] == "user":
            history.append({"role": "user", "parts": [msg["content"]]})
        elif msg["role"] == "assistant":
            history.append({"role": "model", "parts": [msg["content"]]})

    # If system instruction is not provided, set it to an empty string
    if system_instruction is None:
        system_instruction = ""

    # Get the latest user message
    latest_user_message = history[-1]["parts"][0] if history else ""
    history = history[:-1]  # Remove the latest user message from the history

    # Create a generation config
    generation_config = {
        "temperature": temperature,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": max_tokens,
        "response_mime_type": "text/plain",
        "candidate_count": 1,  # Set the number of possible responses
        "stop_sequences": stop,  # Comma-separated string or list that can be integrated if the API supports it
    }

    # Initialize the model with the provided and extracted configuration
    gem_model = genai.GenerativeModel(
        model_name=model,  # Use the model provided in the argument
        generation_config=generation_config,
        system_instruction=system_instruction,
        safety_settings=GEMINI_SAFETY_SETTINGS_BLOCK_NONE,
    )

    # Start a chat session with the extracted history
    chat_session = gem_model.start_chat(history=history)

    # Send the user's latest message and receive a response
    response = chat_session.send_message(latest_user_message)

    return response.candidates[0].content.parts[0].text
