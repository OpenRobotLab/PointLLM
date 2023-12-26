import openai
import time
import random
import os

def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 40,
    max_delay: int = 30,
    errors: tuple = (openai.error.RateLimitError, openai.error.ServiceUnavailableError, openai.error.Timeout),
):
    """Retry a function with exponential backoff."""
    def wrapper(*args, **kwargs):
        num_retries = 0
        delay = initial_delay

        while True:
            try:
                return func(*args, **kwargs)
            except errors as e:
                # * print the error info
                num_retries += 1
                if num_retries > max_retries:
                    print(f"[OPENAI] Encounter error: {e}.")
                    raise Exception(
                        f"[OPENAI] Maximum number of retries ({max_retries}) exceeded."
                    )
                delay *= exponential_base * (1 + jitter * random.random())
                time.sleep(min(delay, max_delay))
            except Exception as e:
                raise e
    return wrapper

class OpenAIGPT():
    def __init__(self, model="gpt-3.5-turbo-0613", temperature=1, top_p=1, max_tokens=2048, **kwargs) -> None:
        setup_openai(model)
        self.default_chat_parameters = {
            "model": model, 
            "temperature": temperature, 
            "top_p": top_p, 
            "max_tokens": max_tokens,
            **kwargs
        }

    @retry_with_exponential_backoff 
    def safe_chat_complete(self, messages, content_only=True, **kwargs):
        chat_parameters = self.default_chat_parameters.copy()
        if len(kwargs) > 0:
            chat_parameters.update(**kwargs)

        response = openai.ChatCompletion.create(
            messages=messages,
            **chat_parameters
        )

        if content_only:
            response = response['choices'][0]["message"]['content']

        return response

def setup_openai(model_name):
    # Setup OpenAI API Key
    print("[OPENAI] Setting OpenAI api_key...")
    openai.api_key = os.getenv('OPENAI_API_KEY')
    print(f"[OPENAI] OpenAI organization: {openai.organization}")
    print(f"[OPENAI] Using MODEL: {model_name}")