import ast
import os
from time import sleep

from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

# Initialize the OpenAI client with API key from environment
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Maximum number of tokens that the openai api allows me to request per minute
RATE_LIMIT = 250000


# To avoid rate limits, we use exponential backoff where we wait longer and longer
# between requests whenever we hit a rate limit. Explanation can be found here:
# https://github.com/openai/openai-cookbook/blob/main/examples/How_to_handle_rate_limits.ipynb
# I'm using default parameters here, I don't know if something else might be
# better.
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
def completion_with_backoff(**kwargs):
    # Convert legacy parameters to new API format
    prompt = kwargs.get("prompt", "")
    
    # Handle string content directly - don't wrap in array
    messages = [{"role": "user", "content": prompt}]
    
    # Map common parameters
    new_kwargs = {
        "model": kwargs.get("model", "gpt-3.5-turbo"),
        "messages": messages,
        "max_tokens": kwargs.get("max_tokens"),
        "temperature": kwargs.get("temperature"),
        "top_p": kwargs.get("top_p"),
        "n": kwargs.get("n"),
        "stop": kwargs.get("stop"),
        "presence_penalty": kwargs.get("presence_penalty"),
        "frequency_penalty": kwargs.get("frequency_penalty"),
        "logit_bias": kwargs.get("logit_bias"),
    }
    
    # Remove None values
    new_kwargs = {k: v for k, v in new_kwargs.items() if v is not None}
    
    response = client.chat.completions.create(**new_kwargs)
    
    # Format response to match legacy API structure
    return {
        "choices": [{
            "text": choice.message.content,
            "index": i,
            "logprobs": None,
            "finish_reason": choice.finish_reason
        } for i, choice in enumerate(response.choices)],
        "model": response.model,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        } if response.usage else None
    }


# Define a function that adds a delay to a Completion API call
def delayed_completion_with_backoff(delay_in_seconds: float = 1, **kwargs):
    """Delay a completion by a specified amount of time."""

    # Sleep for the delay
    sleep(delay_in_seconds)

    # Call the Completion API and return the result
    return completion_with_backoff(**kwargs)


def completion_create_retry(*args, sleep_time=5, **kwargs):
    """A wrapper around OpenAI chat completions that retries the request if it fails for any reason."""

    if 'llama' in kwargs['model'] or 'vicuna' in kwargs['model'] or 'alpaca' in kwargs['model']:
        if type(kwargs['prompt'][0]) == list:
            prompts = [prompt[0] for prompt in kwargs['prompt']]
        else:
            prompts = kwargs['prompt']
        return kwargs['endpoint'](prompts, **kwargs)
    else:
        while True:
            try:
                # Convert to new API format
                prompt = kwargs.get("prompt", "")
                
                # Handle batch prompts
                if isinstance(prompt, list):
                    # Process each prompt separately and collect responses
                    all_choices = []
                    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                    
                    for i, p in enumerate(prompt):
                        # Ensure content is a string, not wrapped in array
                        messages = [{"role": "user", "content": str(p)}]
                        
                        new_kwargs = {
                            "model": kwargs.get("model", "gpt-3.5-turbo"),
                            "messages": messages,
                            "max_tokens": kwargs.get("max_tokens"),
                            "temperature": kwargs.get("temperature"),
                            "top_p": kwargs.get("top_p"),
                            "stop": kwargs.get("stop"),
                            "presence_penalty": kwargs.get("presence_penalty"),
                            "frequency_penalty": kwargs.get("frequency_penalty"),
                            "logit_bias": kwargs.get("logit_bias"),
                        }
                        
                        # Remove None values
                        new_kwargs = {k: v for k, v in new_kwargs.items() if v is not None}
                        
                        response = client.chat.completions.create(**new_kwargs)
                        
                        # Add choices with correct indexing
                        for choice in response.choices:
                            all_choices.append({
                                "text": choice.message.content,
                                "index": i,
                                "logprobs": None,
                                "finish_reason": choice.finish_reason
                            })
                        
                        if response.usage:
                            total_usage["prompt_tokens"] += response.usage.prompt_tokens
                            total_usage["completion_tokens"] += response.usage.completion_tokens
                            total_usage["total_tokens"] += response.usage.total_tokens
                    
                    return {
                        "choices": all_choices,
                        "model": kwargs.get("model", "gpt-3.5-turbo"),
                        "usage": total_usage
                    }
                else:
                    # Single prompt - ensure content is a string
                    messages = [{"role": "user", "content": str(prompt)}]
                    
                    new_kwargs = {
                        "model": kwargs.get("model", "gpt-3.5-turbo"),
                        "messages": messages,
                        "max_tokens": kwargs.get("max_tokens"),
                        "temperature": kwargs.get("temperature"),
                        "top_p": kwargs.get("top_p"),
                        "n": kwargs.get("n"),
                        "stop": kwargs.get("stop"),
                        "presence_penalty": kwargs.get("presence_penalty"),
                        "frequency_penalty": kwargs.get("frequency_penalty"),
                        "logit_bias": kwargs.get("logit_bias"),
                    }
                    
                    # Remove None values
                    new_kwargs = {k: v for k, v in new_kwargs.items() if v is not None}
                    
                    response = client.chat.completions.create(**new_kwargs)
                    
                    # Format response to match legacy API structure
                    return {
                        "choices": [{
                            "text": choice.message.content,
                            "index": i,
                            "logprobs": None,
                            "finish_reason": choice.finish_reason
                        } for i, choice in enumerate(response.choices)],
                        "model": response.model,
                        "usage": {
                            "prompt_tokens": response.usage.prompt_tokens,
                            "completion_tokens": response.usage.completion_tokens,
                            "total_tokens": response.usage.total_tokens
                        } if response.usage else None
                    }
            except Exception as e:
                print(f"Error in completion_create_retry: {e}")
                sleep(sleep_time)
