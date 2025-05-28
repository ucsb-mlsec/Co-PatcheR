import time
import json
from typing import Dict, Union

import anthropic
import openai
import tiktoken


def num_tokens_from_messages(message, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if isinstance(message, list):
        # use last message.
        num_tokens = len(encoding.encode(message[0]["content"]))
    else:
        num_tokens = len(encoding.encode(message))
    return num_tokens


def create_chatgpt_config(
    message: Union[str, list],
    max_tokens: int,
    temperature: float = 1,
    batch_size: int = 1,
    system_message: str = "You are a helpful assistant.",
    model: str = "gpt-3.5-turbo",
    **kwargs,
) -> Dict:
    if 'o1' in model or 'o3' in model or "deepseek" in model:
        # o1 doesn't support system messages
        if isinstance(message, list):
            config = {
                "model": model,
                "max_completion_tokens": max_tokens,
                "messages": message,
            }
        else:
            config = {
                "model": model,
                "max_completion_tokens": max_tokens,
                "messages": [
                    {"role": "user", "content": message},
                ],
            }
    else:
        if isinstance(message, list):
            config = {
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "n": batch_size,
                "messages": [{"role": "system", "content": system_message}] + message,
            }
        else:
            config = {
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "n": batch_size,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": message},
                ],
            }
    config.update(kwargs)
    return config


def handler(signum, frame):
    # swallow signum and frame
    raise Exception("end of time")


def request_chatgpt_engine(config, logger, base_url=None, max_retries=40, timeout=100, api_key=None):
    ret = None
    retries = 0

    if api_key:
        client = openai.OpenAI(base_url=base_url, api_key=api_key)
    else:
        client = openai.OpenAI(base_url=base_url)
    
    while ret is None and retries < max_retries:
        try:
            # Attempt to get the completion
            logger.info("Creating API request")
            ret = client.chat.completions.create(**config)

        except openai.OpenAIError as e:
            if isinstance(e, openai.BadRequestError):
                logger.info("Request invalid")
                print(e)
                logger.info(e)
                return None
            elif isinstance(e, openai.RateLimitError):
                print(config)
                print("Rate limit exceeded. Waiting...")
                logger.info("Rate limit exceeded. Waiting...")
                print(e)
                logger.info(e)
                time.sleep(5)
            elif isinstance(e, openai.APIConnectionError):
                print("API connection error. Waiting...")
                logger.info("API connection error. Waiting...")
                print(e)
                logger.info(e)
                time.sleep(5)
            else:
                print("Unknown error. Waiting...")
                logger.info("Unknown error. Waiting...")
                print(e)
                logger.info(e)
                time.sleep(1)

        except json.decoder.JSONDecodeError as jsone:
            print("JSON parsing in response error. Waiting...")
            logger.info("JSON parsing in response error. Waiting...")
            print(jsone)
            logger.info(jsone)
            time.sleep(5)

        retries += 1

    logger.info(f"API response {ret}")
    return ret


def request_chatgpt_response_engine(config, logger, base_url=None, max_retries=40, timeout=100, api_key=None):
    ret = None
    retries = 0
    input = config["messages"][1]["content"]
    model = config["model"]

    if api_key:
        client = openai.OpenAI(base_url=base_url, api_key=api_key)
    else:
        client = openai.OpenAI(base_url=base_url)

    while ret is None and retries < max_retries:
        try:
            # Attempt to get the completion
            logger.info("Creating API request")
            ret = client.responses.create(
                model=model,
                input = input,
                reasoning={
                    "effort": "medium"
                }
            )

        except openai.OpenAIError as e:
            if isinstance(e, openai.BadRequestError):
                logger.info("Request invalid")
                print(e)
                logger.info(e)
                raise e
            elif isinstance(e, openai.RateLimitError):
                print(config)
                print("Rate limit exceeded. Waiting...")
                logger.info("Rate limit exceeded. Waiting...")
                print(e)
                logger.info(e)
                time.sleep(5)
            elif isinstance(e, openai.APIConnectionError):
                print("API connection error. Waiting...")
                logger.info("API connection error. Waiting...")
                print(e)
                logger.info(e)
                time.sleep(5)
            else:
                print("Unknown error. Waiting...")
                logger.info("Unknown error. Waiting...")
                print(e)
                logger.info(e)
                time.sleep(1)

        except json.decoder.JSONDecodeError as jsone:
            print("JSON parsing in response error. Waiting...")
            logger.info("JSON parsing in response error. Waiting...")
            print(jsone)
            logger.info(jsone)
            time.sleep(5)

        retries += 1

    logger.info(f"API response {ret}")
    return ret


def request_chatgpt_prefill_engine(config, logger, base_url=None, max_retries=40, timeout=100, api_key=None):
    ret = None
    retries = 0

    model = config["model"]
    temperature = config["temperature"]
    max_tokens = config["max_tokens"]
    prompt = config["prompt"]

    if api_key:
        client = openai.OpenAI(base_url=base_url, api_key=api_key)
    else:
        client = openai.OpenAI(base_url=base_url)

    while ret is None and retries < max_retries:
        try:
            # Attempt to get the completion
            logger.info("Creating API request")
            ret = client.completions.create(
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )

        except openai.OpenAIError as e:
            if isinstance(e, openai.BadRequestError):
                logger.info("Request invalid")
                print(e)
                logger.info(e)
                raise e
            elif isinstance(e, openai.RateLimitError):
                print(config)
                print("Rate limit exceeded. Waiting...")
                logger.info("Rate limit exceeded. Waiting...")
                print(e)
                logger.info(e)
                time.sleep(5)
            elif isinstance(e, openai.APIConnectionError):
                print("API connection error. Waiting...")
                logger.info("API connection error. Waiting...")
                print(e)
                logger.info(e)
                time.sleep(5)
            else:
                print("Unknown error. Waiting...")
                logger.info("Unknown error. Waiting...")
                print(e)
                logger.info(e)
                time.sleep(1)

        except json.decoder.JSONDecodeError as jsone:
            print("JSON parsing in response error. Waiting...")
            logger.info("JSON parsing in response error. Waiting...")
            print(jsone)
            logger.info(jsone)
            time.sleep(5)

        retries += 1

    logger.info(f"API response {ret}")
    return ret


def create_anthropic_config(
    message: str,
    max_tokens: int,
    temperature: float = 1,
    batch_size: int = 1,
    system_message: str = "You are a helpful assistant.",
    model: str = "claude-3-5-sonnet-20241022",
    **kwargs,
) -> Dict:
    config = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "system": system_message,
    }
    
    if isinstance(message, list):
        messages = message
    else:
        messages = [{
            "role": "user",
            "content": [{
                "type": "text",
                "text": message
            }]
        }]
        
    if batch_size > 1:
        if isinstance(messages[0]["content"], list):
            messages[0]["content"][0]["cache_control"] = {"type": "ephemeral"}
        else:
            messages[0]["content"] = [{
                "type": "text",
                "text": messages[0]["content"],
                "cache_control": {"type": "ephemeral"}
            }]
    
    config["messages"] = messages
    return config


def request_anthropic_engine(config, logger, base_url=None, max_retries=40, timeout=100):
    ret = None
    retries = 0

    client = anthropic.Anthropic(base_url=None)

    while ret is None and retries < max_retries:
        try:
            logger.info("Creating API request")

            ret = client.messages.create(**config)

        except anthropic.AnthropicError as e:
            print(e)
            if isinstance(e, anthropic.BadRequestError):
                logger.info("Request invalid")
                print(e)
                logger.info(e)
                #raise e
            elif isinstance(e, anthropic.RateLimitError):
                print(config)
                print("Rate limit exceeded. Waiting...")
                logger.info("Rate limit exceeded. Waiting...")
                print(e)
                logger.info(e)
                time.sleep(5)
            elif isinstance(e, anthropic.APIConnectionError):
                print("API connection error. Waiting...")
                logger.info("API connection error. Waiting...")
                print(e)
                logger.info(e)
                time.sleep(5)
            else:
                print("Unknown error. Waiting...")
                logger.info("Unknown error. Waiting...")
                print(e)
                logger.info(e)
                time.sleep(1)

        retries += 1

        logger.info(f"API response {ret}")
        return ret
