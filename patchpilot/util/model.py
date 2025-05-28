from abc import ABC, abstractmethod
from typing import List
import re


from patchpilot.util.api_requests import create_chatgpt_config, request_chatgpt_engine, create_anthropic_config, \
    request_anthropic_engine, request_chatgpt_prefill_engine, request_chatgpt_response_engine

import litellm


class DecoderBase(ABC):
    def __init__(
            self,
            name: str,
            logger,
            batch_size: int = 1,
            temperature: float = 0.8,
            max_new_tokens: int = 1024,
    ) -> None:
        print("Initializing a decoder model: {} ...".format(name))
        self.name = name
        self.logger = logger
        self.batch_size = batch_size
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    @abstractmethod
    def codegen(self, message: str, num_samples: int = 1) -> List[dict]:
        pass

    @abstractmethod
    def is_direct_completion(self) -> bool:
        pass

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name


class OpenSourceChatDecoder(DecoderBase):
    def __init__(self, name, logger, batch_size=1, temperature=0.8, max_new_tokens=1024):
        super().__init__(name, logger, batch_size, temperature, max_new_tokens)

    def codegen(self, message: str, num_samples: int = 1, **kwargs) -> List[dict]:
        port = kwargs.get("port", 2951)
        ip = kwargs.get("ip", "0.0.0.0")
        if "port" in kwargs:
            del kwargs["port"]
        if "ip" in kwargs:
            del kwargs["ip"]
        if self.temperature == 0:
            assert num_samples == 1
        trajs = []
        for _ in range(num_samples):
            config = create_chatgpt_config(
                message=message,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                batch_size=1,
                model=self.name,
                **kwargs,
            )
            if "reasoning_mode" in config:
                del config["reasoning_mode"]
            ret = request_chatgpt_engine(
                config, self.logger, base_url=f"http://{ip}:{port}/v1", api_key="sk1"
            )

            if ret:
                content = ret.choices[0].message.content
                reasoning_content_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)

                if reasoning_content_match:
                    reasoning_content = reasoning_content_match.group(1).strip()
                else:
                    reasoning_content = ""
                trajs.append(
                    {
                        "response": content,
                        "reasoning_content": reasoning_content,
                        "usage": {
                            "completion_tokens": ret.usage.completion_tokens,
                            "prompt_tokens": ret.usage.prompt_tokens,
                        },
                    }
                )
            else:
                trajs.append(
                    {
                        "response": "",
                        "reasoning_content": "",
                        "usage": {
                            "completion_tokens": 0,
                            "prompt_tokens": 0,
                        },
                    }
                )

        return trajs


    def codegen_prefill(self, message: str, prefill: str, num_samples: int = 1, **kwargs) -> List[dict]:
        port = kwargs.get("port", 2951)
        ip = kwargs.get("ip", "0.0.0.0")
        if "port" in kwargs:
            del kwargs["port"]
        if "ip" in kwargs:
            del kwargs["ip"]
        system_prompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        prompt += f"<|im_start|>user\n{message}<|im_end|>\n"
        prompt += f"<|im_start|>assistant\n{prefill}"

        config = {
            "prompt":prompt,
            "max_tokens":self.max_new_tokens,
            "temperature":self.temperature,
            "model":self.name,
        }

        if self.temperature == 0:
            assert num_samples == 1
        trajs = []
        for _ in range(num_samples):
            ret = request_chatgpt_prefill_engine(
                config, self.logger, base_url=f"http://{ip}:{port}/v1", api_key="sk1"
            )
            content = ret.choices[0].text
            full_content = prefill + content
            reasoning_content_match = re.search(r"<think>(.*?)</think>", full_content, re.DOTALL)

            if reasoning_content_match:
                reasoning_content = reasoning_content_match.group(1).strip()
            else:
                reasoning_content = ""

            if ret:
                trajs.append(
                    {
                        "response": full_content,
                        "short_content": content,
                        "reasoning_content": reasoning_content,
                        "usage": {
                            "completion_tokens": ret.usage.completion_tokens,
                            "prompt_tokens": ret.usage.prompt_tokens,
                        },
                    }
                )
            else:
                trajs.append(
                    {
                        "response": "",
                        "short_content": "",
                        "reasoning_content": "",
                        "usage": {
                            "completion_tokens": 0,
                            "prompt_tokens": 0,
                        },
                    }
                )

        return trajs



    def is_direct_completion(self) -> bool:
        return False


class OpenAIChatDecoder(DecoderBase):
    def __init__(self, name: str, logger, **kwargs) -> None:
        super().__init__(name, logger, **kwargs)

    def codegen(self, message: str, num_samples: int = 1, **kwargs) -> List[dict]:
        if "port" in kwargs:
            del kwargs["port"]
        if "ip" in kwargs:
            del kwargs["ip"]

        if self.temperature == 0:
            assert num_samples == 1
        batch_size = min(self.batch_size, num_samples)
        config = create_chatgpt_config(
            message=message,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            batch_size=batch_size,
            model=self.name,
            **kwargs,
        )
        if 'o1' in self.name or 'o3-mini' in self.name:  # o1 doesn't support sampling multiple completions
            responses = []
            all_tool_calls = []
            for _ in range(batch_size):
                ret = request_chatgpt_engine(config, self.logger)
                tool_calls_one_sample = None
                if hasattr(ret.choices[0].message, 'tool_calls'):
                    tool_calls_one_sample = ret.choices[0].message.tool_calls
                if ret:
                    response=ret.choices[0].message.content
                    completion_tokens = ret.usage.completion_tokens
                    prompt_tokens = ret.usage.prompt_tokens
                    respose_block = {
                        "response": response,
                        "tool_call": tool_calls_one_sample,
                        "usage": {
                            "completion_tokens": completion_tokens,
                            "prompt_tokens": prompt_tokens,
                        },
                    }
                else:
                    respose_block = {
                        "response": "",
                        "tool_call": None,
                        "usage": {
                            "completion_tokens": 0,
                            "prompt_tokens": 0,
                        },
                    }
                responses.append(respose_block)
            trajs = [responses[0]]
            for response_block in responses[1:]:
                trajs.append(response_block)
            return trajs
        elif 'o3' in self.name or "o4-mini" in self.name:
            responses = []
            for _ in range(batch_size):
                ret = request_chatgpt_response_engine(config, self.logger)
                tool_calls_one_sample = None
                if ret:
                    response=ret.output_text
                    completion_tokens = ret.usage.output_tokens
                    prompt_tokens = ret.usage.input_tokens
                    respose_block = {
                        "response": response,
                        "tool_call": tool_calls_one_sample,
                        "usage": {
                            "completion_tokens": completion_tokens,
                            "prompt_tokens": prompt_tokens,
                        },
                    }
                else:
                    respose_block = {
                        "response": "",
                        "tool_call": None,
                        "usage": {
                            "completion_tokens": 0,
                            "prompt_tokens": 0,
                        },
                    }
                responses.append(respose_block)
            trajs = [responses[0]]
            for response_block in responses[1:]:
                trajs.append(response_block)
            return trajs
        else:
            ret = request_chatgpt_engine(config, self.logger)
            if ret:
                responses = [choice.message.content for choice in ret.choices]
                all_tool_calls = []
                for choice in ret.choices:
                    if hasattr(choice.message, 'tool_calls'):
                        tool_calls_one_sample = ret.choices[0].message.tool_calls
                        all_tool_calls.append(tool_calls_one_sample)
                    else:
                        all_tool_calls.append(None)
                completion_tokens = ret.usage.completion_tokens
                prompt_tokens = ret.usage.prompt_tokens
            else:
                responses = [""]
                all_tool_calls = [None]
                completion_tokens = 0
                prompt_tokens = 0

            # The nice thing is, when we generate multiple samples from the same input (message),
            # the input tokens are only charged once according to openai API.
            # Therefore, we assume the request cost is only counted for the first sample.
            # More specifically, the `prompt_tokens` is for one input message,
            # and the `completion_tokens` is the sum of all returned completions.
            # Therefore, for the second and later samples, the cost is zero.
            trajs = [
                {
                    "response": responses[0],
                    "tool_call": all_tool_calls[0],
                    "usage": {
                        "completion_tokens": completion_tokens,
                        "prompt_tokens": prompt_tokens,
                    },
                }
            ]
            for index in range(1, len(responses)):
                trajs.append(
                    {
                        "response": responses[index],
                        "tool_call": all_tool_calls[index],
                        "usage": {
                            "completion_tokens": 0,
                            "prompt_tokens": 0,
                        },
                    }
                )
            return trajs

    def is_direct_completion(self) -> bool:
        return False


class DeepSeekChatDecoder(DecoderBase):
    def __init__(self, name: str, logger, **kwargs) -> None:
        super().__init__(name, logger, **kwargs)

    def codegen_litellm(self, message: str, num_samples: int = 1, **kwargs) -> List[dict]:
        if "port" in kwargs:
            del kwargs["port"]
        if "ip" in kwargs:
            del kwargs["ip"]

        if self.temperature == 0:
            assert num_samples == 1
        batch_size = min(self.batch_size, num_samples)
        responses = []
        query_message=[{"role": "user", "content": message}]
        for _ in range(batch_size):
            ret = litellm.completion(
                model=self.name,
                messages=query_message,
                max_tokens=self.max_new_tokens,
                tool_choice="auto",  # auto is default, but we'll be explicit
                temperature=self.temperature,
                **kwargs,
            )
            if ret:
                responses = [choice.message.content for choice in ret.choices]
                all_tool_calls = []
                for choice in ret.choices:
                    if hasattr(choice.message, 'tool_calls'):
                        tool_calls_one_sample = ret.choices[0].message.tool_calls
                        all_tool_calls.append(tool_calls_one_sample)
                    else:
                        all_tool_calls.append(None)
                completion_tokens = ret.usage.completion_tokens
                prompt_tokens = ret.usage.prompt_tokens
            else:
                responses = [""]
                all_tool_calls = [None]
                completion_tokens = 0
                prompt_tokens = 0
            trajs = [
                {
                    "response": responses[0],
                    "tool_call": all_tool_calls[0],
                    "usage": {
                        "completion_tokens": completion_tokens,
                        "prompt_tokens": prompt_tokens,
                    },
                }
            ]
            for index in range(1, len(responses)):
                trajs.append(
                    {
                        "response": responses[index],
                        "tool_call": all_tool_calls[index],
                        "usage": {
                            "completion_tokens": 0,
                            "prompt_tokens": 0,
                        },
                    }
                )
            return trajs

    def codegen(self, message: str, num_samples: int = 1, **kwargs) -> List[dict]:
        if "port" in kwargs:
            del kwargs["port"]
        if "ip" in kwargs:
            del kwargs["ip"]

        if self.temperature == 0:
            assert num_samples == 1
        trajs = []
        self.together_deepseek = True
        for _ in range(num_samples):
            if self.together_deepseek:
                from together import Together
                messages = [{"role": "user", "content": message}]
                client = Together()
                response = client.chat.completions.create(
                    model="deepseek-ai/DeepSeek-R1",
                    messages=messages,
                    temperature=self.temperature,
                )
                content = response.choices[0].message.content
                reasoning_content_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)

                if reasoning_content_match:
                    reasoning_content = reasoning_content_match.group(1).strip()
                else:
                    reasoning_content = ""

                content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

                trajs.append(
                    {
                        "response": content,
                        "reasoning_content": reasoning_content,
                    }
                )

            else:
                config = create_chatgpt_config(
                    message=message,
                    max_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    batch_size=1,
                    model=self.name,
                    **kwargs,
                )
                ret = request_chatgpt_engine(
                    config, self.logger, base_url="https://api.deepseek.com"
                )
                if ret:
                    trajs.append(
                        {
                            "response": ret.choices[0].message.content,
                            "reasoning_content": ret.choices[0].message.model_extra["reasoning_content"],
                            "usage": {
                                "completion_tokens": ret.usage.completion_tokens,
                                "prompt_tokens": ret.usage.prompt_tokens,
                            },
                        }
                    )
                else:
                    trajs.append(
                        {
                            "response": "",
                            "usage": {
                                "completion_tokens": 0,
                                "prompt_tokens": 0,
                            },
                        }
                    )

        return trajs

    def is_direct_completion(self) -> bool:
        return False


class ClaudeChatDecoder(DecoderBase):
    def __init__(self, name: str, logger, **kwargs) -> None:
        super().__init__(name, logger, **kwargs)

    def codegen_litellm(self, message: str, num_samples: int = 1, **kwargs) -> List[dict]:
        if "port" in kwargs:
            del kwargs["port"]
        if "ip" in kwargs:
            del kwargs["ip"]

        if self.temperature == 0:
            assert num_samples == 1
        batch_size = min(self.batch_size, num_samples)
        responses = []
        query_message=[{"role": "user", "content": message}]
        for _ in range(batch_size):
            ret = litellm.completion(
                model=self.name,
                messages=query_message,
                max_tokens=self.max_new_tokens,
                tool_choice="auto",  # auto is default, but we'll be explicit
                temperature=self.temperature,
                **kwargs,
            )
            if ret:
                responses = [choice.message.content for choice in ret.choices]
                all_tool_calls = []
                for choice in ret.choices:
                    if hasattr(choice.message, 'tool_calls'):
                        tool_calls_one_sample = ret.choices[0].message.tool_calls
                        all_tool_calls.append(tool_calls_one_sample)
                    else:
                        all_tool_calls.append(None)
                completion_tokens = ret.usage.completion_tokens
                prompt_tokens = ret.usage.prompt_tokens
            else:
                responses = [""]
                all_tool_calls = [None]
                completion_tokens = 0
                prompt_tokens = 0
            trajs = [
                {
                    "response": responses[0],
                    "tool_call": all_tool_calls[0],
                    "usage": {
                        "completion_tokens": completion_tokens,
                        "prompt_tokens": prompt_tokens,
                    },
                }
            ]
            for index in range(1, len(responses)):
                trajs.append(
                    {
                        "response": responses[index],
                        "tool_call": all_tool_calls[index],
                        "usage": {
                            "completion_tokens": 0,
                            "prompt_tokens": 0,
                        },
                    }
                )
            return trajs

    def codegen(self, message: str, num_samples: int = 1, **kwargs) -> List[dict]:
        if "port" in kwargs:
            del kwargs["port"]
        if "ip" in kwargs:
            del kwargs["ip"]

        if self.temperature == 0:
            assert num_samples == 1
        batch_size = min(self.batch_size, num_samples)
        config = create_anthropic_config(
            message=message,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            batch_size=batch_size,
            model=self.name,
            **kwargs,
        )

        reasoning_mode = kwargs.get("reasoning_mode", False)
        if reasoning_mode:
            thinking = {
                "type": "enabled",
                "budget_tokens": 2200,
            }
            config["thinking"] = thinking
            config["temperature"] = 1

            responses = []
            for _ in range(batch_size):
                ret = request_anthropic_engine(config, self.logger)
                if ret:
                    for choice in ret.content:
                        if choice.type == "thinking":
                            thinking = choice.thinking
                        elif choice.type == "text":
                            response = choice.text
                    completion_tokens = ret.usage.output_tokens
                    prompt_tokens = ret.usage.input_tokens

                    respose_block = {
                        "response": response,
                        "reasoning_content": thinking,
                        "usage": {
                            "completion_tokens": completion_tokens,
                            "prompt_tokens": prompt_tokens,
                        },
                    }
                else:
                    respose_block = {
                        "response":"",
                        "reasoning_content": "",
                        "usage": {
                            "completion_tokens": 0,
                            "prompt_tokens": 0,
                        },
                    }
                responses.append(respose_block)
            trajs = [responses[0]]
            for response_block in responses[1:]:
                trajs.append(response_block)
            return trajs

        else:
            responses = []
            for _ in range(batch_size):
                ret = request_anthropic_engine(config, self.logger)
                if ret:
                    for choice in ret.content:
                        response = choice.text
                    completion_tokens = ret.usage.output_tokens
                    prompt_tokens = ret.usage.input_tokens

                    respose_block = {
                        "response": response,
                        "usage": {
                            "completion_tokens": completion_tokens,
                            "prompt_tokens": prompt_tokens,
                        },
                    }
                else:
                    respose_block = {
                        "response":"",
                        "usage": {
                            "completion_tokens": 0,
                            "prompt_tokens": 0,
                        },
                    }
                responses.append(respose_block)
            trajs = [responses[0]]
            for response_block in responses[1:]:
                trajs.append(response_block)
            return trajs

    def is_direct_completion(self) -> bool:
        return False


def make_model(
        model: str,
        backend: str,
        logger,
        batch_size: int = 1,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        **kwargs,
):
    if backend == "openai":
        return OpenAIChatDecoder(
            name=model,
            logger=logger,
            batch_size=batch_size,
            max_new_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
    elif backend == "deepseek":
        return DeepSeekChatDecoder(
            name=model,
            logger=logger,
            batch_size=batch_size,
            max_new_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
    elif backend == "claude":
        return ClaudeChatDecoder(
            name=model,
            logger=logger,
            batch_size=batch_size,
            max_new_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
    elif backend == "opensource":
        return OpenSourceChatDecoder(
            name=model,
            logger=logger,
            batch_size=batch_size,
            max_new_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
    else:
        raise NotImplementedError
