import time

import openai
from openai.types.chat import ChatCompletion

system_prompt1 = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. User: Please integrate natural language reasoning with programs to solve the coding problems below. If you want to test any python code, writing it inside <python> and </python> tags following with <output>. Please put your final answer in a markdown code block like this: python\nyour code here\n``` without appending anything.
"""

system_prompt2 = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. User: Please integrate natural language reasoning with programs to solve the coding problems below. If the you want to run any python code, execution result will be in the output markdown block like "```output\nexecution result here\n```" following the code block. Please put your final answer in a markdown code block like this: python\nyour code here\n``` without appending anything.
"""

system_prompt3 = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. User: Please solve the coding problems below and put your final answer in a markdown code block like this: python\nyour code here\n``` without appending anything.
"""


def make_request(
    client: openai.Client,
    message: str,
    model: str,
    max_tokens: int = 512,
    temperature: float = 1,
    reasoning_effort: str = "medium",
    n: int = 1,
    **kwargs
) -> ChatCompletion:
    kwargs["top_p"] = 0.95
    # kwargs["max_tokens"] = max_tokens
    if model.startswith("o1-"):  # pop top-p and max_completion_tokens
        kwargs.pop("top_p")
        kwargs.pop("max_completion_tokens")
        temperature = 1.0  # o1 models do not support temperature

    template = f"system\n{system_prompt3}\n\nuser\n{message}\nassistant\n"
    
    return client.completions.create(
        model=model,
        prompt=template,
        temperature=temperature,
        max_tokens=max_tokens,
        n=n,
        **kwargs
    )
    # return client.chat.completions.create(
    #     model=model,
    #     messages=[
    #         {"role": "user", "content": message},
    #     ],
    #     temperature=temperature,
    #     n=n,
    #     **kwargs
    # )


def make_auto_request(*args, **kwargs) -> ChatCompletion:
    ret = None
    while ret is None:
        try:
            ret = make_request(*args, **kwargs)
        except openai.RateLimitError:
            print("Rate limit exceeded. Waiting...")
            time.sleep(5)
        except openai.APIConnectionError:
            print("API connection error. Waiting...")
            time.sleep(5)
        except openai.APIError as e:
            print(e)
        except Exception as e:
            print("Unknown error. Waiting...")
            print(e)
            time.sleep(1)
    return ret