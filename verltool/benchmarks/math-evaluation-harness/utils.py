import os
import json
import random
import json
import os
import numpy as np
from pathlib import Path
from typing import Iterable, Union, Any


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def load_jsonl(file: Union[str, Path]) -> Iterable[Any]:
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except:
                print("Error in loading:", line)
                exit()


def save_jsonl(samples, save_path):
    # ensure path
    folder = os.path.dirname(save_path)
    os.makedirs(folder, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    print("Saved to", save_path)


def lower_keys(example):  
    new_example = {}  
    for key, value in example.items():  
        if key != key.lower():  
            new_key = key.lower()  
            new_example[new_key] = value  
        else:  
            new_example[key] = value  
    return new_example 


def load_prompt(data_name, prompt_type): # TODO: update
    if data_name in ['gsm_hard', 'svamp', 'tabmwp', 'asdiv', 'mawps']:
        data_name = "gsm8k"
    if data_name in ['math_oai', "hungarian_exam"]:
        data_name = "math"
    if data_name in ['sat_math']:
        data_name = "mmlu_stem"
    if prompt_type in ['platypus_fs']:
        prompt_type = "cot"
    if prompt_type in ['tool-integrated']:
        prompt_type = "tora"

    if prompt_type in ['cot', 'pal', 'tora']:
        prompt_path = "./prompts/{}/{}.md".format(prompt_type, data_name)
        if not os.path.exists(prompt_path):
            prompt_path = "./prompts/{}.md".format(prompt_type)
        if os.path.exists(prompt_path):
            with open(prompt_path, 'r', encoding='utf-8') as fp:
                prompt = fp.read().strip() + "\n\n\n"
        else:
            print(f"Error: prompt file {prompt_path} not found")
            prompt = ""
    else:
        prompt = ""
    return prompt

def construct_prompt(example, data_name, args):
    # Base models
    if args.prompt_type in ["direct", "cot", "pal", "tool-integrated"]: 
        demo_prompt = load_prompt(data_name, args.prompt_type)
        if args.prompt_type in ["direct", "cot"]:
            if data_name in ["minerva_math", "math", "math_oai", "mmlu_stem", "sat_math", "mathqa", "hungarian_exam"]:
                context = f"Problem:\n{example['question']}\nSolution:"
            else:
                context = f"Question: {example['question']}\nAnswer:"
            full_prompt = demo_prompt + context
        elif args.prompt_type == "pal":
            context = f"Question: {example['question']}"
            full_prompt = demo_prompt + context
        elif args.prompt_type in ['tool-integreted']:
            context = f"Question: {example['question']}\n\nSolution:"
            full_prompt = demo_prompt + context
    # SFT models
    elif args.prompt_type in ['torl_deepmath_qwen']:
        full_prompt = f"<|im_start|>system\nA conversation between User and Assistant. The user asks a question, and the Assistant solves it. Please integrate natural language reasoning with programs to solve the problem above, and put your final answer within \\boxed{{}}.\n<|im_end|>\n<|im_start|>user\n{example['question']}<|im_end|>\n<|im_start|>assistant\n"
    elif args.prompt_type in ['tool_math_qwen_mtrl']:
        full_prompt = f"<|im_start|>system\nA conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. User: Please integrate natural language reasoning with programs to solve the problem above. If you want to run any code, include \"<|calling system for feedback|>\" at the end of your response for the current turn. Then the system will execute the code in the markdown block and provide the standard output and error. If you think the solution is complete and don't need to test, don't include \"<|calling system for feedback|>\" in the response and put your final answer within \\boxed{{}}. \n<|im_end|>\n<|im_start|>user\n{example['question']}<|im_end|>\n<|im_start|>assistant\n"
    elif args.prompt_type in ['tool_mathcoder_qwen']:
        full_prompt = f"system\nA conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. User: Please integrate natural language reasoning with programs to solve the problem above. For math problems, please put your final answer within \\boxed{{}}. For code problems, please put your final answer in a markdown code block like this: ```python\nyour code here\n```\n\nuser\n{example['question']}\nassistant\n"
    elif args.prompt_type in ['tool_math_qwen_r1']:
        full_prompt = f"system\nA conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. User: Please integrate natural language reasoning with programs to solve the problem above. If you want to test any python code, writing it inside <python> and  </python> tags following with <output>. Please put your final answer within \\boxed{{}}:\n\nuser\n{example['question']}\nassistant\n"
    elif args.prompt_type in ['tool_math_qwen_r1_python']:
        full_prompt = f"system\nA conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. User: Please integrate natural language reasoning with programs to solve the problem above. If you want to test any python code, writing it inside <python> and  </python> tags following with <output>. Please put your final answer within \\boxed{{}}:\n\nuser\n{example['question']}\nassistant\n"
    elif args.prompt_type in ['tool_math_qwen']:
        full_prompt = f"system\nA conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. User: Please integrate natural language reasoning with programs to solve the problem above, and put your final answer within \\boxed{{}}.:\n\nuser\n{example['question']}\nassistant\n"
    elif args.prompt_type in ['qwen-torl']:
        full_prompt = f"<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nPlease integrate natural language reasoning with programs to solve the problem above, and put your final answer within \\boxed{{}}.\n{example['question']}<|im_end|>\n<|im_start|>assistant\n"
    elif args.prompt_type in ['torl']:
        full_prompt = f"A conversation between User and Assistant. The user asks a question, and the Assistant solves it.\nUser: Please integrate natural language reasoning with programs to solve the problem above, and put your final answer within \\boxed{{}}.\n{example['question']}\nAssistant:"
    elif args.prompt_type in ['self-instruct', 'tora']:
        full_prompt = f"<|user|>\n{example['question']}\n<|assistant|>\n"
    elif args.prompt_type in ['self-instruct-boxed']:
        full_prompt = f"<|user|>\n{example['question']}\nEnclose the final answer using \\boxed{{}}.\n<|assistant|>\n"
    elif args.prompt_type == "wizard_zs":
        full_prompt = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
        )
        full_prompt = full_prompt.format(instruction=example['question'])
    elif args.prompt_type == "deepseek-math":
        full_prompt = (
            "User: {instruction}\nPlease reason step by step, "
            "and put your final answer within \\boxed{{}}.\n\nAssistant:"
        )
        full_prompt = full_prompt.format(instruction=example['question'])
    elif args.prompt_type == "kpmath":
        full_prompt = (
            'User: Please reason step by step and put your final answer at the end '
            'with "The answer is: ".\n\n{instruction}\n\nAssistant:'
        )
        full_prompt = full_prompt.format(instruction=example['question'])
    elif args.prompt_type == "qwen-r1":
        full_prompt = (
        "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\nAnswer the following question. You must conduct reasoning inside <think> and </think>, and provide the final answer directly inside <answer> and </answer> without detailed explanations. For example: <answer>42</answer> Question: {instruction}<|im_end|>\n"
        "<|im_start|>assistant\n"
        )
        full_prompt = full_prompt.format(instruction=example['question'])
    elif args.prompt_type == "pot-qwen-r1":
        full_prompt = (
        "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\nAnswer the following question.  You must conduct reasoning inside <think> and </think>. If, during reasoning, you determine that calculations or logical operations are needed, write Python code inside <python> and </python> to call the Python interpreter, ensuring that the desired result is placed inside the print function to interact with the Python interpreter. The output of the print function or any error message will be captured and returned from the Python interpreter between <information> and </information>. You can use the Python interpreter as many times as necessary. If no further calculations or logical operations are required, provide the final answer directly inside <answer> and </answer> without detailed explanations. For example: <answer>42</answer> Question: {instruction}<|im_end|>\n"
        "<|im_start|>assistant\n"
        )
        full_prompt = full_prompt.format(instruction=example['question'])
    elif args.prompt_type == "qwen25-math-cot":
        full_prompt = (
        "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
        "<|im_start|>user\n{input}<|im_end|>\n"
        "<|im_start|>assistant\n"
        )
        full_prompt = full_prompt.format(input=example['question'])
    else:
        raise NotImplementedError(args.prompt_type)
    return full_prompt

key_map = {
    "gt": "Ground Truth",
    "pred": "Prediction",
    "gt_cot": "Reference CoT",
    "score": "Score",
}

def show_sample(sample, print_all_preds=False):
    print("=="*20)
    for key in ["idx", "type", "level", "dataset"]:
        if key in sample:
            # capitalize
            print("{}: {}".format(key[0].upper() + key[1:], sample[key]))
    print("Question:", repr(sample['question']))
    if 'code' in sample:
        if print_all_preds:
            for code in sample['code']:
                print('-'*20)
                print("code:", code)
            print("Execution:", sample['report'])
        else:
            print("Solution:\n", sample['code'][0])
            print("Execution:", sample['report'][0])
    if 'pred' in sample:
        print("Prediction:", repr(sample['pred'][0]))
    for key in ["gt", "score", "unit", "gt_cot"]:
        if key in sample:
            _key  = key_map.get(key, key)
            print("{}: {}".format(_key, repr(sample[key])))
    print()
