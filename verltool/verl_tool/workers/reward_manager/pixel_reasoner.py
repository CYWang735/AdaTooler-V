# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import torch
import json
import time
import numpy as np
import regex as re
from .utils import replace_consecutive_tokens
from .reward_score import _default_compute_score
from .reward_score.torl_math import compute_score as torl_compute_score
from verl.workers.reward_manager import register
from collections import defaultdict
from .torl import ToRLRewardManager
from math_verify import parse, verify
from pathlib import Path
from verl import DataProto


def extract_answer(text):
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return match.group(1).strip() if match else None


def math_format_reward(predict_str: str) -> float:
    pattern = re.compile(r"<think>(.*?)</think>.*<answer>(.*?)</answer>.*", re.DOTALL)
    format_match = re.fullmatch(pattern, predict_str)
    return 1.0 if format_match else 0.0


def math_acc_reward(predict_str: str, ground_truth: str) -> float:
    match = re.search(r"<answer>(.*?)</answer>", predict_str, re.DOTALL)
    if match:
        predict_str_strip= match.group(1)
    else:
        predict_str_strip = predict_str
    # answer = extract_boxed_content(predict_str_strip)
    return 1.0 if grade_answer(predict_str_strip, ground_truth) else 0.0


def math_compute_score(predict_str: str, ground_truth: str, steps = 100000) -> float:
    if math_format_reward(predict_str):
        match = re.search(r"<answer>(.*?)</answer>", predict_str, re.DOTALL)
        if match:
            predict_str_strip= match.group(1)
        else:
            predict_str_strip = predict_str
        return math_acc_reward(predict_str_strip, ground_truth)
    else:
        return 0.0

def choices_compute_score(predict: str, ground_truth: list, steps = 100000) -> float:
    if math_format_reward(predict):
        match = re.search(r"<answer>(.*?)</answer>", predict, re.DOTALL)
        if match:
            
            predict_str_strip= match.group(1)
        else:
            predict_str_strip = predict
        if predict == "":
            return 0.0
        answer = [a.strip() for a in predict_str_strip.split(',')]
        if len(answer)!= len(ground_truth):
            return 0.0
        for a in answer:
            if a not in ground_truth:
                return 0.0
        return 1.0
    else:
        return 0.0

def ocr_compute_score(predict_str: str, ground_truth: str, steps = 100000) -> float:
    if math_format_reward(predict_str):
        match = re.search(r"<answer>(.*?)</answer>", predict_str, re.DOTALL)
        if match:
            predict_str_strip= match.group(1)
        else:
            predict_str_strip = predict_str
        werate=wer(hypothesis=predict_str_strip.strip(), reference=ground_truth.strip())
        return np.exp(-werate)
    else:
        return 0.0

def free_form_compute_score(predict_str: str, ground_truth: str, steps = 100000) -> float:
    if math_format_reward(predict_str):
        match = re.search(r"<answer>(.*?)</answer>", predict_str, re.DOTALL)
        if match:
            predict_str_strip= match.group(1)
        else:
            predict_str_strip = predict_str
        predict_str_strip = predict_str_strip.replace("\n", " ")
        ground_truth=ground_truth.replace("\n", " ")
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(predict_str_strip.strip(), ground_truth.strip())
        r1 = scores['rouge1'].fmeasure
        r2 = scores['rouge2'].fmeasure
        rl = scores['rougeL'].fmeasure
        return (r1+r2+rl)/3
    else:
        return 0.0



def normalize_answer(answer):
    if answer is None: return answer
    if 'dfrac' in answer: answer = answer.replace("dfrac", "frac")
    # if '%' in answer: answer = answer.replace(r'\%',"").replace('%',"")
    if 'text' in answer: answer = answer.replace("\\text","")
    if "\\varnothing" in answer: answer = answer.replace("\\varnothing","\\emptyset")
    if "minutes" in answer: answer = answer.replace("minutes","")
    if "cm" in answer: answer = answer.replace("cm","")
    # if "^\\circ" in answer: answer = answer.replace("^\\circ","")
    # if "a.m." in answer: answer = answer.replace("a.m.","")
    return answer 


def compute_ATReward(delta_s, n_tool, n_max, gamma):
    """
    Compute R_i^t = ΔS_i * exp( -γ * ((n_tool - n_max) / n_max)^2 )

    Args:
        delta_s (float or np.ndarray): ΔS_i
        n_tool (int or np.ndarray): n_tool
        n_max (int or float): n_max (> 0)
        gamma (float): γ

    Returns:
        float or np.ndarray: R_i^t
    """
    assert n_max > 0, "n_max must be positive"

    ratio = (n_tool - n_max) / n_max
    reward = delta_s * np.exp(-gamma * ratio ** 2)
    return reward


def pixel_reasoner_score(predict_str: str, ground_truth, problem_type: str) -> float:
    if problem_type == "numerical":
        return math_compute_score(predict_str, ground_truth)
    elif problem_type == "multiple choice":
        return choices_compute_score(predict_str, ground_truth)
    elif problem_type == "OCR":
        return ocr_compute_score(predict_str, ground_truth)
    else:  # free-form
        return free_form_compute_score(predict_str, ground_truth)

@register("pixel_reasoner")
class PixelReasonerRewardManager:
    """
    A reward manager for the Pixel Reasoner.
    It uses the TORL framework to compute rewards based on the outputs of the model.
    """
    name = "pixel_reasoner"
    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key='data_source', **kwargs) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = pixel_reasoner_score
        self.reward_fn_key = reward_fn_key
        self.add_curiousity_penalty = True
        self.add_action_redundancy_penalty = True
        self.group_tool_call_rate_lower_bound = 0.3 # H in the paper
        self.action_max_limit = 6 # n_{vo} in the paper, add penalty if the number of redundant actions is larger than this limit
        self.alpha = 0.5
        self.beta = 0.05
        self.gamma = 2

    def get_group_info(self, data: DataProto):
        group_info = {}
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            tool_interact_info = data_item.non_tensor_batch.get('tool_interact_info', None)
            num_turn = len(tool_interact_info) if tool_interact_info is not None else 0
            num_valid_action = sum([1 for t in tool_interact_info if t.get('valid_action', False)]) if tool_interact_info is not None else 0
            if "tool_interact_info" in data_item.non_tensor_batch:
                uid = data_item.non_tensor_batch.get('uid', i)
                if uid not in group_info:
                    group_info[uid] = {}
                if 'num_turns' not in group_info[uid]:
                    group_info[uid]['num_turns'] = []
                if 'num_valid_actions' not in group_info[uid]:
                    group_info[uid]['num_valid_actions'] = []
                group_info[uid]['num_turns'].append(num_turn)
                group_info[uid]['num_valid_actions'].append(num_valid_action)
        for uid, info in group_info.items():
            info['num_turns'] = np.array(info['num_turns'])
            info['num_valid_actions'] = np.array(info['num_valid_actions'])
            info['group_tool_call_rate'] = np.mean([1 if num_valid_action > 0 else 0 for num_valid_action in info['num_valid_actions']])
            info['tool_call_total'] = info['num_valid_actions'].sum()
        return group_info    
    
    def add_additional_penalties(self, response: str, data_i, scores_i: dict, group_info:dict, extra_info:dict):
        
        if "tool_interact_info" in data_i.non_tensor_batch:
            tool_interact_info = data_i.non_tensor_batch.get('tool_interact_info', None)
            num_turn = len(tool_interact_info) if tool_interact_info is not None else 0
            num_valid_action = sum([1 for t in tool_interact_info if t.get('valid_action', False)]) if tool_interact_info is not None else 0
            TB_score = extra_info["Tool_Benefit_Score"]
            
            AT_reward = compute_ATReward(TB_score, num_valid_action, self.action_max_limit, self.gamma)
            scores_i['score'] += self.alpha * AT_reward
        
        return scores_i
    
    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        group_info = self.get_group_info(data)
        for i in range(len(data)):
            score = {}
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            if "loss_mask" in data_item.batch:
                loss_mask = data_item.batch['loss_mask']
                valid_response_ids_with_loss_mask = torch.where(loss_mask[prompt_length:prompt_length + valid_response_length] == 1, valid_response_ids, self.tokenizer.pad_token_id)
            else:
                valid_response_ids_with_loss_mask = valid_response_ids

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            extra_info = data_item.non_tensor_batch.get('extra_info', None)


            torl_score = self.compute_score(
                # data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                problem_type = extra_info["problem_type"],
                # extra_info=extra_info,
            ) # 1 or -1
            score['accuracy'] = 1. if torl_score > 0 else 0.
            score['score'] = torl_score

            # add additional penalty
            score = self.add_additional_penalties(response_str, data_item, score, group_info.get(data_item.non_tensor_batch.get('uid', i), {}), extra_info)      

            if score['accuracy'] > 0:
                reward_extra_info['correct_response_length'].append(valid_response_length)
            else:
                reward_extra_info['wrong_response_length'].append(valid_response_length)

            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
                if self.num_examine == 1:
                    reward = score["accuracy"] # for validation
            else:
                if self.num_examine == 1:
                    reward = score if score > 0 else 0.0
                else:
                    reward = score

            reward_tensor[i, valid_response_length - 1] = reward 

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print(f"[score]", score)
                    
            # Save the records
            tool_interact_info_i = data_item.non_tensor_batch.get('tool_interact_info', None)
            if tool_interact_info_i is not None:
                # crop the image
                for tool_interact in tool_interact_info_i:
                    if "image" in tool_interact:
                        if isinstance(tool_interact['image'], list):
                            tool_interact['image'] = [x[:50] for x in tool_interact['image']]  # crop the image to first 50 characters
                        elif isinstance(tool_interact['image'], str):
                            tool_interact['image'] = tool_interact['image'][:50] # for debug
            
        correct_response_length_mean = np.mean(reward_extra_info['correct_response_length']) if reward_extra_info['correct_response_length'] else 0.0
        wrong_response_length_mean = np.mean(reward_extra_info['wrong_response_length']) if reward_extra_info['wrong_response_length'] else 0.0
        reward_extra_info['correct_response_length'] = [correct_response_length_mean] * len(reward_tensor)
        reward_extra_info['wrong_response_length'] = [wrong_response_length_mean] * len(reward_tensor)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": dict(sorted(reward_extra_info.items())),
            }
        else:
            return reward_tensor


