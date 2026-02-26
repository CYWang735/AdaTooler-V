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
"""
Preprocess the GSM8k dataset to parquet format
"""
import fire
import os
import datasets
import zipfile
import cv2
import os
import regex as re
from glob import glob
from pathlib import Path
from huggingface_hub import hf_hub_download
from collections import defaultdict
from copy import deepcopy

from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info


TYPE_TEMPLATE = {
    "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
    "numerical": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
    "OCR": " Please transcribe text from the image/video clearly and provide your text answer within the <answer> </answer> tags.",
    "free-form": " Please provide your text answer within the <answer> </answer> tags."
}



system_prompt = """You are a helpful assistant.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "crop_image", "description": "Zoom in on the image based on the bounding box coordinates.", "parameters": {"type": "object", "properties": {"bbox_2d": {"type": "array", "description": "coordinates for bounding box of the area you want to zoom in. minimum value is 0 and maximum value is the width/height of the image.", "items": {"type": "number"}}, "target_image": {"type": "number", "description": "The index of the image to crop. Index from 1 to the number of images. Choose 1 to operate on original image."}}, "required": ["bbox_2d", "target_image"]}}}
{"type": "function", "function": {"name": "select_frames", "description": "Select frames from a video.", "parameters": {"type": "object", "properties": {"target_frames": {"type": "array", "description": "List of frame indices to select from the video (no more than 8 frames in total).", "items": {"type": "integer", "description": "Frame index from 1 to 16."}}}, "required": ["target_frames"]}}}
{"type": "function", "function": {"name": "PathTracer", "description": "Plot movement or connections between two points on the specified image.", "parameters": {"type": "object", "properties": {"target_image": {"type": "number", "description": "The index of the image to crop. Index from 1 to the number of images. Choose 1 to operate on original image."}, "start_point_2d": {"type": "array", "description": "Starting point coordinates [x1, y1] of the path. minimum value is 0 and maximum value is the width/height of the image.", "items": {"type": "number"}}, "end_point_2d": {"type": "array", "description": "Ending point coordinates [x2, y2] of the path. minimum value is 0 and maximum value is the width/height of the image.", "items": {"type": "number"}}}, "required": ["start_point_2d", "end_point_2d", "target_image"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>"""


guideline = """Guidelines: Understand the given visual information and the user query. 

Determine if it is beneficial to employ the given visual operations (tools).

Determine which tool to use based on the input:
- For a single image, use `crop_image` or `PathTracer`.
- For a video, use `select_frames`, `crop_image`, or `PathTracer`.

Reason with the visual information step by step.
You should:
1. Explain why a tool is necessary.
2. Call the tool.
3. Continue reasoning based on the tool output.
4. Provide the final answer.

Place your text reasoning process within the <think> </think> tags.
Place any function calls within the <tool_call></tool_call> tags.
Place your final answer within the <answer> </answer> tags.
"""




def images_to_video(image_folder, output_path, fps=24):
    images = sorted(glob(os.path.join(image_folder, "*.jpg")))
    if not images:
        raise ValueError("No .jpg images found in folder.")

    # Read the first image to get size
    frame = cv2.imread(images[0])
    height, width, _ = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for img_path in images:
        frame = cv2.imread(img_path)
        out.write(frame)
    out.release()
    print(f"Video saved to {output_path}")

def get_mm_content_len(processor, example):
    # print(example)
    messages = deepcopy(example['prompt'])
    for message in messages:
        content = message["content"]
        content_list = []
        segments = re.split("(<image>|<video>)", content)
        segments = [item for item in segments if item != ""]
        segment_idx = defaultdict(int)
        for segment in segments:
            if segment == "<image>":
                content_list.append({"type": "image", "image": example['images'][segment_idx[segment]]["image"]})
                segment_idx[segment] += 1
            elif segment == "<video>":
                content_list.append({"type": "video", "video": example['videos'][segment_idx[segment]]["video"]})
                segment_idx[segment] += 1
            else:
                content_list.append({"type": "text", "text": segment})

        message["content"] = content_list
    raw_prompt = processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[raw_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    return inputs.input_ids.shape[1]

def main(
    dataset_path: str = 'TIGER-Lab/PixelReasoner-RL-Data',
    local_dir: str = 'data/pixel_reasoner',
    version: str = None,
    seed: int = 42,
    image_sep = "<image>",
    video_sep = "<video>",
    filter_len=None,
    include_videos=True,
):
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-32B-Instruct")
    local_dir = Path(local_dir)
    local_dir = local_dir / (dataset_path.split('/')[-1].replace('-', '_'))
    local_dir.mkdir(parents=True, exist_ok=True)
    
    dataset = datasets.load_dataset(dataset_path, split='train')

    # 500 examples for testing
    train_dataset = dataset

    image_dir = Path('/home/wangcy')
    video_dir = Path('/home/wangcy')
    

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            question_raw = example.pop('question')
            problem_type = example.get("problem_type")
            question_raw += f"\n\n{guideline}"
            question_raw += TYPE_TEMPLATE[problem_type]
            image = example.pop('image')
            is_video = example.pop('is_video')
            answer = example.pop('answer')
            # we use absolute paths for images and videos
            if is_video:
                assert all((video_dir / video).exists() for video in image), f"Some video files do not exist in {video_dir}"
                extra_info_images = [(video_dir / video).absolute().as_posix() for video in image]
            else:
                assert (image_dir / image[0]).exists(), f"Image file {image[0]} does not exist in {image_dir}"
                extra_info_images = [(image_dir / image[0]).absolute().as_posix()]
            mm_content = image_sep * len(extra_info_images) + question_raw

            data = {
                "data_source": dataset_path,
                "prompt": [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": mm_content,
                    }
                ],
                "images": [{"image": image} for image in extra_info_images],
                "ability": "visual_reasoning",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer,
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'qid': example.get('qid', f'{split}_{idx}'),
                    'is_video': bool(is_video),
                    'images': extra_info_images,
                    'problem_type': problem_type
                }
            }
            if filter_len and filter_len > 0:
                mm_content_len = get_mm_content_len(processor, data)
                data['extra_info']['mm_content_len'] = mm_content_len
            return data

        return process_fn

    if not include_videos:
        train_dataset = train_dataset.filter(lambda x: not x['is_video'], num_proc=8)
        print(f"Filtered out video examples. Remaining {len(train_dataset)} examples.")
    
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True, remove_columns=train_dataset.column_names, num_proc=32)
    if filter_len and filter_len > 0:
        _train_dataset = train_dataset.filter(lambda x: x['extra_info']['mm_content_len'] and x['extra_info']['mm_content_len'] <= filter_len, num_proc=8)
        print(f"Filtered {len(train_dataset) - len(_train_dataset)}/{len(train_dataset)} examples from training dataset due to content length > {filter_len}")
        train_dataset = _train_dataset
    # split 100 as val
    train_dataset, val_dataset = train_dataset.train_test_split(test_size=100, seed=seed).values()
    
    print(f"Loaded {len(train_dataset)} training samples")
    print(f"Loaded {len(val_dataset)} validation samples")
    print(f"Example of a training sample:")
    print(train_dataset[0])
    
    if version is not None:
        local_dir = local_dir / version
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    val_dataset.to_parquet(os.path.join(local_dir, 'val.parquet'))
    print(f"Saved to {len(train_dataset)} training samples to {local_dir}/train.parquet")
    print(f"Saved to {len(val_dataset)} validation samples to {local_dir}/val.parquet")

if __name__ == '__main__':
    fire.Fire(main)
    