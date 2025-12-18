# AdaTooler-V: Adaptive Tool-Use for Images and Videos

[[📖 Paper](https://arxiv.org/pdf/2503.21776)] [[🤗 AdaTooler-V-7B-model](https://huggingface.co/ChaoyangWang/AdaTooler-V-7B)] 
[[🤗 AdaTooler-V-SFT-model](ChaoyangWang/Qwen2.5-VL-7B-CoT-SFT)] 
[[🤗 AdaTooler-V-train-data](https://huggingface.co/datasets/ChaoyangWang/AdaTooler-V-300k)] [[🤗 AdaTooler-V-eval](ChaoyangWang/AdaTooler-V-eval)]



## 👀 About AdaTooler-V
We propose **AdaTooler-V**, an MLLM that **performs adaptive tool-use by determining whether a visual problem truly requires tools.**

First, we introduce **AT-GRPO**, a reinforcement learning algorithm that adaptively adjusts reward scales based on the Tool Benefit Score of each sample, encouraging the model to invoke tools only when they provide genuine improvements.

Moreover, we construct two datasets to support training: **AdaTooler-V-CoT-100k** for SFT cold start and **AdaTooler-V-300k** for RL with verifiable rewards across single-image, multi-image, and video data.

Experiments across twelve benchmarks demonstrate the strong reasoning capability of AdaTooler-V, outperforming existing methods in diverse visual reasoning tasks. Notably, AdaTooler-V-7B achieves an accuracy of 89.8\% on the high-resolution benchmark V*, **surpassing the
commercial proprietary model GPT-4o and Gemini 1.5 Pro.**


## 🔍 Dataset

 To overcome the scarcity of high-quality video reasoning training data, we strategically introduce image-based reasoning data as part of training data.  We collect data from a variety of public datasets and carefully sample and balance the proportion of each subset. 

<img src="./images/dataset.png" style="zoom:80%;" />

To facilitate an effective SFT cold start, we leverage Qwen2.5-VL-72B  to generate COT rationales for the samples in Video-R1-260k. After applying basic rule-based filtering to remove low-quality or inconsistent outputs, we obtain a high-quality CoT dataset, Video-R1-COT 165k.

## 🏆 Performance

<img src="./images/performance.png" style="zoom:80%;" />

Video-R1 significantly outperforms previous models across most benchmarks. Notably, on VSI-Bench, which focuses on spatial reasoning in videos, Video-R1-7B achieves a new state-of-the-art accuracy of 35.8%, surpassing GPT-4o, a proprietary model, while using only 32 frames and 7B parameters. 

This highlights the necessity of explicit reasoning capability in solving video tasks, and confirms the effectiveness of reinforcement learning for video tasks.


<div align="center">
  <img src="./images/frames.png" alt="Descriptive alt text" width="85%">
</div>

Besides, although the model is trained using only 16 frames, we find that evaluating on more frames (e.g., 64) generally leads to better performance, particularly on benchmarks with longer videos. These results indicate the importance of training models to reason over more frames.


## 🧠 Aha Moment in Video Reasoning

One of the most intriguing outcomes of reinforcement learning in Video-R1 is the emergence of self-reflection reasoning behaviors, commonly referred to as “aha moments”. Some examples are as follows.

<img src="./images/demo1.png" style="zoom:80%;" />

<img src="./images/demo2.png" style="zoom:80%;" />


## 📈 RL Training Curves

The accuracy reward exhibits a generally upward trend, indicating that the model continuously improves its ability to produce correct answers under RL.

Interestingly, the response length curve first drops at the beginning of RL training, then gradually increases. We guess this is because the model initially discards its previous, potentially sub-optimal reasoning style. Then gradually converges to a better and stable reasoning policy.

<img src="./images/curves.png" style="zoom:80%;" />



## 📐 Set up

```bash
git clone https://github.com/tulerfeng/Video-R1
cd Video-R1

# build environment
conda create -n video-r1 python=3.11 
conda activate video-r1
bash setup.sh

# qwen video extraction setting, e.g., max frames, resolutions
# Use the [decord] feature to improve speed
cd src/qwen-vl-utils
pip install -e .[decord]
cd ..

# download training dataset
git lfs install
git clone https://huggingface.co/datasets/Video-R1/Video-R1-data
```

Please put the downloaded dataset to `src/r1-v/Video-R1-data/`

Then, unzip the data

```
python ./src/unzip.py
```

The `Video-R1-260k.json` file is for RL training while `Video-R1-COT-165k.json` is for SFT cold start.

Qwen2.5-VL has been frequently updated in the Transformers library, which may cause version-related bugs or inconsistencies. Our code is compatible with the following version, please download at [here](https://drive.google.com/file/d/1Kc81WZitEhUZYWXpL6y2GXuSXufLSYcF/view?usp=sharing)

Then install our provided version of transformers

```bash
unzip transformers-main.zip
cd ./transformers-main
pip install .
```

For vLLM library, please use 0.7.2 version.

For trl library, please use 0.16.0 version.

## 🚀 Training

We first perform supervised fine-tuning on the Video-R1-COT-165k dataset for one epoch to obtain the Qwen2.5-VL-7B-SFT model. If you want to perform CoT annotation on your own data, please refer to `src/generate_cot_vllm.py`

```bash
bash ./src/scripts/run_sft_video.sh
```
If you want to skip the SFT process, we also provide one of our SFT models at [🤗Qwen2.5-VL-SFT](https://huggingface.co/Video-R1/Qwen2.5-VL-7B-COT-SFT). 

This is followed by RL training on the Video-R1-260k dataset to produce the final Video-R1 model. Due to current computational resource limitations, we train the model for only 1.2k RL steps.  

The script for training the obtained Qwen2.5-VL-7B-SFT model with T-GRPO or GRPO is as follows

```bash
bash ./src/scripts/run_grpo_video.sh
```

You can also use the following script to enable vLLM acceleration for RL training

```bash
bash ./src/scripts/run_grpo_vllm_qwen25vl.sh
```

For efficiency considerations, we limit the maximum number of video frames to 16 during training. Each frame is processed at a max resolution of 128 × 28 × 28.  You can set this in `src/qwen-vl-utils`

Please keep per_device_train_batch_size=1 as in previous work r1-v

## 🔮 Inference & Evaluation

During inference, we increase the max frame resolution to 256 × 28 × 28 and max frames to 16/32/64 to enhance performance. You can easily set this in `src/qwen-vl-utils`

For all evaluations, we follow the decoding configuration used in the official Qwen2.5-VL demo, with top\_p = 0.001 and temperature = 0.01. Setting large top_p may encounter messy output when inference.

We recommend using our provided json files and scripts for easier evaluation. 

The json files can be downloaded at: [[🤗 Video-R1-eval](https://huggingface.co/datasets/Video-R1/Video-R1-eval)], put them in `/src/r1-v/Evaluation` 

Next, download the evaluation video data from each benchmark’s official website, and place them in `/src/r1-v/Evaluation` as specified in the provided json files.

Finally, conduct evaluation on all benchmarks using the following scripts

```bash
bash ./src/eval_bench.sh
```
For infernce on a single example, you may use:

```bash
python ./src/inference_example.py
```

## Acknowledgements

We sincerely appreciate the contributions of the open-source community. The related projects are as follows: [R1-V](https://github.com/Deep-Agent/R1-V) , [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1) 

## Citations

If you find our work helpful for your research, please consider citing our work.   

```
@article{feng2025video,
  title={Video-R1: Reinforcing Video Reasoning in MLLMs},
  author={Feng, Kaituo and Gong, Kaixiong and Li, Bohao and Guo, Zonghao and Wang, Yibing and Peng, Tianshuo and Wang, Benyou and Yue, Xiangyu},
  journal={arXiv preprint arXiv:2503.21776},
  year={2025}
}
```
