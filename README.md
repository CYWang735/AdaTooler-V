# AdaTooler-V: Adaptive Tool-Use for Images and Videos

[[📖 Paper](https://arxiv.org/pdf/2503.21776)] [[🤗 AdaTooler-V-7B-model](https://huggingface.co/ChaoyangWang/AdaTooler-V-7B)] 
[[🤗 AdaTooler-V-SFT-model](ChaoyangWang/Qwen2.5-VL-7B-CoT-SFT)] 
[[🤗 AdaTooler-V-train-data](https://huggingface.co/datasets/ChaoyangWang/AdaTooler-V-300k)] [[🤗 AdaTooler-V-eval](ChaoyangWang/AdaTooler-V-eval)]



## 👀 About AdaTooler-V
We propose **AdaTooler-V**, an MLLM that performs adaptive tool-use by determining whether a visual problem truly requires tools. 
First, we introduce **AT-GRPO**, a reinforcement learning algorithm that adaptively adjusts reward scales based on the Tool Benefit Score of each sample, encouraging the model to invoke tools only when they provide genuine improvements.
Moreover, we construct two datasets to support training: **AdaTooler-V-CoT-100k** for SFT cold start and **AdaTooler-V-300k** for RL with verifiable rewards across single-image, multi-image, and video data.
Experiments across twelve benchmarks demonstrate the strong reasoning capability of AdaTooler-V, outperforming existing methods in diverse visual reasoning tasks. Notably, AdaTooler-V-7B achieves an accuracy of 89.8\% on the high-resolution benchmark V*, surpassing the
commercial proprietary model GPT-4o and Gemini 1.5 Pro.
