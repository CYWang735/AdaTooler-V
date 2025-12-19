# LLM Math Evaluation Harness
## How to add new prompt?
+ **Step 1**: Add your prompt in `./utils.py`.

+ **Step 2**: Add your model's stop words in `./math_eval.py`.  

As an example, you can search for `torl` to see how it has been integrated and modified.

## Overview
A unified, precise, and extensible toolkit to benchmark LLMs on various mathematical tasks 🧮✨.

### Features:

- **Models**: Seamless compatibility with models from Hugging Face 🤗 and [vLLM](https://github.com/vllm-project/vllm).

- **Datasets**: An extensive array of datasets including `GSM8K`, `MATH 500`, `Minerva Math`, `Olympiad Bench`, `AIME24`, and `AMC23`.

- **Prompts**: Diverse prompting paradigms, from Direct to Chain-of-Thought (CoT), Program-of-Thought (PoT/PAL), and [Tool-Integrated Reasoning (ToRA)](https://github.com/microsoft/ToRA).


## 🚀 Getting Started

### ⚙️ Environment Setup
```
cd math-evaluation-harness
pip install uv # if not install uv
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

### ⚖️ Evaluation
Here we evaluate `Qwen-2.5-Math-1.5B/7B-Verl-Tool` using the following script. More examples can be found in [./scripts](./scripts).

```bash
# Qwen-2.5-Math-1.5B/7B-Verl-Tool
bash scripts/run_eval_math_greedy_deepmath.sh 
```

## Acknowledge
The codebase is adapted from math-evaluation-harness.
