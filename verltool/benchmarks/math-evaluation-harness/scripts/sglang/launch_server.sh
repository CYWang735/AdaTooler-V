export CUDA_VISIBLE_DEVICES=2
python -m sglang.launch_server --model-path Qwen/Qwen2.5-Math-1.5B-Instruct --port 30000  --schedule-conservativeness 0.3