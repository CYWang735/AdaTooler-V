set -ex

export CUDA_VISIBLE_DEVICES=0,1
# NOTE: Check if the prompt type is correct
PROMPT_TYPE="qwen25-math-cot"
MODEL_NAMES=(
    # Qwen/Qwen2.5-Math-1.5B-Instruct
    VerlTool/torl-fsdp-qwen_qwen2.5-math-1.5b-grpo-n16-b128-t1.0-lr1e-6new-no-toolusepenalty-430-step
    # VerlTool/torl-fsdp_agent-qwen_qwen2.5-coder-1.5b-grpo-n16-b128-t1.0-lr1e-6new-240-step
    # VerlTool/torl-fsdp-qwen_qwen2.5-coder-1.5b-grpo-n16-b128-t1.0-lr1e-6new-no-toolusepenalty-430-step
    # VerlTool/acecoder-fsdp_agent-qwen_qwen2.5-coder-1.5b-grpo-n16-b128-t1.0-lr1e-6-410-step
    # mergekit-community/Puffin-Qwen2.5-CodeMath
    # Qwen/Qwen2.5-Math-7B-Instruct
    # hkust-nlp/Qwen-2.5-Math-7B-SimpleRL-Zoo
    # Qwen/Qwen2.5-Math-1.5B
    # VerlTool/torl-fsdp_agent-qwen_qwen2.5-math-7b-grpo-n16-b128-t1.0-lr1e-6new-v2-430-step
    # VerlTool/torl-fsdp_agent-qwen_qwen2.5-7b-grpo-n16-b128-t1.0-lr1e-6new-190-step
    # VerlTool/torl-fsdp_agent-qwen_qwen2.5-math-7b-grpo-n16-b128-t1.0-lr1e-6new-220-step
    # VerlTool/torl-fsdp_agent-qwen_qwen2.5-math-1.5b-grpo-n16-b128-t1.0-lr1e-6new-v2-430-step
    # Qwen/Qwen3-4B 
    # VerlTool/acecoder-fsdp_agent-qwen_qwen2.5-coder-7b-grpo-n16-b128-t1.0-lr1e-6-340-step
    # VerlTool/mathcoder-fsdp_agent-qwen_qwen2.5-1.5b-grpo-n16-b128-t1.0-lr1e-6-420-step
    # VerlTool/mathcoder-fsdp_agent-qwen_qwen2.5-7b-grpo-n16-b128-t1.0-lr1e-6-360-step
    # "VerlTool/torl-fsdp_agent-qwen_qwen2.5-math-7b-grpo-n16-b128-t1.0-lr1e-6"
    # Add more model paths here, one per line
)
DATA_NAMES="gsm8k"
# DATA_NAMES="gsm8k,math500,minerva_math,olympiadbench,aime24,amc23"
SPLIT="test"
NUM_TEST_SAMPLE=-1

for MODEL_NAME_OR_PATH in "${MODEL_NAMES[@]}"; do
    echo "Evaluating model: ${MODEL_NAME_OR_PATH}"
    OUTPUT_DIR=results/${MODEL_NAME_OR_PATH}
    
    # single-gpu
    TOKENIZERS_PARALLELISM=false \
    python3 -u math_eval_sglang.py \
        --model_name_or_path ${MODEL_NAME_OR_PATH} \
        --output_dir ${OUTPUT_DIR} \
        --data_names ${DATA_NAMES} \
        --split ${SPLIT} \
        --prompt_type ${PROMPT_TYPE} \
        --num_test_sample ${NUM_TEST_SAMPLE} \
        --seed 0 \
        --temperature 0 \
        --n_sampling 1 \
        --top_p 1 \
        --start 0 \
        --end -1 \
        --save_outputs \
        --max_tokens_per_call 3072 \
        --use_sglang \
        --max_func_call 4 \
        --overwrite \
        2>&1 | tee "logs_${MODEL_NAME_OR_PATH//\//_}.log"
done

