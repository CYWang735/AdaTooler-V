set -ex

export CUDA_VISIBLE_DEVICES=1
# NOTE: Check if the prompt type is correct
PROMPT_TYPE="tool_math_qwen_mtrl"
MODEL_NAMES=(
    VerlTool/torl-fsdp_agent-qwen_qwen2.5-math-1.5b-grpo-n16-b128-t1.0-lr1e-6-mtrl-v6-280-step
    # VerlTool/torl-fsdp_agent-qwen_qwen2.5-math-1.5b-grpo-n16-b128-t1.0-lr1e-6-mtrl-v6-280-step
    # VerlTool/torl-fsdp_agent-qwen_qwen2.5-math-1.5b-grpo-n16-b128-t1.0-lr1e-6-mtrl-v6-140-step
    # VerlTool/torl-fsdp-qwen_qwen2.5-math-1.5b-grpo-n16-b128-t1.0-lr1e-6new-no-toolusepenalty-430-step
    # VerlTool/torl-fsdp-qwen_qwen2.5-math-1.5b-grpo-n16-b128-t1.0-lr1e-6new-no-toolusepenalty-430-step
    # VerlTool/torl-fsdp_agent-qwen_qwen2.5-math-1.5b-grpo-n16-b128-t1.0-lr1e-6v2-reproduce-430-step
    # VerlTool/torl-fsdp_agent-qwen_qwen2.5-coder-1.5b-grpo-n16-b128-t1.0-lr1e-6new-no-toolusepenalty-430-step
    # VerlTool/torl-fsdp_agent-qwen_qwen2.5-coder-1.5b-grpo-n16-b128-t1.0-lr1e-6new-240-step
    # VerlTool/Qwen2.5-Math-1.5B-TIR-SFT
    # VerlTool/Qwen2.5-Math-1.5B-TIR-SFT-new
    # VerlTool/acecoder-fsdp_agent-qwen_qwen2.5-coder-1.5b-grpo-n16-b128-t1.0-lr1e-6-410-step
    # VerlTool/torl-fsdp_agent-qwen_qwen2.5-7b-grpo-n16-b128-t1.0-lr1e-6new-190-step
    # VerlTool/torl-fsdp_agent-qwen_qwen2.5-math-1.5b-grpo-n16-b128-t1.0-lr1e-6new-320-step
    # VerlTool/torl-fsdp_agent-qwen_qwen2.5-math-1.5b-grpo-n16-b128-t1.0-lr1e-6new-no-toolusepenalty-360-step
    # VerlTool/torl-fsdp_agent-qwen_qwen2.5-math-1.5b-grpo-n16-b128-t1.0-lr1e-6torl_same_train-310-step
    # VerlTool/torl-fsdp_agent-qwen_qwen2.5-coder-1.5b-grpo-n16-b128-t1.0-lr1e-6new-240-step
    # VerlTool/mathcoder-qwen_qwen2.5-coder-1.5b-grpo-n16-b128-t1.0-lr1e-6new-no-toolusepenalty-390-step
    # VerlTool/mathcoder-fsdp_agent-qwen_qwen2.5-coder-1.5b-grpo-n16-b128-t1.0-lr1e-6new-330-step
    # VerlTool/Qwen2.5-Math-1.5B-TIR-SFT
    # Qwen/Qwen2.5-Coder-1.5B-Instruct
    # Qwen/Qwen2.5-Coder-7B-Instruct
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
# DATA_NAMES="aime24"
DATA_NAMES="gsm8k,math500,minerva_math,olympiadbench,aime24,amc23"
SPLIT="test"
NUM_TEST_SAMPLE=-1

for MODEL_NAME_OR_PATH in "${MODEL_NAMES[@]}"; do
    echo "Evaluating model: ${MODEL_NAME_OR_PATH}"
    OUTPUT_DIR=results/${MODEL_NAME_OR_PATH}
    
    # single-gpu
    TOKENIZERS_PARALLELISM=false \
    python3 -u math_eval.py \
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
        --use_vllm \
        --max_func_call 4 \
        --overwrite \
        2>&1 | tee "logs_${MODEL_NAME_OR_PATH//\//_}.log"
done


# multi-gpu
# python3 scripts/run_eval_multi_gpus.py \
#     --model_name_or_path $MODEL_NAME_OR_PATH \
#     --output_dir $OUTPUT_DIR \
#     --data_names ${DATA_NAMES} \
#     --prompt_type "cot" \
#     --temperature 0 \
#     --use_vllm \
#     --save_outputs \
#     --available_gpus 0,1,2,3,4,5,6,7 \
#     --gpus_per_model 1 \
#     --overwrite
