#!/usr/bin/env bash
set -euo pipefail
#source .venv/bin/activate

# --- GPU cleanup: kill only processes owned by user 'prachh' ---
echo "[cleanup] scanning GPU PIDs owned by prachh ..."
GPU_PIDS="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | sort -u || true)"

if [[ -n "${GPU_PIDS}" ]]; then
  # graceful stop first
  for pid in ${GPU_PIDS}; do
    owner="$(ps -o user= -p "${pid}" 2>/dev/null || true)"
    if [[ "${owner}" == "prachh" ]]; then
      echo "[cleanup] SIGTERM -> ${pid}"
      kill -TERM "${pid}" 2>/dev/null || true
    fi
  done
  sleep 3
  # force kill leftovers
  for pid in ${GPU_PIDS}; do
    owner="$(ps -o user= -p "${pid}" 2>/dev/null || true)"
    if [[ "${owner}" == "prachh" ]] && kill -0 "${pid}" 2>/dev/null; then
      echo "[cleanup] SIGKILL -> ${pid}"
      kill -9 "${pid}" 2>/dev/null || true
    fi
  done
else
  echo "[cleanup] no GPU processes found."
fi
# --- end cleanup ---



# Optional: quick preflight to ensure LoRA targets match this base model.
# Set PRECHECK=0 to skip. This loads the base model once to count trainables.
: "${PRECHECK:=1}"
if [[ "${PRECHECK}" == "1" ]]; then
  python - <<'PY' || { echo "[precheck] Failed. Aborting."; exit 1; }
from transformers import Qwen2_5_VLForConditionalGeneration
from peft import LoraConfig, get_peft_model
def count_trainable(m):
  return sum(p.numel() for p in m.parameters() if p.requires_grad)
base="Qwen/Qwen2.5-VL-7B-Instruct"
targets=["q_proj","v_proj","o_proj"]
m = Qwen2_5_VLForConditionalGeneration.from_pretrained(base)
pm = get_peft_model(m, LoraConfig(r=16,lora_alpha=32,lora_dropout=0.05,
                                  target_modules=targets, bias="none", task_type="CAUSAL_LM"))
n = count_trainable(pm)
print(f"[precheck] target_modules={targets} trainable_params={n}")
assert n > 0, "No trainable params with selected target_modules; adjust targets."
PY
fi

# Set environment variables for CUDA and distributed training
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=WARN
export NCCL_NVLS_ENABLE=0
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export TORCH_DISTRIBUTED_DEBUG=OFF
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export PYTHONWARNINGS="ignore"
export TF_CPP_MIN_LOG_LEVEL=3

# Run the training script with DeepSpeed using config file
deepspeed --num_gpus=8 src/train_vlm_sft.py \
    --base_model Qwen/Qwen2.5-VL-7B-Instruct \
    --sft_config configs/sft_config_rank64.json \
    --output_dir outputs/qwen2_5_vl_7b_lora_rank64_e6_again_4 \
    --max_train 50000 \
    --max_eval 2000 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --target_modules v_proj,up_proj,o_proj,down_proj,q_proj,gate_proj,k_proj \
    --flash_attn
