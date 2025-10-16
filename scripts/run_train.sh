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


export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=WARN
export TORCH_DISTRIBUTED_DEBUG=OFF
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export PYTHONWARNINGS="ignore"
export TF_CPP_MIN_LOG_LEVEL=3

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

deepspeed --num_gpus=8 src/train_vlm_sft.py \
  --base_model Qwen/Qwen2.5-VL-7B-Instruct \
  --sft_config configs/sft_config_1epoch.json \
  --output_dir outputs/qwen2_5_vl_7b_sft_chart_text_oct15_e1 \
  --max_train 80000 \
  --max_eval 2000 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --target_modules q_proj,v_proj,o_proj \
  --flash_attn

deepspeed --num_gpus=8 src/train_vlm_sft.py \
  --base_model Qwen/Qwen2.5-VL-7B-Instruct \
  --sft_config configs/sft_config.json \
  --output_dir outputs/qwen2_5_vl_7b_sft_chart_text_oct15_e2 \
  --max_train 80000 \
  --max_eval 2000 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --target_modules q_proj,v_proj,o_proj \
  --flash_attn 