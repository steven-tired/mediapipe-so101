#!/usr/bin/env bash
set -euo pipefail

profile="${1:-smoke}"

case "$profile" in
  smoke)
    steps=20
    save_freq=20
    log_freq=1
    ;;
  full)
    steps=50000
    save_freq=5000
    log_freq=100
    ;;
  resume80)
    steps=80000
    save_freq=5000
    log_freq=100
    ;;
  *)
    echo "Usage: $0 {smoke|full|resume80}" >&2
    exit 2
    ;;
esac

workspace=/home/zhuokai/hand-teleop
run_root="$workspace/training/phase_c_act"
dataset_root="$workspace/datasets/hand_tracking_pv_carton_phase_b"
train_cli="$workspace/.venv-lerobot/bin/lerobot-train"
batch_size="${ACT_BATCH_SIZE:-8}"
timestamp="$(date +%Y%m%d_%H%M%S)"
run_name="${ACT_RUN_NAME:-act_phase_c_${profile}_${timestamp}}"
output_dir="$run_root/outputs/$run_name"
log_path="$run_root/logs/$run_name.log"

mkdir -p "$run_root/outputs" "$run_root/logs" "$run_root/.cache/huggingface/datasets"

export HF_HOME="$run_root/.cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export PYTHONUNBUFFERED=1

echo "profile=$profile"
echo "dataset=$dataset_root"
echo "output_dir=$output_dir"
echo "log=$log_path"
echo "batch_size=$batch_size steps=$steps chunk_size=20 n_action_steps=1 temporal_ensemble=off"

cd "$workspace"
if [[ "$profile" == "resume80" ]]; then
  resume_config="$run_root/outputs/act_phase_c_full_20260822_131930/checkpoints/040000/pretrained_model/train_config.json"
  log_path="$run_root/logs/act_phase_c_resume_40k_to_80k_${timestamp}.log"
  echo "resume_config=$resume_config"
  echo "log=$log_path"
  env -u PYTHONPATH "$train_cli" \
    --config_path="$resume_config" \
    --resume=true \
    --steps="$steps" \
    --save_freq="$save_freq" \
    --log_freq="$log_freq" \
    2>&1 | tee "$log_path"
  exit "${PIPESTATUS[0]}"
fi

env -u PYTHONPATH "$train_cli" \
  --policy.type=act \
  --dataset.repo_id=stevenzenith/hand_tracking_pv_carton_phase_b \
  --dataset.root="$dataset_root" \
  --dataset.video_backend=torchcodec \
  --batch_size="$batch_size" \
  --num_workers=4 \
  --policy.chunk_size=20 \
  --policy.n_action_steps=1 \
  --policy.device=cuda \
  --policy.use_amp=true \
  --policy.push_to_hub=false \
  --steps="$steps" \
  --save_freq="$save_freq" \
  --log_freq="$log_freq" \
  --output_dir="$output_dir" \
  --job_name="act_phase_c_${profile}" \
  --wandb.enable=false \
  2>&1 | tee "$log_path"
