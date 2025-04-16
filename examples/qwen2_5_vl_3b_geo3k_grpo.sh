set -x

export HF_HOME=/ocean/projects/cis210027p/qwang20/EasyR1/hf_home
export HF_HUB_CACHE=/ocean/projects/cis210027p/qwang20/EasyR1/hf_hub_cache

MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct  # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/config_lora.yaml \
    data.train_files=hiyouga/geometry3k@train \
    data.val_files=hiyouga/geometry3k@test \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    trainer.experiment_name=qwen2_5_vl_3b_geo_grpo_3step_ft \
    trainer.n_gpus_per_node=1
