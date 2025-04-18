#!/bin/bash
set -x

export HF_HOME=/ocean/projects/cis210027p/qwang20/EasyR1/hf_home
export HF_HUB_CACHE=/ocean/projects/cis210027p/qwang20/EasyR1/hf_hub_cache

MODEL_PATH=qingzhengwang/qwen2_5_vl_3b_full_sft_geoqa_stepbystep_fix # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/config_full_bs_64_geoqa_stepbystep.yaml \
    data.train_files=LoadingBFX/GeoQA-cot-stepbystep@train \
    data.val_files=LoadingBFX/GeoQA-cot-stepbystep@test \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    trainer.experiment_name=qwen2_5_vl_3b_geo_grpo_3step_ft_geoqa_cot_3step_fix_reward \
    trainer.n_gpus_per_node=1 \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16
