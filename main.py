# main.py
import os
import numpy as np
import ray
from ray.data import from_items
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig
from ray.train import RunConfig
from ray.train import CheckpointConfig

from trainer import train_loop_per_worker, collect_data
from play import play_game
from config import CHECKPOINT_MODEL, GENERATION_SIZE, LATEST_MODEL, RAY_RESULT_PATH
from util import load_model, save_model

def main():
    if ray.is_initialized():
        ray.shutdown()
    ray.init()

    prev_checkpoint = None 
    for gen in range(GENERATION_SIZE):
        all_training_data = collect_data()
        dataset = from_items([
            {
                "state": np.array(state, dtype=np.float32),
                "policy": np.array(policy, dtype=np.float32),
                "value": np.array(value, dtype=np.float32),
            }
            for state, policy, value in all_training_data
        ])
        trainer = TorchTrainer(
            train_loop_per_worker = train_loop_per_worker,
            train_loop_config={"resume_from_checkpoint": prev_checkpoint},
            datasets = {"train": dataset},
            scaling_config = ScalingConfig(use_gpu=True, num_workers=4, resources_per_worker={"GPU": 1}),
            run_config = RunConfig(
                name=f"train_gen_{gen + 1}",
                storage_path = os.path.join(os.getcwd(), RAY_RESULT_PATH),
                checkpoint_config=CheckpointConfig(
                    num_to_keep = 5,
                    checkpoint_score_attribute = "loss",
                    checkpoint_score_order = "min",
                ),
            ),
        )
        
        result = trainer.fit()

        if result.best_checkpoints:
            best_checkpoint = result.best_checkpoints[0][0]
            with best_checkpoint.as_directory() as checkpoint_dir:
                model_path_in_checkpoint = os.path.join(checkpoint_dir, CHECKPOINT_MODEL)
                if os.path.exists(model_path_in_checkpoint):
                    model, _ = load_model(model_path_in_checkpoint)
                    save_model(model, LATEST_MODEL)
                    print(f"Training finished. New model saved to {LATEST_MODEL}")
                else:
                    print(f"ERROR: Model file not found in checkpoint: {model_path_in_checkpoint}")
            prev_checkpoint = best_checkpoint
    ray.shutdown()

    print("--- Starting Game ---")
    play_game()


if __name__ == "__main__":
    main()