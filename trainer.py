# trainer.py


import os

import ray
from ray import train
from ray.air import session
from ray.train import Checkpoint
import tempfile

import wandb

import torch
import torch.optim as optim
from tqdm import tqdm

from model import SimpleNN
from mcts import MCTS
from config import SELF_PLAY_GAMES, BATCH_SIZE, EPOCHS, LEARNING_RATE, NUM_ACTOR, CHECKPOINT_MODEL
from util import load_model, save_model

def collect_data():    
    print("--- Collecting training data through self-play ---")
    self_play_game_per_actor = SELF_PLAY_GAMES // NUM_ACTOR
    
    actors = [MCTS.remote() for i in range(NUM_ACTOR)]
    all_training_data = []

    results_referece = []
    for actor in actors:
        for _ in range(self_play_game_per_actor):
            results_referece.append(actor.self_play.remote())

    for _ in tqdm(range(len(results_referece))):
        done, results_referece = ray.wait(results_referece, num_returns=1)
        game_data = ray.get(done[0])
        all_training_data.extend(game_data)

    return all_training_data

def train_loop_per_worker(config: dict):
    print("--- Starting Training ---")
    checkpoint = config.get("resume_from_checkpoint")
    if checkpoint:        
        with checkpoint.as_directory() as ckpt_dir:
            path = os.path.join(ckpt_dir, CHECKPOINT_MODEL)
            model, device = load_model(path)
    else :
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SimpleNN().to(device) 
    model = train.torch.prepare_model(model)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion_policy = torch.nn.KLDivLoss(reduction='batchmean')
    criterion_value = torch.nn.MSELoss()
    
    worker_id = session.get_world_rank()
    if worker_id == 0:
        wandb.require("core")
        wandb.init(
            project="MCTS_TicTacToe",
            name=f"{session.get_experiment_name()}",
            reinit=True
        )

    """ 수집된 데이터로 모델 훈련 """
    model.train()
    dataset = session.get_dataset_shard("train")

    for epoch in range(EPOCHS):
        total_loss, total_policy_loss, total_value_loss = 0, 0, 0
        count = 0
        for batch in dataset.iter_torch_batches(batch_size=BATCH_SIZE):
            states = batch["state"].to(device)
            policies = batch["policy"].to(device)
            values = batch["value"].view(-1, 1).to(device)
            
            optimizer.zero_grad()
            
            out_policy, out_value = model(states)
            policy_loss = criterion_policy(out_policy, policies)
            value_loss = criterion_value(out_value, values)
            loss = policy_loss + value_loss
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            count += 1

        if worker_id == 0:
            wandb.log({"epoch": (epoch+1)/EPOCHS, "loss": total_loss / count, "policy_loss" : total_policy_loss / count, "value_loss" : total_value_loss / count})
   
        with tempfile.TemporaryDirectory() as tmpdir:    
            model_from_ray = getattr(model, "module", model)
            temp_path = os.path.join(tmpdir, CHECKPOINT_MODEL)
            save_model(model_from_ray, temp_path)
            checkpoint = Checkpoint.from_directory(tmpdir)
            if (epoch + 1) % 5 == 0:
                train.report(
                    metrics={
                        "epoch": epoch,
                        "loss": total_loss / count,
                        "policy_loss": total_policy_loss / count,
                        "value_loss": total_value_loss / count
                    },
                    checkpoint=checkpoint
                )
            else:
                train.report(
                    metrics={
                        "epoch": epoch,
                        "loss": total_loss / count,
                        "policy_loss": total_policy_loss / count,
                        "value_loss": total_value_loss / count
                    },
                )

    if worker_id == 0:
        wandb.finish()
