# trainer.py

import numpy as np
import ray
import wandb

import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from mcts import MCTS
from config import SELF_PLAY_GAMES, BATCH_SIZE, EPOCHS, LEARNING_RATE, NUM_ACTOR
from util import load_latest_model, save_model

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

def train(training_data):
    print("--- Starting Training ---")
    model, device = load_latest_model()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion_policy = torch.nn.KLDivLoss(reduction='batchmean')
    criterion_value = torch.nn.MSELoss()
    wandb.require("core")
    wandb.init(
        project="MCTS_TicTacToe",
        name=f"Gen",
        reinit=True
    )

    """ 수집된 데이터로 모델 훈련 """
    model.train()
    
    states, policies, values = zip(*training_data)
    states = torch.FloatTensor(np.array(states)).to(device)
    policies = torch.FloatTensor(np.array(policies)).to(device)
    values = torch.FloatTensor(np.array(values)).view(-1, 1).to(device)

    dataset = TensorDataset(states, policies, values)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_states, batch_policies, batch_values in dataloader:
            optimizer.zero_grad()
            
            out_policy, out_value = model(batch_states)
            loss_policy = criterion_policy(out_policy, batch_policies)
            loss_value = criterion_value(out_value, batch_values)
            loss = loss_policy + loss_value
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loss_policy += loss_policy.item()
            loss_value += loss_value.item()
        wandb.log({"epoch": (epoch+1)/EPOCHS, "loss": total_loss / len(dataloader)})
        
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss_policy / len(dataloader)} {loss_value / len(dataloader)} {total_loss / len(dataloader):.4f}")
    save_model(model)

    wandb.finish()
