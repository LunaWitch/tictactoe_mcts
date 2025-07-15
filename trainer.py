# trainer.py

import numpy as np
import wandb

import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from mcts import MCTS
from config import SELF_PLAY_GAMES, BATCH_SIZE, EPOCHS, LEARNING_RATE

def collect_data(model, device):    
    print("--- Collecting training data through self-play ---")
    
    mcts = MCTS(model, device)
    all_training_data = []
    for _ in tqdm(range(SELF_PLAY_GAMES), desc="Self Playing"):
        all_training_data.extend(mcts.self_play())
    return all_training_data

def train(model, device, training_data, cnt):
    print("--- Starting Training ---")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion_policy = torch.nn.KLDivLoss(reduction='batchmean')
    criterion_value = torch.nn.MSELoss()
    wandb.require("core")
    wandb.init(
        project="MCTS_TicTacToe",
        name=f"{cnt}",
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
    
    torch.save(model.state_dict(), f"model_gen_{cnt+1}.pth")
    wandb.finish()
