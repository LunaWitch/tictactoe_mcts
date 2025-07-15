# main.py

import torch
from model import SimpleNN
from trainer import train, collect_data
from play import play_game
from config import GENERATION_SIZE, MODEL_PATH

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleNN().to(device)

    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device,weights_only=True))
        print("Model loaded successfully.")
    except FileNotFoundError:
        print(f"Model file not found at {MODEL_PATH}. Playing with an untrained model.")

    for i in range(GENERATION_SIZE):
        all_training_data = collect_data(model, device)
        train(model, device, all_training_data, i)

    print("Training finished and model saved.")
    torch.save(model.state_dict(), MODEL_PATH)

    print("--- Starting Game ---")
    play_game()


if __name__ == "__main__":
    main()