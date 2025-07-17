import torch

from config import MODEL_PATH
from model import SimpleNN


def load_latest_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleNN().to(device)

    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device,weights_only=True))
        print(f"Model loaded successfully.{MODEL_PATH}")
    except FileNotFoundError:
        print(f"Model file not found at {MODEL_PATH}. Playing with an untrained model.")
    return model, device

def save_model(model):
    print("Model saved.")    
    torch.save(model.state_dict(), MODEL_PATH)