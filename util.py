import torch

from model import SimpleNN


def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleNN().to(device)

    try:
        model.load_state_dict(torch.load(model_path, map_location=device,weights_only=True))
        print(f"Model loaded successfully.{model_path}")
    except FileNotFoundError:
        print(f"Model file not found at {model_path}. Playing with an untrained model.")
    return model, device

def save_model(model, model_path):
    print("Model saved.")    
    torch.save(model.state_dict(), model_path)