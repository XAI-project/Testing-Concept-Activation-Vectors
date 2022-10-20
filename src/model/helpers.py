from src.CONSTS import *
import torch


def save_model(model, path=MODEL_PATH):
    """
    Save pytorch model to a specified file
    """
    torch.save(model, path)


def load_model(path=MODEL_PATH):
    """
    Load a pytorch model.
    """
    model = torch.load(path)
    model.eval()
    return model
