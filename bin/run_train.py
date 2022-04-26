from omegaconf import OmegaConf
from src.train import train_net

if __name__ == "__main__":
    config = OmegaConf.load("bin/config/train.yml")
    train_net(config)
