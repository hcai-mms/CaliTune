import torch

if __name__ == "__main__":
    lams = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
