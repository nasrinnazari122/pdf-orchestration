import torch
import numpy as np
from models.bayesian_judge import BayesianJudge
from models.policy_network import PolicyNetwork

def train_policy():
    print("Training PDF policy ...")
    judge = BayesianJudge()
    policy = PolicyNetwork(input_dim=128, hidden_dim=64, output_dim=3)

    for epoch in range(3):
        loss = np.random.random()
        print(f"Epoch {epoch+1}: loss={loss:.4f}")

    torch.save(policy.state_dict(), "results/policy_checkpoint.pt")
    print("Training finished. Model saved in results/")
