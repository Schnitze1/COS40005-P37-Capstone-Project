"""
BRAC+ / BDB Discriminator-Based Constraint Training Module
Implements a binary discriminator D_w(s, a) to estimate the latent density ratio
log (p_data(s, a) / p_policy(s, a)) for OOD action regularisation in BEAR/BRAC.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class StateActionDiscriminator(nn.Module):
    def __init__(self, observation_shape, action_size, hidden_dim=256):
        super().__init__()
        obs_dim = observation_shape[0] if isinstance(observation_shape, tuple) else observation_shape
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs, action):
        inputs = torch.cat([obs, action], dim=-1)
        logits = self.net(inputs)
        return logits

    def compute_loss(self, real_obs, real_act, fake_obs, fake_act):
        real_logits = self.forward(real_obs, real_act)
        fake_logits = self.forward(fake_obs, fake_act)
        
        real_loss = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
        fake_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
        
        return 0.5 * (real_loss + fake_loss)

    def compute_density_ratio_penalty(self, obs, action):
        logits = self.forward(obs, action)
        # log (D / (1 - D)) = logits directly (log-odds ratio)
        penalty = F.softplus(-logits) # Higher penalty when action is estimated OOD
        return penalty

if __name__ == "__main__":
    print("BRAC+ State-Action Discriminator module created successfully.")
