# agent.py
import torch
import numpy as np
from soccer_twos import AgentInterface
from gym_unity.envs import ActionFlattener
import os

from .model import ActorCritic
CHECKPOINT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "ppo_checkpoint_random_12000.pth"
)

class PPOAgent(AgentInterface):
    def __init__(self, env):
        super().__init__()
        self.name = "PPO"
        self.flattener = ActionFlattener(env.action_space.nvec)

        obs_dim = env.observation_space.shape[0]
        act_dim = self.flattener.action_space.n

        self.model = ActorCritic(obs_dim, act_dim)
        self.model.load_state_dict(torch.load(CHECKPOINT_PATH))
        self.model.eval()

    def act(self, observation):
        actions = {}

        for player_id in observation:
            state = torch.tensor(observation[player_id], dtype=torch.float32).unsqueeze(0)
            logits, _ = self.model(state)
            probs = torch.softmax(logits, dim=-1)

            action = torch.argmax(probs, dim=-1).item()
            actions[player_id] = self.flattener.lookup_action(action)

        return actions