import gym
import numpy as np
import optim
from tensor import Tensor 
import collections
import random
# Hyperparameters
LR         = 0.0005
GAMMA      = 0.98
batch_size = 128
EPSILON    = 0.9
EPS_MIN    = 0.05
EPS_DECAY  = 1000
TAU        = 0.005

#env = gym.make("CartPole-v1",render_mode='human')
env = gym.make("CartPole-v1")
env = gym.wrappers.RecordEpisodeStatistics(env)

class Model:
    def __init__(self):
        self.fc1 = Tensor.uniform(4,   128)
        self.fc2 = Tensor.uniform(128, 128)
        self.fc3 = Tensor.uniform(128,   2)

    def forward(self, x):
        x = x.matmul(self.fc1).relu()
        x = x.matmul(self.fc2).relu()
        x = x.matmul(self.fc3)
        return x
    def __call__(self,x): return self.forward(x)

    # epsilon greedy
    def sample_action(self, obs, eps_threshold):
        if np.random.random() > eps_threshold:
            return self.forward(obs).numpy().argmax()
        else:  return env.action_space.sample()

def train(policy, target, optim, buffer, bs):
    if len(buffer) < bs:
        return
    batch = random.sample(buffer, bs)
    states  = Tensor([x[0] for x in batch])
    actions = Tensor([[x[1]] for x in batch])
    rewards = Tensor([[x[2]] for x in batch])
    nstates = Tensor([x[3] for x in batch])
    dones_mask = Tensor([x[4] for x in batch])

    q_pred = policy(states)  #.gather(1, actions)
    q_targ = target(nstates) #.max(1)[0].unsqueeze(1)
    #q_targ[dones_mask] = 0.0  # set all terminal states' value to zero
    q_targ = (rewards + GAMMA ) * q_targ * done_mask
    loss = (0.5*(q_pred - q_targ) * (q_pred - q_targ)).mean()
    print(loss)

    optim.zero_grad()
    loss.backward()
    optim.step()
    pass

policy = Model()
target = Model()
optim  = optim.SGD([policy.fc1, policy.fc2, policy.fc3], lr=LR)
ReplayBuffer = collections.deque(maxlen=50000)

scores = []
for epi in range(100):
    obs, _ = env.reset()
    EPSILON = max(EPS_MIN, EPSILON*EPS_DECAY)

    while True:
        # Apply policy and observe response
        action = policy.sample_action(Tensor(obs), EPSILON)
        n_obs, rew, terminated, truncated, info = env.step( action )

        # Store transitions
        done_mask = 0.0 if (terminated or truncated) else 1.0
        ReplayBuffer.append((obs,action,rew/100.0,n_obs, done_mask))

        # Train model
        train(policy, target, optim, ReplayBuffer, bs=1)

        obs = n_obs
        if (terminated or truncated):
            scores.append( info['episode']['r'] )
            print(f"Episode {epi} return {scores[-1]}")
            break
            
