# This does not yet work, but I'm trying to make a DQN that can handle continuous actions

import gym, torch
from doubleDQNContinuous import Agent

class MountainCarContinuous:
    def __init__(self):
        self.env = gym.make('MountainCarContinuous-v0')
        
        self.observation, self.info = self.env.reset(seed=42)
        self.num_observations = self.env.observation_space.shape[0]

    def step(self, action):
        self.observation, self.reward, self.terminated, self.truncated, self.info = self.env.step(action)
        return self.observation, self.reward, self.terminated, self.truncated, self.info

    def destroyEnv(self):
        self.env.close()
    
    def reset(self):
        self.observation, self.info = self.env.reset(seed=42)
        return self.observation
    
    def render(self):
        self.env.render()
    

if __name__ == "__main__":
    env = MountainCarContinuous()
    agent = Agent(env.num_observations, lr=5e-7)
    epsilon = 1.0
    epsilon_decay = 0.999
    epsilon_min = 0.01
    gamma = 0.99
    batch_size = 128
    num_episodes = 1000
    for i in range(num_episodes):
        # if (i+1) % 10 == 0:
        #     env.env = gym.make('MountainCarContinuous-v0', render_mode="human")
            
        # else:
        #     env.env = gym.make('MountainCarContinuous-v0')
        env.env = gym.make('MountainCarContinuous-v0', render_mode="human")
            
        
        state = torch.Tensor(env.reset())
        done = False
        score = 0
        while not done:
            action = agent.get_action(state, epsilon)
            action = 2 * action

            nS, reward, done, truncated,  info = env.step(action)
            score += reward
            next_state = torch.Tensor(nS)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if epsilon > epsilon_min:
                epsilon *= epsilon_decay
            agent.replay(batch_size, gamma)
        print(f"Episode: {i}, Epsilon: {epsilon}, Score: {score}")
        print(nS, reward, done, truncated,  info)
        if i % 10 == 0:
            agent.update_target_model()
            
    env.destroyEnv()