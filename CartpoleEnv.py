import gym, torch
from doubleDQN import Agent


class CartPole:
    """
    The goal of this game is to make the pole stay up by moving the cart left or right.
    The game ends when the pole falls over or the cart moves out of bounds.

    set render mode to "human" to see the game in action
    set render mode to None (not a string) to run the game in the background (best for training)
    """
    def __init__(self):
        self.env = gym.make('CartPole-v1', render_mode="human")
        
        self.observation, self.info = self.env.reset(seed=42)
        self.num_actions = self.env.action_space.n
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
    env = CartPole()
    agent = Agent(env.num_observations, env.num_actions)
    epsilon = 1.0
    epsilon_decay = 0.999
    epsilon_min = 0.01
    gamma = 0.99
    batch_size = 128
    num_episodes = 1000
    for i in range(num_episodes):
        state = torch.Tensor(env.reset())
        done = False
        score = 0
        while not done:
            action = int(agent.get_action(state, epsilon))
            nS, reward, done, truncated,  info = env.step(action)
            score += reward
            next_state = torch.Tensor(nS)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if epsilon > epsilon_min:
                epsilon *= epsilon_decay
            agent.replay(batch_size, gamma)
        if i % 10 == 0:
            agent.update_target_model()
            print(f"Episode: {i}, Epsilon: {epsilon}, Score: {score}")
            print(nS, reward, done, truncated,  info)
    env.destroyEnv()