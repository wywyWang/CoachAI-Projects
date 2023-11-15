from RLEnvironment import Env

env = Env()
env.reset()

done = False
while not done:
    action = env.action_space.sample()

    next_state, reward = env.step(action)

    # done when reward is -1, means the other player win this score
    done = reward != -1 