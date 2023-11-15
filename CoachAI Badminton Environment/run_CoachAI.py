from CoachAI import badminton_env
from ppo import PPO

# Users' model
agent = PPO()
agent.isTrain = True
agent.load(f'./ppo/data_train/PPO/PPO_1000000.pth')

episodes = 1000
env = badminton_env(episodes, side=1, opponent="Anthony Sinisuka GINTING")

for episode in range(1, episodes+1):
    states, launch = env.reset()
    done = False

    while not done :
        action = agent.train(states, launch)
        states, reward, done, launch = env.step(action, launch)

        print(f"State: {states}")
        print(f"Action: {action}")

env.close()