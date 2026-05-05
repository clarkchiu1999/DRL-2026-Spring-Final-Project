import soccer_twos
from gym_unity.envs import ActionFlattener

env = soccer_twos.make(
    render=True,
    flatten_branched=True,  # converts MultiDiscrete into Discrete action space
    variation=soccer_twos.EnvType.team_vs_policy,
    single_player=True,  # controls a single player while the other stays still
    opponent_policy=lambda *_: 0,  # opponents stay still
    worker = 2
)
# flattener = ActionFlattener(env.action_space.nvec)
print("Observation Space: ", env.observation_space.shape)
print("Action Space: ", env.action_space)

team0_reward = 0
env.reset()
print(env.reset().shape)
while True:
    act = env.action_space.sample()
    # print(type(act))
    obs, reward, done, info = env.step(act)
    team0_reward += reward
    if done:  # if any agent is done
        print("Total Reward: ", team0_reward)
        print(info)
        team0_reward = 0
        env.reset()
