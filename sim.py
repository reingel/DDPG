import numpy as np
import gym
import dpg
from replaybuffer import Transition
from replaybuffer import rb
from tensorboardX import SummaryWriter
from summary_dir import summary_dir

logdir = summary_dir()
writer = SummaryWriter(logdir=logdir)
print(f"Data will be logged in '{logdir}'")

verbose = False

env = gym.make('MountainCarContinuous-v0').unwrapped
# env = gym.make('Pendulum-v0').unwrapped

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
max_episode = 10000
max_step = 6000
solved_reward = 90

agt = dpg.dpg(state_dim, action_dim, max_action, max_episode, max_step)

file = open('result.txt', 'w+')
file.write(f'episode, step, return, noise' + '\n')
file.close()

for episode in range(max_episode):
    R = np.array([0.0])
    s = env.reset()
    # agt.gen_noise(episode)
    noise = np.random.randn(max_step) * np.exp(-1.0 * episode / max_episode)
    for step in range(max_step):
        a = agt(s, step)
        a = np.array([a.item() + noise[step]])

        s1, r, done, _ = env.step(a)

        r = np.array([r])
        transition = Transition(s, a, r, s1, done)
        rb.push(transition)

        buffer_size = len(rb)
        if buffer_size >= rb.capacity and step % 1000 == 0:
            agt.train(rb.sample(batch_size=100))

        if (episode+1) % 100 == 0 and step < 500:
            env.render()

        s = s1.copy()
        R += r

        # if(verbose): print(transition)
        # if(step % 100 == 0):
        #     print(f'episode = {episode+1}, step = {step}, return = {R.item():.3f}')

        if done:
            break

    writer.add_scalar('status/step', step, episode)
    writer.add_scalar('status/return', R.item(), episode)
    
    print(f'{episode+1}, {step}, {R.item():.3f}')

    file = open('result.txt', 'a')
    file.write(f'{episode+1}, {step}, {R.item():.3f}' + '\n')
    file.close()

env.close()
writer.close()

