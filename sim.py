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
noise = 2.0
decay = 0.99
noise_min = 0.05

env = gym.make('MountainCarContinuous-v0').unwrapped
agt = dpg.dpg()

file = open('result.txt', 'w+')
file.write(f'episode, step, return, noise' + '\n')
file.close()

for episode in range(1000):
    R = np.array([0.0])
    s = env.reset()
    for step in range(10000):
        a = agt([s])
        a = np.array([a.item()])
        a += np.random.randn(1) * noise
        a = np.clip(a, -1.0, 1.0)

        s1, r, done, _ = env.step(a)

        r = np.array([r])
        transition = Transition(s, a, r, s1)
        rb.push(transition)

        buffer_size = len(rb)
        if buffer_size >= rb.capacity and step % 1000 == 0:
            print(f'training ----------------')
            agt.train(rb.sample(batch_size=100))

        if (episode+1) % 10 == 0 and step < 1000:
            env.render()

        s = s1.copy()
        R += r

        if(verbose): print(transition)
        if(step % 100 == 0):
            print(f'episode = {episode+1}, step = {step}, return = {R.item():.3f}, noise = {noise:.3f}')
            writer.add_scalars('status', {'step': step, 'return': R.item(), 'noise': noise}, episode)

        if done:
            break

    file = open('result.txt', 'a')
    file.write(f'{episode+1}, {step}, {R.item():.3f}, {noise:.3f}' + '\n')
    file.close()
    noise = noise*decay if noise > noise_min else noise_min

env.close()
writer.close()

