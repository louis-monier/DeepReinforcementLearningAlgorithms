import numpy as np
from collections import deque


class ReplayBuffer:
    def __init__(self, replay_size):
        self.replay_buffer = deque(maxlen=int(replay_size))

    def __len__(self):
        return len(self.replay_buffer)

    def store_transition(self, transition):
        return self.replay_buffer.append(transition)

    def sample_buffer(self, batch_size):
        # draw batch_size indices from the replay_buffer
        samples_indices = np.random.choice(
            len(self.replay_buffer), batch_size, replace=False
        )

        # draw samples
        states, actions, rewards, dones, next_states = zip(
            *[self.replay_buffer[inds] for inds in samples_indices]
        )

        # formatting
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.uint8)
        next_states = np.array(next_states, dtype=np.float32)

        return states, actions, rewards, dones, next_states
