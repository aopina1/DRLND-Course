import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.eps = 1
        self.alpha = 0.02
        self.episodes = 0
        self.Q = defaultdict(lambda: np.zeros(self.nA))

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        if state in self.Q.keys():
            greedy_action = self.Q[state].argmax()
            probs = self.eps * np.ones(self.nA)/self.nA
            probs[greedy_action] = 1 - self.eps + (1/self.nA)*self.eps
            
            return np.random.choice(np.array(self.nA), p=probs)
        else:
            return np.random.choice(self.nA)

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        
        self.Q[state][action] += self.alpha*(reward + self.Q[next_state].max() - 
                                                      self.Q[state][action])
        
        if done:
            self.eps = max(self.eps*0.999, 0.01)
            self.episodes += 1
            if self.episodes%10000 == 0:
                self.eps = 1