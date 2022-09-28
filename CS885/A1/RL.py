import numpy as np
import MDP

class RL:
    def __init__(self,mdp,sampleReward):
        '''Constructor for the RL class

        Inputs:
        mdp -- Markov decision process (T, R, discount)
        sampleReward -- Function to sample rewards (e.g., bernoulli, Gaussian).
        This function takes one argument: the mean of the distributon and 
        returns a sample from the distribution.
        '''

        self.mdp = mdp
        self.sampleReward = sampleReward

    def sampleRewardAndNextState(self,state,action):
        '''Procedure to sample a reward and the next state
        reward ~ Pr(r)
        nextState ~ Pr(s'|s,a)

        Inputs:
        state -- current state
        action -- action to be executed

        Outputs: 
        reward -- sampled reward
        nextState -- sampled next state
        '''

        reward = self.sampleReward(self.mdp.R[action,state])
        cumProb = np.cumsum(self.mdp.T[action,state,:])
        nextState = np.where(cumProb >= np.random.rand(1))[0][0]
        return [reward,nextState]

    def qLearning(self,s0,initialQ,nEpisodes,nSteps,epsilon=0,temperature=0.0):
        '''qLearning algorithm.  
        When epsilon > 0: perform epsilon exploration (i.e., with probability epsilon, select action at random )
        When epsilon == 0 and temperature > 0: perform Boltzmann exploration with temperature parameter
        When epsilon == 0 and temperature == 0: no exploration (i.e., selection action with best Q-value)

        Inputs:
        s0 -- initial state
        initialQ -- initial Q function (|A|x|S| array)
        nEpisodes -- # of episodes (one episode consists of a trajectory of nSteps that starts in s0
        nSteps -- # of steps per episode
        epsilon -- probability with which an action is chosen at random
        temperature -- parameter that regulates Boltzmann exploration

        Outputs: 
        Q -- final Q function (|A|x|S| array)
        policy -- final policy
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        avg_cum_rewards = np.zeros(nEpisodes)
        trials = 100
        for trial in range(trials):
            Q = np.copy(initialQ)
            N = np.zeros([self.mdp.nActions, self.mdp.nStates])
            cum_rewards = np.zeros(nEpisodes)
            for episode in range(nEpisodes):
                s = s0
                prev_Q = np.copy(Q)
                for step in range(nSteps):
                    if epsilon > 0 and np.random.rand(1) < epsilon:
                        a = np.random.randint(self.mdp.nActions)
                    elif epsilon == 0 and temperature > 0:
                        pr = np.exp(Q[:, s] / temperature)
                        pr = pr / np.sum(pr)
                        cumProb = np.cumsum(pr)
                        a = np.where(cumProb >= np.random.rand(1))[0][0]
                    else:
                        a = np.argmax(Q[:, s])
                    r, s1 = self.sampleRewardAndNextState(s, a)
                    N[a, s] += 1
                    alpha = 1.0 / N[a, s]
                    v1 = Q[a, s]
                    Q[a, s] = Q[a, s] + alpha * (r + self.mdp.discount * np.max(Q[:, s1]) - Q[a, s])
                    v2 = Q[a, s]
                    # print(v2 - v1, alpha)
                    s = s1
                    cum_rewards[episode] += pow(self.mdp.discount, step) * r
                # print(np.max(np.abs(Q - prev_Q)))
            avg_cum_rewards += cum_rewards
        avg_cum_rewards = avg_cum_rewards / trials
        policy = np.argmax(Q, axis=0)
        return [Q,policy,avg_cum_rewards]    