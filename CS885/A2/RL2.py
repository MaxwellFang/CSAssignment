from socket import NI_NAMEREQD
import numpy as np
import MDP

class RL2:
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

    def epsilonGreedyBandit(self,nIterations):
        '''Epsilon greedy algorithm for bandits (assume no discount factor).  Use epsilon = 1 / # of iterations.

        Inputs:
        nIterations -- # of arms that are pulled

        Outputs: 
        empiricalMeans -- empirical average of rewards for each arm (array of |A| entries)
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        empiricalMeans = np.zeros(self.mdp.nActions)
        actionCounts = np.zeros(self.mdp.nActions)
        rewards = np.zeros(nIterations)
        for i in range(nIterations):
            epsilon = 1.0 / (i + 1)
            if np.random.randn(1) < epsilon:
                a = np.random.randint(self.mdp.nActions)
            else:
                a = np.argmax(empiricalMeans, axis=0)
            r = self.sampleRewardAndNextState(0, a)[0]
            rewards[i] = r
            sum_r = empiricalMeans[a] * actionCounts[a] + r
            actionCounts[a] += 1.0
            empiricalMeans[a] = sum_r / actionCounts[a]
        return empiricalMeans, rewards

    def thompsonSamplingBandit(self,prior,nIterations,k=1):
        '''Thompson sampling algorithm for Bernoulli bandits (assume no discount factor)

        Inputs:
        prior -- initial beta distribution over the average reward of each arm (|A|x2 matrix such that prior[a,0] is the alpha hyperparameter for arm a and prior[a,1] is the beta hyperparameter for arm a)  
        nIterations -- # of arms that are pulled
        k -- # of sampled average rewards

        Outputs: 
        empiricalMeans -- empirical average of rewards for each arm (array of |A| entries)
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        empiricalMeans = np.zeros(self.mdp.nActions)

        rewards = np.zeros(nIterations)
        for i in range(nIterations):
            samples = np.random.beta(prior[:, 0], prior[:, 1], size=(k, prior.shape[0]))
            empiricalMeans = np.mean(samples, axis=0)
            a = np.argmax(empiricalMeans)
            r = self.sampleRewardAndNextState(0, a)[0]
            rewards[i] = r
            prior[a, 0] += r
            prior[a, 1] += 1 - r

        return empiricalMeans, rewards

    def UCBbandit(self,nIterations):
        '''Upper confidence bound algorithm for bandits (assume no discount factor)

        Inputs:
        nIterations -- # of arms that are pulled

        Outputs: 
        empiricalMeans -- empirical average of rewards for each arm (array of |A| entries)
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        empiricalMeans = np.zeros(self.mdp.nActions)

        rewards = np.zeros(nIterations)
        # Try every action once
        actionCounts = np.zeros(self.mdp.nActions)
        for a in range(self.mdp.nActions):
            r = self.sampleRewardAndNextState(0, a)[0]
            rewards[a] = r
            empiricalMeans[a] = r
            actionCounts[a] += 1

        for i in range(self.mdp.nActions, nIterations):
            n = i + 1
            a = np.argmax(empiricalMeans + np.sqrt(2 * np.log(n) / actionCounts))
            r = self.sampleRewardAndNextState(0, a)[0]
            rewards[i] = r
            sum_r = empiricalMeans[a] * actionCounts[a] + r
            actionCounts[a] += 1
            empiricalMeans[a] = sum_r / actionCounts[a]


        return empiricalMeans, rewards