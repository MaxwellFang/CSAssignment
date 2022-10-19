import numpy as np
import MDP
import RL2

import matplotlib.pyplot as plt

def sampleBernoulli(mean):
    ''' function to obtain a sample from a Bernoulli distribution

    Input:
    mean -- mean of the Bernoulli
    
    Output:
    sample -- sample (0 or 1)
    '''

    if np.random.rand(1) < mean: return 1
    else: return 0


# Multi-arm bandit problems (3 arms with probabilities 0.25, 0.5 and 0.75)
T = np.array([[[1]],[[1]],[[1]]])
R = np.array([[0.25],[0.5],[0.75]])
discount = 0.999
mdp = MDP.MDP(T,R,discount)
banditProblem = RL2.RL2(mdp,sampleBernoulli)

trials = 100
nIterations = 200
t = np.arange(0, nIterations)
avg_rewards = np.zeros(nIterations)
# Test epsilon greedy strategy
for i in range(trials):
    empiricalMeans, rewards = banditProblem.epsilonGreedyBandit(nIterations=nIterations)
    avg_rewards += rewards
avg_rewards = avg_rewards / trials
plt.plot(t, avg_rewards, 'r', label='Epsilon greedy')
print("\nepsilonGreedyBandit results")
print(empiricalMeans)

# Test Thompson sampling strategy
avg_rewards = np.zeros(nIterations)
for i in range(trials):
    empiricalMeans, rewards = banditProblem.thompsonSamplingBandit(prior=np.ones([mdp.nActions,2]),nIterations=nIterations)
    avg_rewards += rewards
avg_rewards = avg_rewards / trials
plt.plot(t, avg_rewards, 'g', label='Thompson sampling')
print("\nthompsonSamplingBandit results")
print(empiricalMeans)

# Test UCB strategy
avg_rewards = np.zeros(nIterations)
for i in range(trials):
    empiricalMeans, rewards = banditProblem.UCBbandit(nIterations=nIterations)
    avg_rewards += rewards
avg_rewards = avg_rewards / trials
plt.plot(t, avg_rewards, 'b', label='UCB')
print("\nUCBbandit results")
print(empiricalMeans)


plt.xlabel("iteration")
plt.ylabel("average rewards")
plt.title("TestBandit")
plt.legend()
plt.show()

"""
The result is not surprising. Epsilon greedy eventually reach to a point where it is very unlikely to do exploration, so it may get stucked at a local minimum if the rate of decent of epsilon is too quick.
For UCB, the benefit is that the an action will always be chosen at some point because 2 * log(n) / na will become very large at some point. On the downside that it may has longer convergence time because we are 
not choosing the action which brings us the best reward but an upper bound estimates, and that leads to more exploration, depending on the setting of the upper bound.
For Thompson sampling, it seems like it perform the best in this case. It sample the emperical mean based on prior distribution and update the prior distribution based on the observed reward. When the sample size is low,
it will try to explore other actions (as it will be less likely to sample to the mean of the best action), However, once we have enough sample, it became more certain which action is the best to take, so it will more and 
more likely to stick to the best action required.
"""