# Deep Q-Networks



As you'll learn in this lesson, the Deep Q-Learning algorithm represents the optimal action-value function $q_*​$ as a neural network (instead of a table).

Unfortunately, reinforcement learning is [notoriously unstable](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.73.3097&rep=rep1&type=pdf) when neural networks are used to represent the action values. In this lesson, you'll learn all about the Deep Q-Learning algorithm, which addressed these instabilities by using **two key features**:

- Experience Replay
- Fixed Q-Targets

## Additional References

------

- Riedmiller, Martin. "Neural fitted Q iteration–first experiences with a data efficient neural reinforcement learning method." European Conference on Machine Learning. Springer, Berlin, Heidelberg, 2005. <http://ml.informatik.uni-freiburg.de/former/_media/publications/rieecml05.pdf>
- Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning." Nature518.7540 (2015): 529. <http://www.davidqiu.com:8888/research/nature14236.pdf>





Atari Games:

![1558263554747](/home/roees/DRL course/typoraImages/Part2/atari_intro.png)

The Q-value approximation network produces q values for all actions.

![1558263781588](/home/roees/DRL course/typoraImages/Part2/atari_intro2.png)



####  DeepMind leveraged a **Deep Q-Network (DQN)** to build the Deep Q-Learning algorithm that learned to play many Atari video games better than humans

- The DQN takes the state as input, and returns the corresponding predicted action values for each possible game action.
- For each atari game, the DQN was trained from scratch



## Experience Replay (coping with correlation type 1)

![1558264321374](/home/roees/DRL course/typoraImages/Part2/ValFun_ExperienceReplay_1.png)



Saving tuples of SARS and going over it better usage of experience.

The reason that these tuples are sampled is to deal with the problem that temporally close SA can be correlated which creates oscillations and divergence (?). Randomly sampling the history helps deal with this problem.

![1558264983647](/home/roees/DRL course/typoraImages/Part2/ValFun_ExperienceReplay_2.png)



Experience replay:

- Experience replay is based on the idea that we can learn better, if we do multiple passes over the same experience.
- Experience replay is used to generate uncorrelated experience data for online training of deep RL agents.



## Fixed Q-Targets  (coping with correlation type 2)

The gradient descent update rule:

![1558265329798](/home/roees/DRL course/typoraImages/Part2/ValFun_FixedQTargets_1.png)



This update rule is like chasing a moving target. Hence the following solution:

![1558265544673](/home/roees/DRL course/typoraImages/Part2/ValFun_FixedQTargets_2.png)

We freeze $w$ by saving its latest value into $w^-$ learn for a few steps and so on. This decouples the target from the parameters. Makes the algorithm much more stable and less likely to diverge or fall into oscillations.

## Summary

------

In Q-Learning, we **update a guess with a guess**, and this can potentially lead to harmful correlations. To avoid this, we can update the parameters *w* in the network $\hat{q}$ to better approximate the action value corresponding to state *S* and action *A* with the following update rule:

![1558266052594](/home/roees/DRL course/typoraImages/Part2/ValFun_FixedQTargets_3.png)

where w^-*w*− are the weights of a separate target network that are not changed during the learning step, and (*S*, *A*, *R*, *S*′) is an experience tuple.

**Note**: Ever wondered how the example in the video would look in real life? See: [Carrot Stick Riding](https://www.youtube.com/watch?v=-PVFBGN_zoM).



The following are true:

- The Deep Q-Learning algorithm uses two separate networks with identical architectures.
- The target Q-Network's weights are updated less often (or more slowly) than the primary Q-Network.
- Without fixed Q-targets, we would encounter a harmful form of correlation, whereby we shift the parameters of the network based on a constantly moving target.

Its recommended to read the [research paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) that introduces the Deep Q-Learning algorithm.

