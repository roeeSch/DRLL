## Review:

<img src="/home/roees/DRL course/MC_method_typImg/review.png" style="zoom:50%" /> <img src="/home/roees/DRL course/MC_method_typImg/review2.png" style="zoom:50%" /> <img src="/home/roees/DRL course/MC_method_typImg/1556792050086.png" style="zoom:70%" />





Go over the last two lessons: The RL Framework (The Problem and The Solution)



## Gridworld Example

![1556792941394](/home/roees/DRL course/MC_method_typImg/gridworld_example2.png)

4 options for the robot:



The size of the set \mathcal{S}^+S+ of all states (including terminal states) =4

The size of the set \mathcal{A}A of all actions=4







#### Intuition based on 2 episodes:![1556793316562](/home/roees/DRL course/MC_method_typImg/episodes.png)

From the 2 episodes above : Episode 1 - choosing up in state s0 yielded a final reward of 6 as opposed to Episode 2 - "right" in s0 and yielded final value of 5. ==> informative? 



![1556793788882](/home/roees/DRL course/MC_method_typImg/episodes_cont2.png)



#### MC prediction

The policy obtaind by choosin maximum action:


![1556793878465](/home/roees/DRL course/MC_method_typImg/episodes_cont3.png)

On the way to obtain an optimal policy:

![1556794000760](/home/roees/DRL course/MC_method_typImg/episodes_cont4.png)



#### Q table

![1556794180789](/home/roees/DRL course/MC_method_typImg/MC_prediction.png)

**note:** If the agent follows a policy for many episodes, we can use the results to directly estimate the action-value function corresponding to the same policy.

**note** The Q-table is used to estimate the action-value function.



Estimating the action-value function with a Q-table is an important intermediate step. It is also refered to as the **prediction problem**.

**Prediction Problem**: *Given a policy, how might the agent estimate the value function for that policy?*

#### First-visit MC \ Every-visit MC:

If in a single episode we visit the same state twice, how do we update the Q-table? Here are two optiosn:

![1556794722448](/home/roees/DRL course/MC_method_typImg/MC_methods.png)



**Option 1: Every-visit MC Prediction**

Average the returns following all visits to each state-action pair, in all episodes.

**Option 2: First-visit MC Prediction**

For each episode, we only consider the first visit to the state-action pair. The pseudocode for this option can be found below.

**The pseudo code:**

![1556794975168](/home/roees/DRL course/MC_methods2.md)

- *Q* - Q-table, with a row for each state and a column for each action. The entry corresponding to state s*s* and action a*a* is denoted Q(s,a).
- *N* - table that keeps track of the number of first visits we have made to each state-action pair.
- *returns*_*sum* - table that keeps track of the sum of the rewards obtained after first visits to each state-action pair.

In the algorithm, the number of episodes the agent collects is equal to num_episodes. After each episode, N and returns_sum are updated to store the information contained in the episode. Then, after all of the episodes have been collected and the values in N and returns_sum have been finalized, we quickly obtain the final estimate for Q.





#### First-visit or Every-visit?


Both the first-visit and every-visit method are **guaranteed to converge** to the true action-value function, as the number of visits to each state-action pair approaches infinity. (*So, in other words, as long as the agent gets enough experience with each state-action pair, the value function estimate will be pretty close to the true value.*) In the case of first-visit MC, convergence follows from the [Law of Large Numbers](https://en.wikipedia.org/wiki/Law_of_large_numbers), and the details are covered in section 5.1 of the [textbook](http://go.udacity.com/rl-textbook).

If you are interested in learning more about the difference between first-visit and every-visit MC methods, you are encouraged to read Section 3 of [this paper](http://www-anw.cs.umass.edu/legacy/pubs/1995_96/singh_s_ML96.pdf). The results are summarized in Section 3.6. The authors show:

- Every-visit MC is [biased](https://en.wikipedia.org/wiki/Bias_of_an_estimator), whereas first-visit MC is unbiased (see Theorems 6 and 7).
- Initially, every-visit MC has lower [mean squared error (MSE)](https://en.wikipedia.org/wiki/Mean_squared_error), but as more episodes are collected, first-visit MC attains better MSE (see Corollary 9a and 10a, and Figure 4).



### OpenAI Gym: BlackJackEnv



contunue here:

https://classroom.udacity.com/nanodegrees/nd893/parts/8f607726-757e-4ef5-8b64-f2368755b89a/modules/a85374fa-6a60-425b-a480-85b211c5bd5d/lessons/b1d9586f-1b1a-48f4-be1e-4c08a0912082/concepts/c18ebeaf-68cb-4636-b527-7eb0d7fc7efe