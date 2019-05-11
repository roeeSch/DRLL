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



### Greedy Policies

You can think of the agent who follows an ϵ*-greedy policy as always having a (potentially unfair) coin at its disposal, with probability ϵ* of landing heads. After observing a state, the agent flips the coin.

- If the coin lands tails (so, with probability 1−*ϵ*), the agent selects the greedy action.
- If the coin lands heads (so, with probability \epsilon*ϵ*), the agent selects an action *uniformly* at random from the set of available (non-greedy **AND** greedy) actions.

In order to construct a policy \pi*π* that is \epsilon*ϵ*-greedy with respect to the current action-value function estimate *Q*, we will set

If *a* maximizes *Q*(*s*,*a*)

​      π*(*a*|*s*)⟵1−*ϵ*+*ϵ*∣|A(*s*)|  

else

​      π(*a*∣*s*)⟵ϵ*∣|A(*s*)|

for each *s*∈S and *a*∈A(*s*).

Mathematically,A(*s*) is the set of all possible actions at state s*s* (which may be 'up', 'down','right', 'left' for example), and ∣A(*s*)∣ the number of possible actions (including the optimal one!). The reason why we include an extra term *ϵ*/∣A(*s*)∣ for the optimal action is because the sum of all the probabilities needs to be 1. If we sum over the probabilities of performing non-optimal actions, we will get (∣A(*s*)∣−1)×*ϵ*/∣A(*s*)∣, and adding this to 1−*ϵ*+*ϵ*/∣A(*s*)∣ gives one.

Note that *ϵ* must always be a value between 0 and 1, inclusive (that is, *ϵ*∈[0,1]).



# MC Control



So far, you have learned how the agent can take a policy *π*, use it to interact with the environment for many episodes, and then use the results to estimate the action-value function *$q_\pi$* with a Q-table.

Then, once the Q-table closely approximates the action-value function  *$q_\pi​$*, the agent can construct the policy *π*′ that is ϵ*-greedy with respect to the Q-table, which will yield a policy that is better than the original policy *π*.

Furthermore, if the agent alternates between these two steps, with:

- **Step 1**: using the policy *$π$* to construct the Q-table, and
- **Step 2**: improving the policy by changing it to be *ϵ*-greedy with respect to the Q-table ($\pi' \leftarrow \epsilon\text{-greedy}(Q)$, $\pi \leftarrow \pi'$),

we will eventually obtain the optimal policy $\pi_*​$.

Since this algorithm is a solution for the **control problem** (defined below), we call it a **Monte Carlo control method**.

> **Control Problem**: Estimate the optimal policy.

It is common to refer to **Step 1** as **policy evaluation**, since it is used to determine the action-**value**function of the policy. Likewise, since **Step 2** is used to **improve** the policy, we also refer to it as a **policy improvement** step.





![1557483217518](/home/roees/DRL course/typoraImages/Part1/E-GREEDY.png)

So, using this new terminology, we can summarize what we've learned to say that our **Monte Carlo control method** alternates between **policy evaluation** and **policy improvement** steps to recover the optimal policy $\pi_*$.

## The Road Ahead

------

You now have a working algorithm for Monte Carlo control! So, what's to come?

- In the next concept (**Exploration vs. Exploitation**), you will learn more about how to set the value of \epsilon*ϵ* when constructing \epsilon*ϵ*-greedy policies in the policy improvement step.
- Then, you will learn about two improvements that you can make to the policy evaluation step in your control algorithm.
  - In the **Incremental Mean** concept, you will learn how to update the policy after every episode (instead of waiting to update the policy until after the values of the Q-table have fully converged from many episodes).
  - In the **Constant-alpha** concept, you will learn how to train the agent to leverage its most recent experience more effectively.

Finally, to conclude the lesson, you will write your own algorithm for Monte Carlo control to solve OpenAI Gym's Blackjack environment, to put your new knowledge to practice!





### Exploration vs. Exploitation
                       <img src="/home/roees/DRL course/typoraImages/exploration-vs-exploitation.png" style="zoom:30%" />

## Solving Environments in OpenAI Gym

------

In many cases, we would like our reinforcement learning (RL) agents to learn to maximize reward as quickly as possible. This can be seen in many OpenAI Gym environments.

For instance, the [FrozenLake-v0](https://gym.openai.com/envs/FrozenLake-v0/) environment is considered solved once the agent attains an average reward of 0.78 over 100 consecutive trials.

![img](/home/roees/DRL course/typoraImages/Part1/E-GREEDY2.png)

Algorithmic solutions to the [FrozenLake-v0](https://gym.openai.com/envs/FrozenLake-v0/) environment are ranked according to the number of episodes needed to find the solution.



Solutions to [Taxi-v1](https://gym.openai.com/envs/Taxi-v1/), [Cartpole-v1](https://gym.openai.com/envs/CartPole-v1/), and [MountainCar-v0](https://gym.openai.com/envs/MountainCar-v0/) (along with many others) are also ranked according to the number of episodes before the solution is found. Towards this objective, it makes sense to design an algorithm that learns the optimal policy $\pi_*$ as quickly as possible.



## Exploration-Exploitation Dilemma

------

Recall that the environment's dynamics are initially unknown to the agent. Towards maximizing return, the agent must learn about the environment through interaction.

At every time step, when the agent selects an action, it bases its decision on past experience with the environment. And, towards minimizing the number of episodes needed to solve environments in OpenAI Gym, our first instinct could be to devise a strategy where the agent always selects the action that it believes (*based on its past experience*) will maximize return. With this in mind, the agent could follow the policy that is greedy with respect to the action-value function estimate. We examined this approach in a previous video and saw that it can easily lead to convergence to a sub-optimal policy.

To see why this is the case, note that in early episodes, the agent's knowledge is quite limited (and potentially flawed). So, it is highly likely that actions *estimated* to be non-greedy by the agent are in fact better than the *estimated* greedy action.

With this in mind, a successful RL agent cannot act greedily at every time step (*that is*, it cannot always **exploit** its knowledge); instead, in order to discover the optimal policy, it has to continue to refine the estimated return for all state-action pairs (*in other words*, it has to continue to **explore** the range of possibilities by visiting every state-action pair). That said, the agent should always act *somewhat greedily*, towards its goal of maximizing return *as quickly as possible*. This motivated the idea of an \epsilon*ϵ*-greedy policy.

We refer to the need to balance these two competing requirements as the **Exploration-Exploitation Dilemma**. One potential solution to this dilemma is implemented by gradually modifying the value of \epsilon*ϵ*when constructing ϵ*-greedy policies.

## Setting the Value of $\epsilon$, in Theory

------

It makes sense for the agent to begin its interaction with the environment by favoring **exploration** over **exploitation**. After all, when the agent knows relatively little about the environment's dynamics, it should distrust its limited knowledge and **explore**, or try out various strategies for maximizing return. With this in mind, the best starting policy is the equiprobable random policy, as it is equally likely to explore all possible actions from each state. You discovered in the previous quiz that setting *ϵ*=1 yields an \epsilon*ϵ*-greedy policy that is equivalent to the equiprobable random policy.

At later time steps, it makes sense to favor **exploitation** over **exploration**, where the policy gradually becomes more greedy with respect to the action-value function estimate. After all, the more the agent interacts with the environment, the more it can trust its estimated action-value function. You discovered in the previous quiz that setting *ϵ*=0 yields the greedy policy (or, the policy that most favors exploitation over exploration).

Thankfully, this strategy (of initially favoring exploration over exploitation, and then gradually preferring exploitation over exploration) can be demonstrated to be optimal.

## Greedy in the Limit with Infinite Exploration (GLIE)

------

In order to guarantee that MC control converges to the optimal policy $\pi_*$, we need to ensure that two conditions are met. We refer to these conditions as **Greedy in the Limit with Infinite Exploration (GLIE)**. In particular, if:

- every state-action pair s, a*s*,*a* (for all $s\in\mathcal{S}​$ and $a\in\mathcal{A}(s)​$) is visited infinitely many times, and
- the policy converges to a policy that is greedy with respect to the action-value function estimate $Q$,

then MC control is guaranteed to converge to the optimal policy (in the limit as the algorithm is run for infinitely many episodes). These conditions ensure that:

- the agent continues to explore for all time steps, and
- the agent gradually exploits more (and explores less).

One way to satisfy these conditions is to modify the value of \epsilon*ϵ* when specifying an \epsilon*ϵ*-greedy policy. In particular, let \epsilon_i*ϵ**i* correspond to the i*i*-th time step. Then, both of these conditions are met if:

- $\epsilon_i > 0$ for all time steps i*i*, and
- $\epsilon_i$ decays to zero in the limit as the time step *i* approaches infinity (that is, $\lim_{i\to\infty} \epsilon_i = 0$).

For example, to ensure convergence to the optimal policy, we could set $\epsilon_i = \frac{1}{i}​$. (You are encouraged to verify that $\epsilon_i > 0​$ for all *i*, and $\lim_{i\to\infty} \epsilon_i = 0​$.



## Setting the Value of $\epsilon$, in Practice

------

As you read in the above section, in order to guarantee convergence, we must let \epsilon_i*ϵ**i* decay in accordance with the GLIE conditions. But sometimes "guaranteed convergence" *isn't good enough* in practice, since this really doesn't tell you how long you have to wait! It is possible that you could need trillions of episodes to recover the optimal policy, for instance, and the "guaranteed convergence" would still be accurate!

> Even though convergence is **not** guaranteed by the mathematics, you can often get better results by either:
>
> - using fixed $\epsilon$, or
> - letting $\epsilon_i$ decay to a small positive number, like 0.1.

This is because one has to be very careful with setting the decay rate for \epsilon*ϵ*; letting it get too small too fast can be disastrous. If you get late in training and \epsilon*ϵ* is really small, you pretty much want the agent to have already converged to the optimal policy, as it will take way too long otherwise for it to test out new actions!

As a famous example in practice, you can read more about how the value of \epsilon*ϵ* was set in the famous DQN algorithm by reading the Methods section of [the research paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf):

> *The behavior policy during training was epsilon-greedy with epsilon annealed linearly from 1.0 to 0.1 over the first million frames, and fixed at 0.1 thereafter.*

When you implement your own algorithm for MC control later in this lesson, you are strongly encouraged to experiment with setting the value of *ϵ* to build your intuition.



### Incremental Mean - update estimation of $Q$ after each episode



In our current algorithm for Monte Carlo control, we collect a large number of episodes to build the Q-table (as an estimate for the action-value function corresponding to the agent's current policy). Then, after the values in the Q-table have converged, we use the table to come up with an improved policy.

Maybe it would be more efficient to update the Q-table **after every episode**. Then, the updated Q-table could be used to improve the policy. That new policy could then be used to generate the next episode, and so on.



![MC Control with Incremental Mean](/home/roees/DRL course/typoraImages/incrementalMean.png)



![1557493984239](/home/roees/DRL course/typoraImages/incrementalMean2.png)



This is yields much more efficient convergance to optimal policy. 

In this case, even though we're updating the policy before the values in the Q-table accurately approximate the action-value function, this lower-quality estimate nevertheless still has enough information to help us propose successively better policies. If you're curious to learn more, you can read section 5.6 of [the textbook](https://s3-us-west-1.amazonaws.com/udacity-dlnfd/suttonbookdraft2018jan1.pdf).

![1557494348264](/home/roees/DRL course/typoraImages/Part1/incrementalMeanMC_control.png)

There are two relevant tables:

- $Q$ - Q-table, with a row for each state and a column for each action. The entry corresponding to state **$s$** and action a*a* is denoted $Q(s,a)$.
- *N* - table that keeps track of the number of first visits we have made to each state-action pair.

The number of episodes the agent collects is equal to **num_episodes**.

The algorithm proceeds by looping over the following steps:

- **Step 1**: The policy *π* is improved to be \epsilon*ϵ*-greedy with respect to *Q*, and the agent uses \pi*π* to collect an episode.
- **Step 2**: *N* is updated to count the total number of first visits to each state action pair.
- **Step 3**: The estimates in *Q* are updated to take into account the most recent information.

In this way, the agent is able to improve the policy after every episode!





### Constant-alpha


Another improvement that you can make to your Monte Carlo control algorithm.

**Incremental Mean** just averages over all previous returns equally. Since policy updates after each episode, it is better to increase the weight $ \frac{1}{N(a,s)}$ as experience is obtained. New weight is denoteated as $\alpha$.

The pseudocode for constant-\alpha*α* GLIE MC Control:

<img src="/home/roees/DRL course/constAlphaMC_control.png" style="zoom:20%" />



### Setting the Value of $\alpha$

------

Recall the update equation that we use to amend the values in the Q-table: 

​                                               $Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha (G_t - Q(S_t, A_t))$

To examine how to set the the value of $\alpha​$ in more detail, we will slightly rewrite the equation as follows:

​                                                   $Q(S_t,A_t) \leftarrow (1-\alpha)Q(S_t,A_t) + \alpha G_t$



Here are some guiding principles that will help you to set the value of α* when implementing constant-*α* MC control:

- You should always set the value for \alpha*α* to a number greater than zero and less than (or equal to) one.
  - If $\alpha=0$, then the action-value function estimate is never updated by the agent.
  - If $\alpha = 1$, then the final value estimate for each state-action pair is always equal to the last return that was experienced by the agent (after visiting the pair).
- Smaller values for \alpha*α* encourage the agent to consider a longer history of returns when calculating the action-value function estimate. Increasing the value of \alpha*α* ensures that the agent focuses more on the most recently sampled returns.

> **Important Note**: When implementing constant-*α* MC control, you must be careful to not set the value of *α* too close to 1. This is because very large values can keep the algorithm from converging to the optimal policy $\pi_*$. However, you must also be careful to not set the value of *α* too low, as this can result in an agent who learns too slowly. The best value of *α* for your implementation will greatly depend on your environment and is best gauged through trial-and-error.





### Summary:

![img](/home/roees/DRL course/typoraImages/Part1/MC_blackjack_summary.png)



### Monte Carlo Methods

------

- Monte Carlo methods - even though the underlying problem involves a great degree of randomness, we can infer useful information that we can trust just by collecting a lot of samples.
- The **equiprobable random policy** is the stochastic policy where - from each state - the agent randomly selects from the set of available actions, and each action is selected with equal probability.

### MC Prediction

------

- Algorithms that solve the **prediction problem** determine the value function $v_\pi$ (or $q_\pi$) corresponding to a policy $\pi$.
- When working with finite MDPs, we can estimate the action-value function q_\pi*q**π* corresponding to a policy \pi*π* in a table known as a **Q-table**. This table has one row for each state and one column for each action. The entry in the *s*-th row and *a*-th column contains the agent's estimate for expected return that is likely to follow, if the agent starts in state *s*, selects action *a*, and then henceforth follows the policy $\pi$.
- Each occurrence of the state-action pair *s*,*a* ($s\in\mathcal{S},a\in\mathcal{A}$) in an episode is called a **visit to s,a**.
- There are two types of MC prediction methods (for estimating $q_\pi$):
  - **First-visit MC** estimates $q_\pi(s,a)$ as the average of the returns following *only first* visits to s,a*s*,*a*(that is, it ignores returns that are associated to later visits).
  - **Every-visit MC** estimates $q_\pi(s,a)$ as the average of the returns following *all* visits to s,a*s*,*a*.

### Greedy Policies

------

- A policy is **greedy** with respect to an action-value function estimate *Q* if for every state s*∈S, it is guaranteed to select an action *a*∈A(*s*) such that $a = \arg\max_{a\in\mathcal{A}(s)}Q(s,a)$. (It is common to refer to the selected action as the **greedy action**.)
- In the case of a finite MDP, the action-value function estimate is represented in a Q-table. Then, to get the greedy action(s), for each row in the table, we need only select the action (or actions) corresponding to the column(s) that maximize the row.

### Epsilon-Greedy Policies

------

- A policy is *ϵ*-greedy with respect to an action-value function estimate *Q* if for every state $s\in\mathcal{S}$,
  - with probability $1-\epsilon$, the agent selects the greedy action, and
  - with probability *ϵ*, the agent selects an action *uniformly* at random from the set of available (non-greedy **AND** greedy) actions.

### MC Control

------

- Algorithms designed to solve the **control problem** determine the optimal policy \pi_**π*∗ from interaction with the environment.
- The **Monte Carlo control method** uses alternating rounds of policy evaluation and improvement to recover the optimal policy.

### Exploration vs. Exploitation

------

- All reinforcement learning agents face the **Exploration-Exploitation Dilemma**, where they must find a way to balance the drive to behave optimally based on their current knowledge (**exploitation**) and the need to acquire knowledge to attain better judgment (**exploration**).

- In order for MC control to converge to the optimal policy, the

   

  Greedy in the Limit with Infinite Exploration (GLIE)

   

  conditions must be met:

  - every state-action pair s, a*s*,*a* (for all *s*∈S and *a*∈A(*s*)) is visited infinitely many times, and
  - the policy converges to a policy that is greedy with respect to the action-value function estimate *Q*.

### Incremental Mean

------

- (In this concept, we amended the policy evaluation step to update the Q-table after every episode of interaction.)

### Constant-alpha

------

- (In this concept, we derived the algorithm for **constant-\alphaα MC control**, which uses a constant step-size parameter $\alpha$.)
- The step-size parameter \alpha*α* must satisfy $0 < \alpha \leq 1$. Higher values of *α* will result in faster learning, but values of α* that are too high can prevent MC control from converging to $\pi_*$.