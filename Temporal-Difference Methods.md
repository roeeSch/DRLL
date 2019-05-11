# Temporal-Difference Methods 

------



### Recap MC methods:

In MC methods we learned about the **control problem** in reinforcement learning and implemented some Monte Carlo (MC) control methods.

> > ​	**Control Problem**: Estimate the optimal policy.
>
> In this lesson, you will learn several techniques for Temporal-Difference (TD) control.

### Review


Before continuing, please review **Constant-alpha MC Control** from the previous lesson.

Remember that the constant-\alpha*α* MC control algorithm alternates between **policy evaluation** and **policy improvement** steps to recover the optimal policy $\pi_*$.



<img src="/home/roees/DRL course/typoraImages/Part1/TD_recap_MC.png" style="zoom:30%"	>

​																**Constant-alpha MC Control**



In the **policy evaluation** step, the agent collects an episode $S_0, A_0, R_1, \ldots, S_T$ using the most recent policy π*. After the episode finishes, for each time-step *t*, if the corresponding state-action pair $(S_t,A_t)$ is a first visit, the Q-table is modified using the following **update equation**:

​				$Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha(G_t - Q(S_t, A_t))$

where $G_t := \sum_{s={t+1}}^T\gamma^{s-t-1}R_s$ is the return at time step *t*, and $Q(S_t,A_t)$ is the entry in the Q-table corresponding to state $S_t$ and action $A_t$.

The main idea behind this **update equation** is that $Q(S_t,A_t)$ contains the agent's estimate for the expected return if the environment is in state $S_t$ and the agent selects action $A_t$. If the return $G_t$ is **not** equal to $Q(S_t,A_t)$, then we push the value of $Q(S_t,A_t)$ to make it agree slightly more with the return. The magnitude of the change that we make to $Q(S_t,A_t)$ is controlled by the hyper-parameter *α*>0.



### TD Control: Sarsa


Monte Carlo (MC) control methods require us to complete an entire episode of interaction before updating the Q-table. Temporal Difference (TD) methods will instead update the Q-table after every time step.



<img src="/home/roees/DRL course/typoraImages/Part1/TD_intro.png" style="zoom:50%">



The difference here (Compared to MC methods) is that the updated estimate relies on the current state-action-reward and **next state-action estimated return ($G$)**.

<img src="/home/roees/DRL course/typoraImages/Part1/TD_vs_MC.png" style="zoom:50%">



The temporal aspect - update estimate of $Q$ after each **s**tate-**a**ction **r**eward based on next **s**tate-**a**ction (**sarsa**)

<img src="/home/roees/DRL course/typoraImages/Part1/TD_sarsa0.png" style="zoom:50%">

**Pseudocode**:

<img src="/home/roees/DRL course/typoraImages/Part1/TD_sarsa_pseudoCode.png" style="zoom:20%">

In the algorithm, the number of episodes the agent collects is equal to *num\_episodes*. For every time step *t*≥0, the agent:

- **takes** the action $A_t$ (from the current state $S_t​$) that is *ϵ*-greedy with respect to the Q-table,
- receives the reward $R_{t+1}$ and next state $S_{t+1}$,
- **chooses** the next action $A_{t+1}$ (from the next state $S_{t+1}$) that is *ϵ*-greedy with respect to the Q-table,
- uses the information in the tuple ($S_t$, $A_t$, $R_{t+1}$, $S_{t+1}$, $A_{t+1}$) to update the entry $Q(S_t, A_t)$ in the Q-table corresponding to the current state $S_t$ and the action $A_t$.



#### Quiz Example:

Suppose the agent is using **Sarsa** in its search for the optimal policy, with **\alpha=0.1α=0.1**.

At the end of the 99th episode, the Q-table has the following values:

<img src="/home/roees/DRL course/typoraImages/Part1/TD_1step_example.png" style="zoom:14%">

​																**Beginning of the 100th episode**

Which entry in the Q-table is updated? s1,a-->

What is the new value in the Q-table corresponding to the state-action pair you selected in the answer to the question above?

The action-value for state 1 and action right can be calculated as : 6 + 0.1(-1 + 8 - 6) = 6.1



#### Sarsa(0) and Sarsa-max (Q-Learning):

​						<img src="/home/roees/DRL course/typoraImages/Part1/TD_sarsa0_2.png" style="zoom:60%"> 			<img src="/home/roees/DRL course/typoraImages/Part1/TD_sarsaMax.png" style="zoom:60%">			     

Check out this (optional) [research paper](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.80.7501&rep=rep1&type=pdf) to read the proof that Q-Learning (or Sarsamax) converges.

**Pseudocode**:
								<img src="/home/roees/DRL course/typoraImages/Part1/TD_sarsa_max_pseudocode.png" style="zoom:20%">





**Quiz Example:**

Suppose the agent is using **Q-Learning** in its search for the optimal policy, with **\alpha=0.1α=0.1**.

At the end of the 99th episode, the Q-table has the following values:

​														<img src="/home/roees/DRL course/typoraImages/Part1/TD_QL_example.png" style="zoom:10%">

Say that at the beginning of the 100th episode, the agent starts in **state 1** and selects **action right**. As a result, it receives **reward -1**, and the next state is **state 2**.

<img src="/home/roees/DRL course/typoraImages/Part1/TD_1step_example.png" style="zoom:14%">

​																**Beginning of the 100th episode**

The action-value for state 1 and action right can be calculated as: 	6 + 0.1(-1 + 9 - 6) = 6.2.



#### Expected sarsa:

<img src="/home/roees/DRL course/typoraImages/Part1/TD_expected_Sarsa.png" style="zoom:50%">

**Pseudocode:**

<img src="/home/roees/DRL course/typoraImages/Part1/TD_expected_Sarsa_pseudocode.png" style="zoom:20%">



**Quiz Example (expected sarsa):**

What is the new value in the Q-table corresponding to the state-action pair you selected in the answer to the question above?

(*Suppose that when selecting the actions for the first two timesteps in the 100th episode, the agent was following the epsilon-greedy policy with respect to the Q-table, with epsilon = 0.4.*)

`6 + 0.1*(-1 + (0.4/4*(8+7+8)+(1-0.4+0.4/4)*9) - 6) = 6 + 0.1*(-1 + (2.3+6.3) - 6) = 6 + 0.1*(-1 + 8.6 - 6) = 6.16`



### TD Control: Theory and Practice
------


#### Greedy in the Limit with Infinite Exploration (GLIE)

The **Greedy in the Limit with Infinite Exploration (GLIE)** conditions were introduced in the previous lesson, when we learned about MC control. There are many ways to satisfy the GLIE conditions, all of which involve gradually decaying the value of *ϵ* when constructing *ϵ*-greedy policies.

In particular, let $\epsilon_i$ correspond to the *i*-th time step. Then, to satisfy the GLIE conditions, we need only set $\epsilon_i$ such that:

- $\epsilon_i > 0$ for all time steps *i*, and
- $\epsilon_i$ decays to zero in the limit as the time step *i* approaches infinity (that is, $\lim_{i\to\infty} \epsilon_i = 0$),


#### In Theory

All of the TD control algorithms we have examined (Sarsa, Sarsamax, Expected Sarsa) are **guaranteed to converge** to the optimal action-value function $q_*$, as long as the step-size parameter $\alpha$ is sufficiently small, and the GLIE conditions are met.

Once we have a good estimate for $q_*$, a corresponding optimal policy $\pi_*$ can then be quickly obtained by setting $\pi_*(s) = \arg\max_{a\in\mathcal{A}(s)} q_*(s, a)$ for all $s\in\mathcal{S}$.

#### In Practice

In practice, it is common to completely ignore the GLIE conditions and still recover an optimal policy. (*You will see an example of this in the solution notebook.*)

#### Optimism

You have learned that for any TD control method, you must begin by initializing the values in the Q-table. It has been shown that [initializing the estimates to large values](http://papers.nips.cc/paper/1944-convergence-of-optimistic-and-incremental-q-learning.pdf) can improve performance. For instance, if all of the possible rewards that can be received by the agent are negative, then initializing every estimate in the Q-table to zeros is a good technique. In this case, we refer to the initialized Q-table as **optimistic**, since the action-value estimates are guaranteed to be larger than the true action values.







