# Summary



![img](/home/roees/DRL course/typoraImages/theRLframework__summary___The_solution.png)

State-value function for golf-playing agent (Sutton and Barto, 2017)



### Policies

------

- A **deterministic policy** is a mapping \pi: \mathcal{S}\to\mathcal{A}*π*:S→A. For each state s\in\mathcal{S}*s*∈S, it yields the action a\in\mathcal{A}*a*∈Athat the agent will choose while in state s*s*.
- A **stochastic policy** is a mapping \pi: \mathcal{S}\times\mathcal{A}\to [0,1]*π*:S×A→[0,1]. For each state s\in\mathcal{S}*s*∈S and action a\in\mathcal{A}*a*∈A, it yields the probability \pi(a|s)*π*(*a*∣*s*) that the agent chooses action a*a* while in state s*s*.



### State-Value Functions

------

- The **state-value function** for a policy \pi*π* is denoted v_\pi*v**π*. For each state s \in\mathcal{S}*s*∈S, it yields the expected return if the agent starts in state s*s* and then uses the policy to choose its actions for all time steps. That is, v_\pi(s) \doteq \text{} \mathbb{E}_\pi[G_t|S_t=s]*v**π*(*s*)≐E*π*[*G**t*∣*S**t*=*s*]. We refer to v_\pi(s)*v**π*(*s*) as the **value of state ss under policy \piπ**.
- The notation \mathbb{E}_\pi[\cdot]E*π*[⋅] is borrowed from the suggested textbook, where \mathbb{E}_\pi[\cdot]E*π*[⋅] is defined as the expected value of a random variable, given that the agent follows policy \pi*π*.



### Bellman Equations

------

- The **Bellman expectation equation for v_\pivπ** is: v_\pi(s) = \text{} \mathbb{E}_\pi[R_{t+1} + \gamma v_\pi(S_{t+1})|S_t =s].*v**π*(*s*)=E*π*[*R**t*+1+*γ**v**π*(*S**t*+1)∣*S**t*=*s*].



### Optimality

------

- A policy \pi'*π*′ is defined to be better than or equal to a policy \pi*π* if and only if v_{\pi'}(s) \geq v_\pi(s)*v**π*′(*s*)≥*v**π*(*s*) for all s\in\mathcal{S}*s*∈S.
- An **optimal policy \pi_\*π∗** satisfies \pi_* \geq \pi*π*∗≥*π* for all policies \pi*π*. An optimal policy is guaranteed to exist but may not be unique.
- All optimal policies have the same state-value function v_**v*∗, called the **optimal state-value function**.



### Action-Value Functions

------

- The **action-value function** for a policy \pi*π* is denoted q_\pi*q**π*. For each state s \in\mathcal{S}*s*∈S and action a \in\mathcal{A}*a*∈A, it yields the expected return if the agent starts in state s*s*, takes action a*a*, and then follows the policy for all future time steps. That is, q_\pi(s,a) \doteq \mathbb{E}_\pi[G_t|S_t=s, A_t=a]*q**π*(*s*,*a*)≐E*π*[*G**t*∣*S**t*=*s*,*A**t*=*a*]. We refer to q_\pi(s,a)*q**π*(*s*,*a*) as the **value of taking action aa in state ss under a policy \piπ** (or alternatively as the **value of the state-action pair s, as,a**).
- All optimal policies have the same action-value function q_**q*∗, called the **optimal action-value function**.



### Optimal Policies

------

- Once the agent determines the optimal action-value function q_**q*∗, it can quickly obtain an optimal policy \pi_**π*∗ by setting \pi_*(s) = \arg\max_{a\in\mathcal{A}(s)} q_*(s,a)*π*∗(*s*)=argmax*a*∈A(*s*)*q*∗(*s*,*a*).