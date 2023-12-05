# Reinforcement Learning: conceptual

**The RL problem**
* Agent takes action
* Environment changes; agent transitions to a new state
* Agent receives a reward

Information used to determine what happens next, where `State` depends on history `H`

h = S_1, A_1, R_1, S_2, R_2,..., S_t

## RL Foundation: Markov Decision Process (MPD)
All of RL is based on the theoretical framework of Markov Decision Processes (MPDs). 
* They are a way to model any dynamic environment where actions lead to predictable changes to the state
* A Markov state is a state that contains all information about the current state of the world or environment


## Value-Based Reinforcement Learning
In **value-based** reinforcement learning methods, agents maintain a value for all state-action pairs and use those estimates to choose actions that maximise that value (instead of maintaining a policy directly like policy gradient methods).

We represent the function mapping state-action pairs to values (otherwise known as a **Q-function**) for a specific policy $\pi$ in a given [MDP](https://en.wikipedia.org/wiki/Markov_decision_process) as:

$$ Q^{\pi}(\color{#ed005a}{s},\color{#0175c2}{a}) = \mathbb{E}_{\tau \sim P^{\pi}} \left[ \sum_t \gamma^t \color{#00ba47}{R_t}| s_0=\color{#ed005a}s,a=\color{#0175c2}{a_0} \right]$$

where $\tau = \{\color{#ed005a}{s_0}, \color{#0175c2}{a_0}, \color{#00ba47}{r_0}, \color{#ed005a}{s_1}, \color{#0175c2}{a_1}, \color{#00ba47}{r_1}, \cdots \}$. 

In other words, $Q^{\pi}(\color{#ed005a}{s},\color{#0175c2}{a})$ is the expected **value** (sum of discounted rewards) of being in a given <font color='#ed005a'>**state**</font> $\color{#ed005a}s$ and taking the <font color='#0175c2'>**action**</font> $\color{#0175c2}a$ and then following policy ${\pi}$ thereafter.

Efficient value estimations are based on the **_Bellman Optimality Equation_**:

$$ Q^\pi(\color{#ed005a}{s},\color{#0175c2}{a}) =  \color{#00ba47}{r}(\color{#ed005a}{s},\color{#0175c2}{a}) + \gamma  \sum_{\color{#ed005a}{s'}\in \color{#ed005a}{\mathcal{S}}} P(\color{#ed005a}{s'} |\color{#ed005a}{s},\color{#0175c2}{a}) V^\pi(\color{#ed005a}{s'}) $$

which breaks down $Q^{\pi}(\color{#ed005a}{s},\color{#0175c2}{a})$ into 2 parts: 
(1) the immediate reward associated with being in state $\color{#ed005a}{s}$ and taking action $\color{#0175c2}{a}$, 
(2) and the discounted sum of all future rewards. Note that $V^\pi$ here is the expected $Q^\pi$ value for a particular state, i.e.

$$V^\pi(\color{#ed005a}{s}) = \sum_{\color{#0175c2}{a} \in \color{#0175c2}{\mathcal{A}}} \pi(\color{#0175c2}{a} |\color{#ed005a}{s}) Q^\pi(\color{#ed005a}{s},\color{#0175c2}{a})$$

> So, the basic idea behind Q-learning Bellman optimality equation as an iterative update