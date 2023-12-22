## RL for RecSys TL;DR

**Simulated environments**
* In RL, the quality of a policy is often measured by the average reward received if the policy is followed by the agent to select actions. 
* If the environment can be simulated (e.g., gaming), evaluation is as simple as running the policy. 
* However, for some real-life problems (e.g., autonomous driving and healthcare), running a new policy in the actual environment can be expensive, risky and/or unethical
* Training from a simulated environment for policy evaluation is common practice, but building a simulator that accurately reflects the real-world is challenging, e.g., how do you shape rewards for actions that haven't been observed? What biases could we introduce the our agent?

**Off-policy**
* RL agents essentially do two things when training (learning):
  * take actions --> aka `Behavior Policy`
  * learn which actions are good vs bad (`experience`) --> aka `Target Policy` 
* An agent uses a **Behavior Policy** to determine its action (behavior) in a given state
* An agent uses a **Target Policy** to learn from the rewards recieved for its actions (i.e., to determine updated Q values)

> `Off-policy` refers to training or evaluating a policy (the “target policy”) with historical data collected by a different policy (the “behavior policy”)

Off-policy is challenging in some RL use cases, because training data is in the form of a "trajectory" (i.e., a sequence of state-action-reward tuples where states depend on actions chosen earlier in the sequence)
* This means if a policy "deviates" from the trajectory (i.e., if it chooses different actions than those observed in historical data), all future states and rewards will change, further deviating from the collected data

**Contextual Bandits**
* Contextual bandits are a subclass of RL algorithms where the agent's actions **do not affect future states**
* Learns the optimal policy of a dynamic environment, where the objective of the agent then translates to maximizing the average future reward or in other words, picking the best arm (the one with the highest expected payoff)
* Receives context (side information); specifically for RecSys, these are the features of the user and their past interactions (e.g., the items they interacted with)

## Why reinforcement learning?
* train algorithms that consider long-term (cumulative value) of decisions
* **Explore & Exploit tradeoffs** between short and long term value (e.g., the difference between the short term value of "click-bait" vs the long-term value of overall user satisafaction, as highlighted in  [DNN for YouTube Recommendations](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45530.pdf))
* make a sequence of decisions, where each decision, or action, possibly impacts future decisions
* return a **distribution** over predictions rather than a single prediction*

### Using RL for recommendations
* User vectors are the environment observations
* Items to recommend are the agent actions applied on the environment
* Approximate user ratings are the environment rewards generated as feedback to the observations and actions

| RL concept | Traditional RL | RL for RecSys |
| :--------: | :------------: | :-----------: |
|   Agent    | algorithm that learns from trial and error by interacting with the environment | candidate generator |
| Environment| world through which the agent moves / interacts | historical interaction data | 
|   State    | Current condition returned by the environment | use intersts, context |
|   Reward   | An instant return from the environment to appraise the last action | user satisfaction |
|   Action   | possible steps that an agent can take | select from a lare corpus |
|   Policy   | The approach the agent learns to use to determine the next best action based on state | equivalent to a "model" in supervised learning |

For custom training, we implement **off-policy training**, using a static set of pre-collected data records.
* Meaning, given a data record, and its observation, the current policy in training might not choose the same action as the one in said data record.

## RL flavors - TODO

### Model-based vs Model-free

#### Model-based RL - build model while learning/acting

> Goal: learn a stochastic policy to maximize expected return

* Policies accept some number of tensors as an observation and usually pass it through one or more neural networks
* Emit a distribution over actions
* This Network is a FFN that takes images and emits logits over number_of_actions decisions.
* Policy accepts the Network(s) and provides at least the `_distribution` method.
* Side info becomes part of the Trajectory.  It can be used by metrics and is stored in replay buffers to be used by the training algorithm.

### Model-free RL - learn value function or policy directly from data

> **TODO**


### Target Policy vs Behavior Policy

* Behavior: the policy the agent uses to determine its action (behavior) in a given state (e.g., for a given set of user-item features)
* Target: the policy the agent uses to learn from the rewards recieved for its actions (i.e., to detemine optimal Q-values)

### On-Policy Agent (learner)

> if Target policy and Behavior policy are the same, Agent is said to be `on-policy`

* This method requires the training trajectory to be generated from the current policy.
* Typically runs the model / policy in a simulator during training
* Note: distributed training is difficult for this case, due to the need to synchronize model update and data generation in simulation.

### Off-Policy Agent (learner)

> if Target policy != Behavior policy, Agent is said to be `off-policy`

* "Off-policy" refers to the situation where for a data record, given its observation, the current policy in training might not choose the same action as the one in said data record
* This method can learn from training trajectory generated from any policy
* During training, policy is typically trained on static set of pre-collected data; includes observation, action, and reward

## Q-learning

> TODO

## Slate Optimization
In recommender systems, slate optimization is typically the last stage of the recommendation pipeline. The input is a set of candidate items and their features. The output is a slate, i.e., an ordered subset of items. The goal of slate optimization is to find the ordered-subset of items that maximizes long-term user experience. 
* given *K* candidates, find *N* items to fill user's recommendation slate
* Items in the slate impact user response (reward) of others 
* Value of slate depends on user choice model
* Promoting diversity among the items in the slate can improve long-term user experience

![alt text](https://github.com/tottenjordan/tf_vertex_agents/blob/main/imgs/slate_optimization_high_level.png)

Slate optimization in RecSys can be formulated as a Markov Decision Process (MPD)
* **State:** representation of the sequence of items user has already engaged with
* **Action:** set of remaining candidate items; i.e., which item to recommend next
* **State Transition:** how adding a candidate item will lead from one state to another.
* **Reward:** long-term effects of the slate.

> joint optimization of the slate