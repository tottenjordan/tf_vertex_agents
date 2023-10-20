## Why reinforcement learning?
* train algorithms that consider long-term (cumulative value) of decisions
* explore & exploit tradeoffs between short and long term value (e.g., the difference between the short term value of "click-bait" vs the long-term value of overall user satisafaction, as highlighted in  [DNN for YouTube Recommendations](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45530.pdf))
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

For custom training, we implement **off-policy training**, using a static set of pre-collected data records. "Off-policy" refers to the situation where for a data record, given its observation, the current policy in training might not choose the same action as the one in said data record.

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

* This method can learn from training trajectory generated from any policy
* During training, policy is typically trained on static set of pre-collected data; includes observation, action, and reward


## The online learning paradigm
* RL deployments that run batch training and push models at a specific cadence are commonly categorized into `off-policy learning`. And, they are prone to `system bias` because of the long delay from user feedback to model updates
> * In RL terms, this means the algorithm accumulates *regret* as user’s preferences change
* Supervised learning approaches also ignore *exploration*, which is crucial to respond to changing items and user preferences. 
* However, an online approach using continuous training can minimize overall regret
> * match users and items with real-time algorhtms and systems
> * Online and on-policy learning with principled exploration (aka `bandits`)
> * In practice, on-policy systems may reduce the end-to-end policy update delay to < 1 hour, where their batch counterparts could have a delay of several hours to several days 

### Online challenges 
* **Large output space** --> large exploration space without action space reduction
* **Efficient bandits learning in real-time** --> modeling user and context with a *good* tradeoff between accuracy and learning efficiency

### Online + Offline
One approach is to use concepts from both online and offline:
* Offline: dual-encoder models for learning user and item embeddings
* Online: sparse [bipartite graph](https://www.geeksforgeeks.org/bipartite-graph/#:~:text=A%20Bipartite%20Graph%20is%20a,V%20and%20v%20to%20U.) created from offline embeddings

This can help with (a) the cold-start problem by connecting users with fresh content, as well as both (b) corpus exploration and (c) interest exploration

### References
1. [Cascading Linear Submodular Bandits: Accounting for Position Bias and Diversity in Online Learning to Rank](http://auai.org/uai2019/proceedings/papers/248.pdf), G. Hiranandani, H. Singh, P. Gupta, I. A. Burhanuddin, Z. Wen and B. Kveton, 35th Conference on Uncertainty in Artificial Intelligence (2019)
> * account for both position bias and diversity in forming the list of items to recommend
2. [Contextual Combinatorial Cascading Bandits](http://proceedings.mlr.press/v48/lif16.html), , S. Li, B. Wang, S. Zhang, W. Chen, Proceedings of The 33rd International Conference on Machine Learning, PMLR 48:1245-1253, 2016

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