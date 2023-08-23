# Vertex Agents & Bandits

> training and serving Bandits with TF Agents and Vertex AI 

### Why reinforcement learning?
* train algorithms that consider long-term (cumulative value) of decisions
* explore & exploit tradeoffs between short and long term value (e.g., the difference between the short term value of "click-bait" vs the long-term value of overall user satisafaction, as highlighted in  [DNN for YouTube Recommendations](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45530.pdf)
* make a sequence of decisions, where each decision, or action, possibly impacts future decisions

> return a **distribution** over predictions rather than a single prediction

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

## RL flavors

### Policy-based RL
> Goal: learn a stochastic policy to maximize expected return

* Policies accept some number of tensors as an observation and usually pass it through one or more neural networks
* Emit a distribution over actions
* This Network is a FFN that takes images and emits logits over number_of_actions decisions.
* Policy accepts the Network(s) and provides at least the `_distribution` method.
* Side info becomes part of the Trajectory.  It can be used by metrics and is stored in replay buffers to be used by the training algorithm.

# Design in TF-Agents

### Environment 
* The `environment` represents the user and the agent is the recommender system. 
* This is a POMDP because we don’t know the user’s actual internal state s_t. 
* We have observations from the user o_t which include things like clicks (item selected), likes etc. 
* The rewards could be likes, watch time etc. The actions of the recommender are a set A_t = {a1, … ak} of k items recommended at a time. 

### Policy
* The main job of the Policy is to map observations from the user to actions. 
* The action is a set of K recommended items. The policy is created by the Agent and contains a reference to the Network: 

```
Policy.__init__(time_step_spec, action_spec, network, ...)
```
### Trajectory
* batch of tensors representing `observations`, `actions`, `rewards`, `discounts`

