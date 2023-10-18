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

**Model-based RL:** build/exploit model while learning/acting

**Model-free RL:** learn value function or policy directly from data

## Model-based RL
> Goal: learn a stochastic policy to maximize expected return

* Policies accept some number of tensors as an observation and usually pass it through one or more neural networks
* Emit a distribution over actions
* This Network is a FFN that takes images and emits logits over number_of_actions decisions.
* Policy accepts the Network(s) and provides at least the `_distribution` method.
* Side info becomes part of the Trajectory.  It can be used by metrics and is stored in replay buffers to be used by the training algorithm.

## Model-free RL

> TODO

## Q-learning

> TODO

## Slate Optimization
* given *K* candidates, find *N* items to fill user's recommendation slate
* Items in the slate impact user response (reward) of others 
* Value of slate depends on user choice model

> joint optimization of the slate