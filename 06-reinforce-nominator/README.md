# REINFORCE Recommender Agent

> Top-K Off-Policy Correction for a REINFORCE Recommender System
Minmin Chen, Alex Beutel, Paul Covington, Sagar Jain, Francois Belletti, Ed Chi https://arxiv.org/pdf/1812.02353.pdf

This agent is similar to the base [REINFORCE agent](https://github.com/tensorflow/agents/blob/c8460133b18bd72afd4e806afe3aa7b7b1fdca83/tf_agents/agents/reinforce/reinforce_agent.py#L121) in TF-agents with a few modifcations:
* **off-policy correction**: an additional scaling factor to the standard REINFORCE gradient based on the ratio between the (target) policy we are training and the behaviour/collection policy. Basically a ratio between action probabilities
* **Top-K off-policy correction**: since the agent proposes k actions at a time, the correction factor has to be further adjusted e.g., with log(action_prob)
* **estimating the behavior policy**: the behaviour policy is predicted/approximated by another softmax head in the network. This is used for computing the importance ratio


## TODOs

**1. ScaNN layer in action space (policy)**
* when ScaNN is used, logits will not be computed for all actions.
* So in this case, either do not emit logits in the policy, or emit a tuple `<logits, canidate_actions>`, and update metrics such as `WeightedReturn` to use this structure. 
* Currently it is not possible to detect whether ScaNN is used inside the policy, only in the agent

**2. Confirm using `emit_logits_as_info` and ScaNN (policy)**
* This should not be used in conjunction with ScaNN (provided through `get_candidate_actions_fn`) since in this case the logits will not be computed for all actions.

**3. agent trajectories**
* Does agent assume a trajectory is a single episode, such that `trajectory.step_type` and `trajectory.discount` are essentially ignored?

**4. optimize train step:**

```
@common.function(autograph=False)
def _train_step_fn(data):
    
    # trajectory, weights = data

    def replicated_train_step(experience):
        return tf_agent.train(experience).loss

    per_replica_losses = distribution_strategy.run(
        replicated_train_step, 
        args=(data,)
    )

    # return agent.train(experience=trajectories).loss
    return distribution_strategy.reduce(
        tf.distribute.ReduceOp.MEAN, 
        per_replica_losses, # loss, 
        axis=None
    )
```


## Experiment ideas

**Create experiment comparing:**

[1] rnn
* off_policy_correction_exponent = None
* use_supervised_loss_for_main_policy = True

[2] REINFORCE
* off_policy_correction_exponent = None
* use_supervised_loss_for_main_policy = False

[3] topk REINFORCE
* off_policy_correction_exponent = ~16
* use_supervised_loss_for_main_policy = False



## Sequence data for REINFORCE Recommender

For each `user`, we consider a sequence of user historical interactions with the RecSys, recording the actions taken by the recommender (e.g., items recommended), as well as user feedback (e.g.,`ratings`)

Given such a sequence, we predict the next `action` to take, i.e., items to recommend, so that user satisfaction metrics, e.g., indicated by `ratings` improve

**MPD definiton**

> We translate this setup into a Markov Decision Process (MDP)

**{`S`, `A`, `P`, `R`, `p0`, `y`}**

* **`S`**: a continuous state space describing the user states
* **`A`**: a discrete action space, containing items available for recommendation
* **`P`** : S × A × S → R is the state transition probability
* **`R`** : S × A → R is the reward function, where 𝑟(𝑠, 𝑎) is the immediate reward obtained by performing action 𝑎 at user state `s`
* **`p0`** is the initial state distribution
* **`y`** is the discount factor for future rewards

#### Trajectories

**A `Trajectory` represents a sequence of aligned time steps** 

It captures:
* `observation` and `step_type` from current time step with the computed `action` and `policy_info`
* `Discount`, `reward` and `next_step_type` come from the next time step.
  
We allow `experience` to contain trajectories of different lengths in the *time dimension*, but these have to be padded with dummy values to have a constant size of `T` in the time dimension

* Both `trajectory.reward` and `weights` have to be 0 for these dummy values
* `experience` can be provided in other formats such as `Transition`'s if they can be converted into Trajectories.

#### TimeSteps

**A `TimeStep` contains the data emitted by an environment at each step of interaction**. They include:
* a `step_type`, 
* an `observation` (e.g., NumPy array, dict, or list of arrays), 
* and an associated `reward` and `discount`

**sequential ordering**
* first `TimeStep` in a sequence equals `StepType.FIRST`
* final `TimeStep` in a sequence equals `StepType.LAST`
* All other `TimeStep`s in a sequence equal `StepType.MID`

#### Discounted rewards

> A discounting factor is introduced for:
* Reducing variance 
* Prescribing the effective time horizon we optimize over



## Notes from whitepaper

> [Top-𝐾 Off-Policy Correction for a REINFORCE Recommender System](https://arxiv.org/pdf/1812.02353.pdf)

**section 4.2 Estimating the behavior policy 𝛽**

* Despite a substantial sharing of parameters between the two policy heads 𝜋𝜃 and 𝛽𝜃′ , there are two noticeable differences between them: 
> * (1) While the main policy 𝜋𝜃 is effectively trained using a weighted softmax to take into account of long term reward, the behavior policy head 𝛽𝜃′ is trained using only the state-action pairs;
> * (2) While the main policy head 𝜋𝜃 is trained using only items on the trajectory with non-zero reward, the behavior policy 𝛽𝜃′ is trained using all of the items on the trajectory to avoid introducing bias in the 𝛽 estimate