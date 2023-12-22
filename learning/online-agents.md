# The online learning paradigm

*An Online Agent refines its policy by only using user’s (and system’s) feedback to its past predictions*

> TODO - add a TLDR here

## Supervised learning-based RecSys

**To better understand the need for Online RL in RecSys**, let's first describe the general interaction between users and a RecSys...

A typical supervised learning-based RecSys consists of the following parts:
* A `regression model` (linear or deep) that predicts a score for each (user, item) pair, and
* A `ranker` that ranks item based on the score of the regression model return top K items.

The interaction between users and the recommendation system is as follows
1. The **users see** a combination of those K products at the batch, and **give feedback** (click, like, etc) for the impressions
2. The system will **re-train** model based on the examples and users' feedback.
3. User's treatment will change when a **new model is pushed**.
4. The impressions will also affect user's latent state (e.g., preferences, future behavior, etc.)

![alt text](https://github.com/tottenjordan/tf_vertex_agents/blob/main/imgs/overview_sl_recsys.png)

We can think of the general **offline, supervised learning depolyment MLOps** like this:

1. Currently deployed model predicts (recommends) relevant candidates
2. Predictions and user feedback logged for future training
3. Predictions begin to deviate --> kick off MLOps retraining 
4. Train new model with most recent collected data
5. Deploy newly trained model
6. Newly deployed model serves predictions influenced by retraining procedure

## Motivation for Online learning in Contextual Bandits

From here, the **motivation for "online RL"** starts to become more clear, and it is mainly two fold:

1. batch, "offline" learning deployments are always behind, i.e., user behaviors and preferences are constantly changing
2. improve velocity of making future improvements, i.e., reduce cycle time of offline steps above

## Conceptual understanding of Online learning deployments

For "online learning" to take place, the agent's policy needs to be updated, where "updated" is conceptually similar to retraining a traditional supervised learning model (with latest training examples) and deploying the newly trained model

The bandit agent's policy is updated when the agent receives a trajectory that includes both the prediction/action AND the (often delayed) feedback from the pred/action

> TODO - insert visual

**Conceptually, we can think of these steps:**

1. prediction request sent to online bandit agent
2. bandit agent makes prediction (`pred_1`) given current policy (`policy_v1`)
3. delayed feedback for `pred_1` comes from user
4. trajectory for `pred_1`, including user feedback is fed to online bandit agent (in TF-Agents: ` agent.train()`)
5. Bandit agent's policy updated from `policy_v1` to `policy_v2`

Step (4) implies that our online bandit will be making at least two actions: `prediction` and `learning`. 
* Once our online agent trains on the most recent trajectory, the policy is updated (e.g., `policy_v2`). 
* As soon as you call `agent.train()` the policy is updated; its no longer `policy_v1` because the agent owns the policy and updates its weights

In this way we can see the improved **velocity** in which an Online Agent can learn from near-real time feedback and update its policy


### Challenges with online learning

* Real-time and even "near real-time" infra is difficult
* Delayed feedback
* Efficient bandits learning in real-time --> modeling user and context with a *good* tradeoff between accuracy and learning efficiency

> TODO


## Intersection of `Online Learning` and `on-policy vs off-policy` RL

> TODO