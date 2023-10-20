# Design in TF-Agents

### Environment 
* The `environment` represents the user and the agent is the recommender system. 
* This is a POMDP because we don’t know the user’s actual internal state s_t. 
* We have observations from the user o_t which include things like clicks (item selected), likes etc. 
* The rewards could be likes, watch time etc. The actions of the recommender are a set A_t = {a1, … ak} of k items recommended at a time. 
* Typically implemented as Markov Decision Processes (MPD), and they give us the flexibility to model a *distribution of actions*

### Policy
* The main job of the Policy is to map observations from the user to actions. 
* The action is a set of K recommended items. The policy is created by the Agent and contains a reference to the Network: 

```
Policy.__init__(time_step_spec, action_spec, network, ...)
```
### Trajectory
* batch of tensors representing `observations`, `actions`, `rewards`, `discounts`