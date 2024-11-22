# Online learning with Contextual Bandits

**TODOs**
* run these from new instance with GPU etc.
* organize and consolidate notebooks 


## Generalized Policy Iteration (GPI)

> If not familiar with GPI, refer to [chapter 4](http://www.incompleteideas.net/book/ebook/node40.html) of Sutton and Barto’s book, [Reinforcement Learning: an Introduction](http://www.incompleteideas.net/book/ebook/the-book.html)

<p align="center">
    <img src='imgs/gpi.png' width='855' height='262' />
</p>

* GPI refers to a general RL framework that uses value functions to organize and structure the search for better policies
* Specifically, GPI describes two interacting processes: (i) [policy evaluation](http://www.incompleteideas.net/book/ebook/node41.html) and (ii) [policy improvement](http://www.incompleteideas.net/book/ebook/node42.html), that eventually converge on the optimal policy and value functions as an agent interacts with an environment


## the Training is *in* the server?

<p align="center">
    <img src='imgs/in_process_learning_RA.png' width='1200' />
</p>