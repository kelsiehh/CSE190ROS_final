#CSE 190 Final Project Q-learning

0. **Intention** -  Our team is interested in machine learning. Since the class introduces us to reinforcement learning which is fundamental to machine learning, we decided to implement reinforcement learning to get a step ahead on machine learning, and compare the resulting policies with the true optimal policies to analyze how much the robot successfully learns.

1. **Goal** - Implement Qlearning, which is a branch of model free reinforcement learning, and compare the resulting optimal policies with the true optimal policies given by MDP.

2. **Qlearning Implementation Details** - 
Formulas used:
(1) newQvalue: Qnew(S, a) = (1- α) * Qold(S, a) + α * [ R(S, a, S') + γ * maxa f(S', a)]
(2) Fvalue: f(S, a) = Q(S, a) + L(S, a)
(3) Lvalue: L(S', a) = C / N

(We need to include Fvalue while updating Qvalues because we want to make sure that not only the current state fully explores its neighbors, but also its neighbors fully explore their neighbors.)

- For each run:
	- pick a start state randomly on valid positions 
	- while not reaching the absorbing states:
		- choose an action in the current state with maximum Fvalues (using (2) to calculate the Fvalues)
		- execute the action to find the actual landing state of the robot, the reward for this step, and if it reaches the absorbing states (if the robot hits a wall, it doesn't move, but will return a reward for hitting wall)
		- Update the Qvalue of the current position's current action based on formula (1). If the robot reaches an absorbing state, maxa f(S', a) is just the reward for reaching that absorbing state. Otherwise, Loop through the landing state's Fvalues (calculated using formula (2)) and pick the maximum Fvalue.
		- After updating Qvalues, update the current state's chosen action's count for how many times the current position's that action is taken. This count is used to calculate Lvalue, where Lvalue indicates which direction of that position is not fully explored.
- After reaching the absorbing states for “max_iteration” times or if the difference of Qvalues between each run is smaller than “threshold_difference”, we stop Qlearning, and publish the resulting optimal policies.

3. **Results Analysis** - The optimal policies differ from that given by MDP: Around 15% is different.
This might because with Qlearning, it takes longer than MDP to converge. Since we use the same “max_iteration” for both Qlearning and MDP, and we stop at most max_iteration loops, it is likely that Qlearning doesn't fully converge while MDP converges. We think that this error rate is permissible.