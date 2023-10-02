# Evolutionary Algorithms in Reinforcement Learning - Multi-objective Optimization in Inventory Management
### Project
- Motivation: Strike a balance between financial gains and transporation environmental impact of supply chain operations
- Goal: Identify the trade-off solutions (Pareto front)

### Methodology 
<img width="1326" alt="Screenshot 2023-10-02 at 00 55 32" src="https://github.com/yueqiu2/Multi-objective_SCM/assets/146023548/5c44749b-68b7-44fc-8907-8381a64cb810">
- Apply reinforcement learning framework
- Use multi-objective evolutionary algorithms (MOEAs) to optimize the policy net
- The MOEAs are: (1) NSGA-II (classic!), (2) AGE-MOEA (state-of-the-art).
- Use Bayesian optimization to smart tune hyperparameters of the MOEAs

### Result
#### Case 1: State formulation - Inventory level, backlog, unfulfilled order
![MicrosoftTeams-image copy](https://github.com/yueqiu2/Multi-objective_SCM/assets/146023548/ab3342b3-14ee-4c88-a089-b40460361bed)
- Converge within evaluation budget
- Well-defined Pareto front

#### Case 2 (when agent knows more): State formulation - Inventory level, backlog, unfulfilled order + Previous customer demand
![MicrosoftTeams-image (1) copy](https://github.com/yueqiu2/Multi-objective_SCM/assets/146023548/5be9924a-e9df-4b98-94b9-486eae3972fb)
- **Pareto front with better diversity if the agent has more info about the environment!**

### Summary
Novel methodology works for this multi-objective optimization problem! but with way more to expand on methodological front and supply chain environment setting.
