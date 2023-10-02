# Evolutionary Algorithms in Reinforcement Learning - Multi-objective Optimization in Inventory Management
### Project
- Motivation: Strike a balance between financial gains and transporation environmental impact of supply chain operations
- Goal: Identify the trade-off solutions (Pareto front)
- Key library: pymoo

### Supply Chain Network in this problem
<img width="314" alt="Screenshot 2023-10-02 at 11 32 29" src="https://github.com/yueqiu2/Multi-objective_SCM/assets/146023548/80fc7d07-a555-42a2-83e0-a82ee3338c2b">

### Methodology 
<img width="1326" alt="Screenshot 2023-10-02 at 00 55 32" src="https://github.com/yueqiu2/Multi-objective_SCM/assets/146023548/5c44749b-68b7-44fc-8907-8381a64cb810">

- Apply reinforcement learning framework
- Use multi-objective evolutionary algorithms (MOEAs) to optimize the policy net
- The MOEAs are: (1) NSGA-II (classic!), (2) AGE-MOEA (state-of-the-art).
- Use Bayesian optimization to smart tune hyperparameters of the MOEAs

### Result
#### Case 1: State formulation - Inventory level, backlog, unfulfilled order
<img width="773" alt="Screenshot 2023-10-02 at 11 33 38" src="https://github.com/yueqiu2/Multi-objective_SCM/assets/146023548/a149dbcc-3233-4d95-895d-f83bc8d65d0b">

- Converge within evaluation budget
- Well-defined Pareto front

#### Case 2 (when agent knows more): State formulation - Inventory level, backlog, unfulfilled order + Previous customer demand
<img width="764" alt="Screenshot 2023-10-02 at 11 33 57" src="https://github.com/yueqiu2/Multi-objective_SCM/assets/146023548/bfd5082a-a463-4a3c-877e-088ace827ded">

- **Pareto front with better diversity if the agent has more info about the environment!**

#### Investigation of NSGA-II hyperparameter: 
- (1) Ratio of number of offspring & population size
- (2) Ratio of population size & number of generation
<img width="719" alt="Screenshot 2023-10-02 at 11 39 58" src="https://github.com/yueqiu2/Multi-objective_SCM/assets/146023548/ac1cafe0-5826-4343-9572-fcb0af3da605">

#### Investigation of AGE-MOEA hyperparameter: 
- Ratio of population size & number of generation
<img width="354" alt="Screenshot 2023-10-02 at 11 39 34" src="https://github.com/yueqiu2/Multi-objective_SCM/assets/146023548/2d42b19c-0b2e-488f-acc9-23324b0252d5">

- **The hyperparameter ratios obtained by BO are the best (with highest hypervolume!**

### Summary
- Novel methodology works for this multi-objective optimization (MOO) problem of inventory management, the first to combine RL+MOO.
- BO can successfully fine-tune the hyperparameter
- But more to expand on methodological front and supply chain environment setting.
