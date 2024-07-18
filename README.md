
# Bridging the Sim-to-Real Gap in Reinforcement Learning: A Comparative Analysis of Algorithms on the OpenAI Gym Hopper Problem

## Overview

This project presents an introductory analysis of reinforcement learning (RL) algorithms on the OpenAI Gym Hopper problem, focusing on addressing the sim-to-real gap. We implemented and evaluated various RL algorithms, including REINFORCE, Actor-Critic (A2C), and Proximal Policy Optimization (PPO). Our results indicate that while REINFORCE and A2C struggled to solve the Hopper task, PPO demonstrated effective performance. We further enhanced generalization and performance using Uniform Domain Randomization (UDR) and the DROPO off-policy method to identify optimal environment parameters.

## Authors

- Carlo Marra (s334220)
- Giovanni Pellegrino (s331438)
- Alessandro Valenti (s328131)

## Abstract

This project focuses on the evaluation of various reinforcement learning algorithms on the OpenAI Gym Hopper problem. Our results show that while REINFORCE and A2C had limited success, PPO proved to be effective. We also applied Uniform Domain Randomization (UDR) to improve generalization and used the DROPO off-policy method to identify optimal environment parameters, effectively minimizing the sim-to-real gap.

## Introduction

Reinforcement Learning (RL) is a machine learning paradigm where an agent learns to make decisions by interacting with an environment to maximize cumulative rewards. In our project, we utilized custom Hopper environments based on the original OpenAI Gym Hopper. We created two custom environments: "source" for training and "target" for evaluation, with the key difference being the mass of the torso.

## Tasks

- Task 2: REINFORCE
- Task 3: Actor Critic
- Task 4: PPO
- Task 5: Train and Test of the policies
- Task 6: UDR
- Task 7: DROPO (Project Extension)

## Algorithms

### REINFORCE

The REINFORCE algorithm is a Monte Carlo policy gradient method used for optimizing stochastic policies. Despite improvements using a baseline, REINFORCE alone was insufficient to solve the Hopper environment due to its complexity.

### Actor-Critic (A2C)

The Actor-Critic algorithm consists of two components: the Actor, which updates the policy, and the Critic, which evaluates the actions. Despite some improvements with an entropy coefficient to encourage exploration, A2C failed to solve the Hopper task effectively.

### Proximal Policy Optimization (PPO)

PPO is a state-of-the-art RL algorithm designed to improve policy stability and performance by limiting the magnitude of policy updates. Through hyperparameter tuning and extensive training, PPO successfully solved the Hopper task, demonstrating its effectiveness.

## Results

- **REINFORCE:** Introduction of a baseline improved learning but was insufficient to solve the Hopper task.
- **Actor-Critic (A2C):** Improved stability and efficiency but failed to produce effective policies due to specification gaming issues.
- **PPO:** Successfully solved the Hopper task with effective performance after hyperparameter tuning.

## Conclusion

PPO outperformed REINFORCE and A2C in solving the OpenAI Gym Hopper problem. By employing UDR and DROPO methods, we improved generalization and minimized the sim-to-real gap, highlighting the potential of these approaches in real-world applications.

```

## References

- Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction.
- Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
