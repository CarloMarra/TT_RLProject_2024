# ToDo

## Task 2
### Training
- [ ] Train REINFORCE (baseline = 0) for 2500 episodes
- [x] Train REINFORCE (baseline = 20) for 2500 episodes
- [ ] Train REINFORCE (baseline = 50) for 2500 episodes
- [ ] Train REINFORCE (baseline = 70) for 2500 episodes
- [ ] Train REINFORCE (baseline = -25) for 2500 episodes (?)

### Testing
- [ ] Test REINFORCE (baseline = 0): ...
- [ ] Test REINFORCE (baseline = 20): ...
- [ ] Test REINFORCE (baseline = 50): ...
- [ ] Test REINFORCE (baseline = 70): ...
- [ ] Test REINFORCE (baseline = -25): ... (?)

## Task 3
- [x] Train Base A2C for 2500 episodes
- [x] Train A2C with entropy = 0.1 for 2500 episodes
- [x] Train Base A2C with entropy = 0.1 and lower alive reward for 2500 episodes
- [x] Test Base A2C: 279.449 $\pm$ 3.284
- [x] Test A2C with entropy = 0.1: 474.668 $\pm$ 0.702
- [x] Test A2C with entropy = 0.1 and lower alive reward: 551.684 $\pm$ 0.511

## Task 4
- [x] Check to see if Default Saved Model is on 500_000 ts

## Task 5
- [x] Source Hyperparamer Tuning (250_000 ts)
  1) a8z1tsrg: 1076.443 $\pm$ 0.610
  2) n06zwey6: 1250.197 $\pm$ 4.380
  3)  ajm85y7j: 1059.368 $\pm$ 4.590
  4)  bh8ero5k: 1044.853 $\pm$ 0.810
  5)  **0hmxm2fs**: 1657.577 $\pm$ 31.660

- [x] Target Hyperparamer Tuning (250_000 ts) 
  1) gedhu3vt: 1054.739 $\pm$ 0.620
  2) x6cg1s5i: 1109.542 $\pm$ 5.040
  3) **f79be73v**: 1470.484 $\pm$ 162.070
  4) blxbaz4c: 1404.382 $\pm$ 49.130
  5) 9len8kfk: 964.165 $\pm$ 22.02

- [x] 250_000 ts on Best Source Model (0hmxm2fs)
- [x] 250_000 ts on Best Target Model (f79be73v)
- [x] Source $\rightarrow$ Source Test
    - Final Result: 1743.206 $\pm$ 6.540
- [x] Source $\rightarrow$ Target Test (**Lower Bound**)
    - Final Result: 1145.021 $\pm$ 23.760
- [x] Target $\rightarrow$ Target Test (**Upper Bound**)
    - Final Result: 1689.101 $\pm$ 2.220

## Task 6
   - [ ] Train Best Source HyperParameters Model with UDR:
     - 2_000_000 ts (StopTrainingOnNoModelImprovement) for:
       - [ ] UDR: 0.10
       - [ ] UDR: 0.25
       - [ ] UDR: 0.50 
   - [ ] UDR Source $\rightarrow$ Target Test (**New Lower Bound**)
