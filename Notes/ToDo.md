# ToDo

## Task 3
- [ ] Train Base A2C for 2500 episodes
- [ ] Train A2C with entropy = 0.1 for 2500 episodes
- [ ] Train Base A2C with entropy = 0.1 and lower alive reward for 2500 episodes

## Task 4
- [x] Check to see if Default Saved Model is on 500_000 ts

## Task 5
- [x] Source Hyperparamer Tuning (250_000 ts)
  1) a8z1tsrg: 1076.443 $\plusmn$ 0.610
  2) n06zwey6: 1250.197 $\plusmn$ 4.380
  3)  ajm85y7j: 1059.368 $\plusmn$ 4.590
  4)  bh8ero5k: 1044.853 $\plusmn$ 0.810
  5)  **0hmxm2fs**: 1657.577 $\plusmn$ 31.660

- [x] Target Hyperparamer Tuning (250_000 ts) 
  1) gedhu3vt: 1054.739 $\plusmn$ 0.620
  2) x6cg1s5i: 1109.542 $\plusmn$ 5.040
  3) **f79be73v**: 1470.484 $\plusmn$ 162.070
  4) blxbaz4c: 1404.382 $\plusmn$ 49.130
  5) 9len8kfk: 964.165 $\plusmn$ 22.02

- [x] 250_000 ts on Best Source Model (0hmxm2fs)
- [x] 250_000 ts on Best Target Model (f79be73v)
- [x] Source $\rightarrow$ Source Test
    - Final Result: 1743.206 $\plusmn$ 6.540
- [x] Source $\rightarrow$ Target Test (**Lower Bound**)
    - Final Result: 1145.021 $\plusmn$ 23.760
- [x] Target $\rightarrow$ Target Test (**Upper Bound**)
    - Final Result: 1689.101 $\plusmn$ 2.220

## Task 6
   - [ ] Train Best Source HyperParameters Model with UDR:
     - 2_000_000 ts (StopTrainingOnNoModelImprovement) for:
       - [ ] UDR: 0.10
       - [ ] UDR: 0.25
       - [ ] UDR: 0.50 
   - [ ] UDR Source $\rightarrow$ Target Test (**New Lower Bound**)
