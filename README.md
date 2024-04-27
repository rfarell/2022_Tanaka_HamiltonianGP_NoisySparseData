# Symplectic Spectrum Gaussian Processes (2022 Tanaka, Iwata & Ueda)

This repository is dedicated to a comprehensive summary, analysis, and implementation of the methodologies presented in the paper "Symplectic Spectrum Gaussian Processes: Learning Hamiltonians from Noisy and Sparse Data" by Yusuke Tanaka, Tomoharu Iwata, and Naonori Ueda, published in the 2022 proceedings of the Advances in Neural Information Processing Systems (NeurIPS). The original paper can be accessed [here](https://openreview.net/forum?id=W4ZlZZwsQmt).

## Repository Structure

```
2022_Tanaka_LearningHamiltoniansFromNoisyData
├── 00_visualize_system.py
├── 01_create_dataset.py
├── 02_visualize_dataset.py
├── 03_train_model.py
├── 04_visualize_results.py
├── LICENSE
├── README.md
├── my_model.py
├── my_system.py
```


After running all the code, the extra files that will be enerated are
```
2022_Tanaka_LearningHamiltoniansFromNoisyData
├── best.pth.tar
├── losses.png
├── pendulum_animation.mp4
├── pendulum_test_trajectories.pth
├── pendulum_train_trajectories.png
└── pendulum_train_trajectories.pth
```

## Paper Summary

The paper by Tanaka, Iwata, and Ueda introduces Symplectic Spectrum Gaussian Processes (SSGP), a novel method designed to learn Hamiltonian dynamics from noisy and sparse observational data. This approach leverages the properties of Hamiltonian systems and the theory behind Gaussian processes to effectively reconstruct the underlying dynamics of complex systems, even with limited data inputs.

For a more detailed summary and analysis, please refer to `paper_summary.md` in this repository.

## Implementation and Code

