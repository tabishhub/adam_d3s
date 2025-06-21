# Adam_d3s


## 🚀 Quick Start

1. **Download/Clone the Repository**
   - **On GitHub**: Click on `Code` → `Download ZIP`.
   - **On CLI**:
     ```bash
     git clone https://github.com/andrewlferguson/IMSI_LSS.git
     ```

3. **Running the Notebooks**
   - On Google Colab, go to `File` → `Upload notebook`
   - Local machine: run the notebook directly.
   - Execute each cell by:
     - Clicking the ▶️ play button on the left of each cell, or
     - Pressing `Shift + Enter`


## Table of Contents
- [Introduction](#introduction)
This repository contains the implementation of alternating gradient descent algorithms like **Adam** for learning dynamical systems from data. Key focus is on approximating the **Koopman operator** and applying **Sparse Identification of Nonlinear Dynamics (SINDy)**. The proposed algorithms are called **parametric EDMD** and **parametric SINDy**. The main benefit of the proposed algorithms is that we can use a parametric set of basis funtions instead of fixed basis functions like in **EDMD** and **SINDy**. We present application of the proposed algorithms on learning the Koopman operator for different stochastic dynamical systems and protein folding problem. The protein data is obtained from [`D.E. Shaw Research`](https://www.deshawresearch.com/resources.html). 

- [Usage](#usage)
This version of the implementation contains basic Jupyter notebooks for each experiment alongside the algorithm implmentation. You just have to navigate to the different folders and find experiments to run.

- [Results](#results)
Current experiments include the Ornstein-Uhlenbeck (OU) process, triple-well 2D potential Chignolin protein folding using the Koopman operator. For the parametric SINDy, we have the parametric Chua's circuit.

- [References](#references)
@article{tabish2024learning,
  title={Learning dynamical systems from data: Gradient-based dictionary optimization},
  author={Tabish, Mohammad and Chada, Neil K and Klus, Stefan},
  journal={arXiv preprint arXiv:2411.04775},
  year={2024}
}

Link of the paper [Adam_d3s](https://arxiv.org/abs/2411.04775)