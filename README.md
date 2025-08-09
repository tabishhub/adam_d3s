# Adam_d3s

## Table of Contents
- [Introduction](#introduction)
- [Usage](#usage)
- [Results](#results)
- [System Dynamics](#system-dynamics)
- [Quick Guide](#quick-guide)
- [References](#references)

---

## 📘 Introduction

This repository contains the implementation of our recently proposed alternating gradient descent algorithms like **Alternating Adam** for learning dynamical systems from data.

Our focus is on approximating the **Koopman operator** and applying **Sparse Identification of Nonlinear Dynamics (SINDy)** using a set of **parametric basis functions**. These methods extend traditional **EDMD** and **SINDy** by optimizing over basis functions rather than using fixed ones.

### Key Highlights:
- Learn **data-driven dynamical systems** using parametric models.
- Use gradient-based optimization with alternating minimization (Adam).
- Apply to **stochastic systems**, **chaotic circuits**, and **protein folding**.
- Demonstrated on datasets like the **Chignolin protein** from [`D. E. Shaw Research`](https://www.deshawresearch.com/resources.html).

---

## ⚙️ Usage

This repository is organized by experiment. Each folder includes:
- A Jupyter notebook
- Model training code
- Visualization scripts

You can explore models and results by opening the notebooks inside each experiment folder.

---

## 📊 Results

The following systems are modeled using either the Koopman operator or parametric SINDy:

- ✅ **Ornstein-Uhlenbeck (OU) process**
- ✅ **Triple-well potential (2D)**
- ✅ **Chignolin protein folding**
- ✅ **Chua’s circuit**
- ✅ **Nonlinear heat equation**

Each demonstrates improved performance using parametric basis learning over classical fixed dictionaries.

---

## 📐 System Dynamics

### 🔌 Chua's Circuit

The dynamics of Chua’s circuit are governed by the following system of ODEs:

```latex
\dot{x}_1 = \alpha \left(x_2 - x_1 - f(x_1)\right) \\
\dot{x}_2 = \frac{1}{R C_2} \left(x_1 - x_2 + R x_3\right) \\
\dot{x}_3 = -\beta x_2
```

with the nonlinear function:

```latex
f(x_1) = -b \sin\left(\frac{\pi x_1}{a} + d\right)
```

---

### 🌡️ Nonlinear Heat Equation 

Adapted from [this paper](https://arxiv.org/abs/1811.06337), the PDE is:

```latex
\rho c_p \frac{\partial u}{\partial t} = \frac{\partial}{\partial x} \left( \kappa(u) \frac{\partial u}{\partial x} \right)
```

Expanded using the chain rule:

```latex
\rho c_p \frac{\partial u}{\partial t} = \frac{\partial \kappa(u)}{\partial u} \left( \frac{\partial u}{\partial x} \right)^2 + \kappa(u) \frac{\partial^2 u}{\partial x^2}
```

---

## 🚀 Quick Guide

### 1. Download or Clone the Repository

```bash
git clone https://github.com/tabishhub/data_driven_dynamical_systems.git
```

Or use the **Download ZIP** button from the GitHub interface.

---

### 2. Run the Jupyter Notebooks

You can run experiments in either Google Colab or your local Python environment.

#### 🔁 On Google Colab:
- Open [Google Colab](https://colab.research.google.com/)
- Click `File → Upload Notebook`
- Upload and run any notebook from this repo

#### 💻 On Local Machine:
- Make sure you have Python 3.8+ and Jupyter installed.
- Navigate to the notebook's folder and run:

```bash
jupyter notebook
```

Then open the desired `.ipynb` file and run cells using `Shift + Enter`.

---

## 📚 References

### 📄 Citation

If you use this work, please cite the following:

```bibtex
@article{TABISH2025134822,
title = {Learning dynamical systems from data: Gradient-based dictionary optimization},
author = {Mohammad Tabish and Neil K. Chada and Stefan Klus},
journal = {Physica D: Nonlinear Phenomena},
volume = {481},
pages = {134822},
year = {2025},
issn = {0167-2789},
doi = {https://doi.org/10.1016/j.physd.2025.134822},
url = {https://www.sciencedirect.com/science/article/pii/S0167278925002994},
}
```

### 🔗 Paper Link

- [DOI](https://doi.org/10.1016/j.physd.2025.134822)

---

## 🙌 Acknowledgements

Protein folding data used in this work is sourced from [`D. E. Shaw Research`](https://www.deshawresearch.com/resources.html). We thank the authors and maintainers of the datasets.

---
