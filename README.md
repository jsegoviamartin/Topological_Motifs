# Topological Motifs in Evolutionary Games
### Comparing Cooperation on 2-D Lattices and Random-Regular Graphs

This repository contains the code and data used in the study  
[*Topological Motifs in Evolutionary Games: Comparing Cooperation on 2-D Lattices and Random-Regular Graphs*](https://github.com/jsegoviamartin/Topological_Motifs).

The repository is organised into three sections, corresponding to Sections 3.1–3.3 of the manuscript:

---

## 📂 1. Reproduction Nowak and May (Section 3.1)
- Python implementation of the canonical Prisoner’s Dilemma with **unconditional imitation**.
- Reproduces the cooperation profile corresponding to the seminal work of Nowak & May (1992) and reviewed in Roca *et al.* (2009, see Fig. 10).
- Useful for verifying the baseline dynamics before extending to other networks.

---

## 📂 2. Cooperation Profiles (Section 3.2)
- Python scripts to generate cooperation profiles and cluster counts on:
  - **Square lattice**  
  - **Regular random (RR) network**
- Additional implementations for **Erdős–Rényi** and **Barabási–Albert** graphs are provided in the code (not analysed in the manuscript).
- Produces time series, cooperation curves, and comparative plots.

---

## 📂 3. Cluster Identification (Section 3.3)
- Python files for running simulations and collecting:
  - Cooperation fractions  
  - Counts of cooperators, defectors, oscillators  
  - Stable and unstable clusters  
  - Average clustering coefficient
- Includes tools to generate **network visualisations** and **donut plots** of cluster composition.

---

## 🛠 Requirements
- Python 3.7+  
- `numpy`, `networkx`, `matplotlib`, `PIL`, `ast`, `pandas`

---

## ▶️ Usage
Each section contains runnable Python scripts. Example usage:

```bash
cd "Cooperation profiles"
python run_lattice.py
