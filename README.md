# MSM – Markov State Model for IDP–Ligand Binding

This repository contains code used to build and analyze **Markov State Models (MSMs)** describing interaction pathways between small molecules and intrinsically disordered proteins (IDPs).

The methodology is described in:

Louet et al.  
**Binding Paths: A Markov state model framework for small-molecule interactions with IDPs**  
https://www.biorxiv.org/content/10.1101/2025.11.19.688850v1

---

## Main Script

The MSM model and network graphs are generated with:

`community_identification/run_auto.py`

This script:

- builds the **Markov State Model**
- constructs the **transition matrix**
- performs **community identification**
- generates **network graphs of states and transitions**

## Repository Structure

**community_identification**  
- `run_auto.py` — main MSM generation pipeline  
- `clustering_uplets.py` — clustering of interaction patterns  
- `kd_calculation.py` — kinetic / Kd calculations  
- `population_equilibrium.py` — equilibrium population calculations  

**VAMPNET**  
- `vampnet_2_iterations.py` — VAMPNet comparison model  
- `ami_score.py` — clustering comparison metrics  

**chapman_kolmogorov_test**  
- `enumeration_ck_test.py` — MSM validation (Chapman–Kolmogorov test)  

**matrix_handoff**  
- `matrix_handoff.py` — transition matrix utilities  

**visualization**  
- `visualizing_clusters.py` — plotting and network visualization
