
# Summary
|Chemical systems|Element ratios|# of properties|# of configurations|# of atoms|
|---|---|---|---|---|
|AlNiTi, AlNi, Al, AlTi, Ni, NiTi, Ti|Ni (38.3%), Ti (32.1%), Al (29.6%)|2684|2666|24851|

# Name

AlNiTi_CMS2019

# Authors

K. Gubaev

E. V. Podryabinkin

G. L. W. Hart

A. V. Shapeev

# Links

https://www.sciencedirect.com/science/article/pii/S0927025618306372?via%3Dihub

https://gitlab.com/kgubaev/accelerating-high-throughput-searches-for-new-alloys-with-active-learning-data

# Description

This dataset was generated using the following active learning scheme: 1) candidate structures relaxed by a partially-trained MTP model, 2) structures for which the MTP had to perform extrapolation are passed to DFT to be re-computed, 3) the MTP is retrained included the structures that were re-computed with DFT, 4) steps 1-3 are repeated until the MTP does not extrapolate on any of the original candidate structures. The original candidate structures for this dataset included about 375,000 binary and ternary structures enumerating all possible unit cells with different symmetries (BCC, FCC, and HCP) and different number of atoms

# Storage format

|Elements|File|Format|Name field|
|---|---|---|---|
| Ni, Ti, Al | 8024633491454427016 | mongo | _name |

# Properties

|Property|KIM field|ASE field|Units
|---|---|---|---|
| [energy-forces-stress](/home/josh/colabfit-tools/colabfit/examples/AlNiTi_CMS2019/energy-forces-stress.edn) | energy | energy | eV
| [energy-forces-stress](/home/josh/colabfit-tools/colabfit/examples/AlNiTi_CMS2019/energy-forces-stress.edn) | forces | forces | eV/Ang
| [energy-forces-stress](/home/josh/colabfit-tools/colabfit/examples/AlNiTi_CMS2019/energy-forces-stress.edn) | stress | stress | GPa

# Property settings

|ID|Method|Description|Labels|Files|
|---|---|---|---|---|
| 2231808486057270801 | VASP | energies/forces/stresses |  |  |

# Configuration sets

|ID|Description|# of structures| # of atoms|
|---|---|---|---|
| 6873314056031893128 | Configurations generated using active learning by iteratively fitting a MTP model, identifying configurations that required the MTP to extrapolate, re-computing the energies/forces/structures of those configurations with DFT, then retraining the MTP model. | 2666 | 24851 |
| 2025718299560966014 | Configurations used in the first stage of training | 1706 | 16186 |
| -1964656355228270700 | Configurations used in the second stage of training | 963 | 8669 |

# Configuration labels

|Labels|Counts|
|---|---|
| active_learning | 2666 |

# Figures
![The results of plot_histograms](histograms.png)
