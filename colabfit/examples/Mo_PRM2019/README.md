
# Summary
|Chemical systems|Element ratios|# of properties|# of configurations|# of atoms|
|---|---|---|---|---|
|Mo|Mo (100.0%)|3785|3785|45667|

# Name

Mo_PRM2019

# Authors

J. Byggmästar

K. Nordlund

F. Djurabekova

# Links

https://journals.aps.org/prmaterials/abstract/10.1103/PhysRevMaterials.4.093802

https://gitlab.com/acclab/gap-data/-/tree/master/Mo

# Description

This dataset was designed to ensure machine-learning of Mo elastic, thermal, and defect properties, as well as surface energetics, melting, and the structure of the liquid phase. The dataset was constructed by starting with the dataset from J. Byggmästar et al., Phys. Rev. B 100, 144105 (2019), then rescaling all of the configurations to the correct lattice spacing and adding in gamma surface configurations.

# Storage format

|Elements|File|Format|Name field|
|---|---|---|---|
| Mo | 2669550080465425079 | mongo | _name |

# Properties

|Property|KIM field|ASE field|Units
|---|---|---|---|
| [energy-forces-stress](/home/jvita/scripts/colabfit-tools/colabfit/examples/Mo_PRM2019/energy-forces-stress.edn) | energy | energy | eV
| [energy-forces-stress](/home/jvita/scripts/colabfit-tools/colabfit/examples/Mo_PRM2019/energy-forces-stress.edn) | forces | forces | eV/Ang
| [energy-forces-stress](/home/jvita/scripts/colabfit-tools/colabfit/examples/Mo_PRM2019/energy-forces-stress.edn) | stress | stress | GPa

# Property settings

|ID|Method|Description|Labels|Files|
|---|---|---|---|---|
| 2231808486057270801 | VASP | energies/forces/stresses | LDA, GGA, PBE |  |

# Configuration sets

|ID|Description|# of structures| # of atoms|
|---|---|---|---|
| -6879005969428655404 | Configurations designed to ensure machine-learning of elastic, thermal, and defect properties, as well as surface energetics, melting, and the structure of the liquid phase. | 3785 | 45667 |
| 5581358389494786641 | Liquid with densities around the experimental density of 17.6 g/cm^3 | 45 | 5760 |
| 1655338112405619486 | Configurations with single self-interstitial defects | 32 | 3872 |
| 8750790239854870991 | Single-vacancy configurations | 210 | 11130 |
| 7645957540170849107 | A15 configurations with random lattice distortions | 100 | 800 |
| 5793002635583525694 | BCC configurations with random strains up to +/- 30% to help train the far-from-equilibrium elastic response | 547 | 1094 |
| -7500200451517309708 | C15 configurations with random lattice distortions | 100 | 600 |
| 7786299080462930239 | Configurations with two self-interstitial defects | 13 | 2106 |
| -3126071615879835914 | Divacancy configurations | 10 | 1180 |
| 8122141655150353037 | Diamond configurations with random lattice distortions | 100 | 200 |
| 8779203562000600561 | Dimers to fit to the full dissociation curve starting from 1.1 angstrom | 19 | 38 |
| -854031477752715945 | FCC crystals with random lattice distortions | 100 | 100 |
| -42894441245076817 | Configurations representing the full gamma surface | 178 | 2136 |
| -5898556728539397440 | HCP configurations with random lattice distortions | 100 | 200 |
| -1988006184926480442 | Isolated W atom | 1 | 1 |
| 5359046533315605074 | MD snapshots taken at 1000K for three different volumes | 50 | 2700 |
| -7184822891142729249 | Simple cubic crystals with random lattice distortions | 100 | 100 |
| -7371954082514867913 | BCC crystals with random interstitial atom defects to capture short-range many-body dynamics | 90 | 4860 |
| 3624173460003986235 | Randomly distorted primitive bcc unit cells drawn from Szlachta et al.'s database | 1776 | 1776 |
| -5385407476537885739 | Damaged and half-molten (110) and (100) surfaces | 24 | 3264 |
| 4203371701417391308 | Configurations with single self-interstitial defects | 45 | 540 |
| -5004714194623343308 | (110) surface configurations | 45 | 540 |
| 3936526891413695858 | (111) surface configurations | 41 | 492 |
| 8311622814909245786 | (112) surface configurations | 45 | 540 |
| 5196039886009526574 | Trivacancy configurations | 14 | 1638 |

# Configuration labels

|Labels|Counts|
|---|---|
| bcc | 2413 |
| strain | 2823 |
| surface | 378 |
| vacancy | 234 |
| c15 | 100 |
| interstitial | 135 |
| warning | 109 |
| large_forces | 109 |
| repulsive | 109 |
| gamma_surface | 178 |
| liquid | 69 |
| fcc | 100 |
| a15 | 100 |
| aimd | 50 |
| diamond | 100 |
| sc | 100 |
| hcp | 100 |
| dimer | 19 |
| divacancy | 24 |
| trivacancy | 14 |
| isolated_atom | 1 |

# Figures
![The results of plot_histograms](histograms.png)
