
# Summary
|Chemical systems|Element ratios|# of properties|# of configurations|# of atoms|
|---|---|---|---|---|
|Ta|Ta (100.0%)|3775|3775|45439|

# Name

Ta_PRM2019

# Authors

J. Byggmästar

K. Nordlund

F. Djurabekova

# Links

https://journals.aps.org/prmaterials/abstract/10.1103/PhysRevMaterials.4.093802https://gitlab.com/acclab/gap-data/-/tree/master/Ta

# Description

This dataset was designed to ensure machine-learning of Ta elastic, thermal, and defect properties, as well as surface energetics, melting, and the structure of the liquid phase. The dataset was constructed by starting with the dataset from J. Byggmästar et al., Phys. Rev. B 100, 144105 (2019), then rescaling all of the configurations to the correct lattice spacing and adding in gamma surface configurations.

# Storage format

|Elements|File|Format|Name field|
|---|---|---|---|
| Ta | -2817115958536634477 | mongo | _name |

# Properties

|Property|KIM field|ASE field|Units
|---|---|---|---|
| [energy-forces-stress](/home/jvita/scripts/colabfit-tools/colabfit/examples/Ta_PRM2019/energy-forces-stress.edn) | energy | energy | eV
| [energy-forces-stress](/home/jvita/scripts/colabfit-tools/colabfit/examples/Ta_PRM2019/energy-forces-stress.edn) | forces | forces | eV/Ang
| [energy-forces-stress](/home/jvita/scripts/colabfit-tools/colabfit/examples/Ta_PRM2019/energy-forces-stress.edn) | stress | stress | GPa

# Property settings

|ID|Method|Description|Labels|Files|
|---|---|---|---|---|
| 2231808486057270801 | VASP | energies/forces/stresses | GGA, PBE, LDA |  |

# Configuration sets

|ID|Description|# of structures| # of atoms|
|---|---|---|---|
| -8982816651304268686 | Configurations designed to ensure machine-learning of elastic, thermal, and defect properties, as well as surface energetics, melting, and the structure of the liquid phase. | 3775 | 45439 |
| 4745391250136382420 | Liquid configurations with densities around the experimental density | 45 | 5760 |
| 3749751851580194892 | Configurations with single self-interstitial defects | 32 | 3872 |
| 2652955360017592045 | Single-vacancy configurations | 210 | 11130 |
| 1052227234302280776 | A15 configurations with random lattice distortions | 100 | 800 |
| 1619681722447208960 | BCC configurations with random strains up to +/- 30% to help train the far-from-equilibrium elastic response | 546 | 1092 |
| 4300924259814916589 | C15 configurations with random lattice distortions | 100 | 600 |
| 2661857600540254291 | Configurations with two self-interstitial defects | 13 | 2106 |
| 7012349981954511692 | Divacancy configurations | 10 | 1180 |
| 3967086951955858726 | Diamond configurations with random lattice distortions | 100 | 200 |
| -1252197385126982586 | Dimers to fit to the full dissociation curve starting from 1.1 angstrom | 14 | 28 |
| -936995604349397505 | FCC crystals with random lattice distortions | 100 | 100 |
| 4350617082744801866 | Configurations representing the full gamma surface | 178 | 2136 |
| -6209793569497754614 | HCP configurations with random lattice distortions | 100 | 200 |
| -8946848860633940618 | Isolated W atom | 1 | 1 |
| -3085976378011225183 | MD snapshots taken at 1000K for three different volumes | 50 | 2700 |
| 7647645685667182496 | Simple cubic crystals with random lattice distortions | 100 | 100 |
| -3072678575497456065 | BCC crystals with random interstitial atom defects to capture short-range many-body dynamics | 86 | 4644 |
| -5461758645436659837 | Randomly distorted primitive bcc unit cells drawn from Szlachta et al.'s database | 1776 | 1776 |
| 632702784321209060 | Damaged and half-molten (110) and (100) surfaces | 24 | 3264 |
| -4871019023155330830 | Configurations with single self-interstitial defects | 45 | 540 |
| 1244850196469881493 | (110) surface configurations | 45 | 540 |
| 9159056700276447537 | (111) surface configurations | 41 | 492 |
| 2569349629169076921 | (112) surface configurations | 45 | 540 |
| 2097726978311304540 | Trivacancy configurations | 14 | 1638 |

# Configuration labels

|Labels|Counts|
|---|---|
| bcc | 2408 |
| strain | 2822 |
| interstitial | 131 |
| warning | 100 |
| large_forces | 100 |
| repulsive | 100 |
| vacancy | 234 |
| divacancy | 24 |
| surface | 378 |
| liquid | 69 |
| gamma_surface | 178 |
| c15 | 100 |
| hcp | 100 |
| a15 | 100 |
| diamond | 115 |
| fcc | 100 |
| sc | 100 |
| aimd | 50 |
| dimer | 14 |
| sh | 10 |
| trivacancy | 14 |
| isolated_atom | 1 |

# Figures
![The results of plot_histograms](histograms.png)
