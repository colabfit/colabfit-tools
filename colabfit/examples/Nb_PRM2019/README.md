
# Summary
|Chemical systems|Element ratios|# of properties|# of configurations|# of atoms|
|---|---|---|---|---|
|Nb|Nb (100.0%)|3787|3787|45641|

# Name

Nb_PRM2019

# Authors

J. Byggmästar

K. Nordlund

F. Djurabekova

# Links

https://journals.aps.org/prmaterials/abstract/10.1103/PhysRevMaterials.4.093802https://gitlab.com/acclab/gap-data/-/tree/master/Nb

# Description

This dataset was designed to ensure machine-learning of Nb elastic, thermal, and defect properties, as well as surface energetics, melting, and the structure of the liquid phase. The dataset was constructed by starting with the dataset from J. Byggmästar et al., Phys. Rev. B 100, 144105 (2019), then rescaling all of the configurations to the correct lattice spacing and adding in gamma surface configurations.

# Storage format

|Elements|File|Format|Name field|
|---|---|---|---|
| Nb | 1479475274877589533 | mongo | _name |

# Properties

|Property|KIM field|ASE field|Units
|---|---|---|---|
| [energy-forces-stress](/home/jvita/scripts/colabfit-tools/colabfit/examples/Nb_PRM2019/energy-forces-stress.edn) | energy | energy | eV
| [energy-forces-stress](/home/jvita/scripts/colabfit-tools/colabfit/examples/Nb_PRM2019/energy-forces-stress.edn) | forces | forces | eV/Ang
| [energy-forces-stress](/home/jvita/scripts/colabfit-tools/colabfit/examples/Nb_PRM2019/energy-forces-stress.edn) | stress | stress | GPa

# Property settings

|ID|Method|Description|Labels|Files|
|---|---|---|---|---|
| 2231808486057270801 | VASP | energies/forces/stresses | PBE, LDA, GGA |  |

# Configuration sets

|ID|Description|# of structures| # of atoms|
|---|---|---|---|
| 8751040679407895386 | Configurations designed to ensure machine-learning of elastic, thermal, and defect properties, as well as surface energetics, melting, and the structure of the liquid phase. | 3787 | 45641 |
| -1596621928581199676 | Liquid configurations with densities around the experimental density | 45 | 5760 |
| 2610670258562758959 | Configurations with single self-interstitial defects | 32 | 3872 |
| 3035246810945411528 | Single-vacancy configurations | 200 | 10600 |
| -8295294327591310829 | A15 configurations with random lattice distortions | 100 | 800 |
| -1031645312498663118 | BCC configurations with random strains up to +/- 30% to help train the far-from-equilibrium elastic response | 547 | 1094 |
| -6369461220721652 | C15 configurations with random lattice distortions | 100 | 600 |
| 2971172782395454507 | Configurations with two self-interstitial defects | 13 | 2106 |
| -4604584220454448374 | Divacancy configurations | 10 | 1180 |
| 4263255223391369792 | Diamond configurations with random lattice distortions | 100 | 200 |
| 3831339198838380579 | Dimers to fit to the full dissociation curve starting from 1.1 angstrom | 24 | 48 |
| -4936744130997799309 | FCC crystals with random lattice distortions | 100 | 100 |
| -8679208073246478903 | Configurations representing the full gamma surface | 178 | 2136 |
| 5948088804675292308 | HCP configurations with random lattice distortions | 100 | 200 |
| 8887751376090547750 | Isolated W atom | 1 | 1 |
| 3119562324801669586 | MD snapshots taken at 1000K for three different volumes | 50 | 2700 |
| -2501518307430232349 | Simple cubic crystals with random lattice distortions | 100 | 100 |
| -6179644367731275038 | BCC crystals with random interstitial atom defects to capture short-range many-body dynamics | 100 | 5390 |
| 4672352770729486412 | Randomly distorted primitive bcc unit cells drawn from Szlachta et al.'s database | 1776 | 1776 |
| -7006009324790241020 | Damaged and half-molten (110) and (100) surfaces | 24 | 3264 |
| -6044163661052682031 | Configurations with single self-interstitial defects | 45 | 540 |
| -6319777710072090904 | (110) surface configurations | 45 | 540 |
| 8535745979338442047 | (111) surface configurations | 38 | 456 |
| 1212831078854177554 | (112) surface configurations | 45 | 540 |
| -8166563994143249518 | Trivacancy configurations | 14 | 1638 |

# Configuration labels

|Labels|Counts|
|---|---|
| bcc | 2423 |
| strain | 2823 |
| surface | 375 |
| a15 | 100 |
| vacancy | 224 |
| sc | 100 |
| fcc | 100 |
| divacancy | 24 |
| trivacancy | 14 |
| c15 | 100 |
| gamma_surface | 178 |
| aimd | 50 |
| diamond | 100 |
| interstitial | 145 |
| warning | 124 |
| large_forces | 124 |
| repulsive | 124 |
| hcp | 100 |
| liquid | 69 |
| dimer | 24 |
| isolated_atom | 1 |

# Figures
![The results of plot_histograms](histograms.png)
