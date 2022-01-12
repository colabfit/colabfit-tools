
# Summary
|Chemical systems|Element ratios|# of properties|# of configurations|# of atoms|
|---|---|---|---|---|
|CH, CFHN, CFHNO, CNO, CHN, CFNO, CHO, CN, CFN, CFH, CFHO, CF, CHNO|C (35.2%), H (51.2%), N (5.5%), O (7.9%), F (0.1%)|139874|139874|2522524|

# Name

QM9_filtered

# Authors

Raghunathan Ramakrishnan

Pavlo Dral

Matthias Rupp

O. Anatole von Lilienfeld

# Links

https://www.nature.com/articles/sdata201422

https://figshare.com/collections/Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978904

# Description

The QM9 dataset, filtered by removing values with a>=20 or b>=10

# Storage format

|Elements|File|Format|Name field|
|---|---|---|---|
| C, H, N, O, F | 8002475287264684058 | mongo | _name |

# Properties

|Property|KIM field|ASE field|Units
|---|---|---|---|
| [qm9-property](/home/jvita/scripts/colabfit-tools/colabfit/examples/QM9_filtered/qm9-property.edn) | a | a | GHz
| [qm9-property](/home/jvita/scripts/colabfit-tools/colabfit/examples/QM9_filtered/qm9-property.edn) | b | b | GHz
| [qm9-property](/home/jvita/scripts/colabfit-tools/colabfit/examples/QM9_filtered/qm9-property.edn) | c | c | GHz
| [qm9-property](/home/jvita/scripts/colabfit-tools/colabfit/examples/QM9_filtered/qm9-property.edn) | mu | mu | Debye
| [qm9-property](/home/jvita/scripts/colabfit-tools/colabfit/examples/QM9_filtered/qm9-property.edn) | alpha | alpha | Bohr\*Bohr\*Bohr
| [qm9-property](/home/jvita/scripts/colabfit-tools/colabfit/examples/QM9_filtered/qm9-property.edn) | homo | homo | Hartree
| [qm9-property](/home/jvita/scripts/colabfit-tools/colabfit/examples/QM9_filtered/qm9-property.edn) | lumo | lumo | Hartree
| [qm9-property](/home/jvita/scripts/colabfit-tools/colabfit/examples/QM9_filtered/qm9-property.edn) | gap | gap | Hartree
| [qm9-property](/home/jvita/scripts/colabfit-tools/colabfit/examples/QM9_filtered/qm9-property.edn) | r2 | r2 | Bohr*Bohr
| [qm9-property](/home/jvita/scripts/colabfit-tools/colabfit/examples/QM9_filtered/qm9-property.edn) | zpve | zpve | Hartree
| [qm9-property](/home/jvita/scripts/colabfit-tools/colabfit/examples/QM9_filtered/qm9-property.edn) | u0 | u0 | Hartree
| [qm9-property](/home/jvita/scripts/colabfit-tools/colabfit/examples/QM9_filtered/qm9-property.edn) | u | u | Hartree
| [qm9-property](/home/jvita/scripts/colabfit-tools/colabfit/examples/QM9_filtered/qm9-property.edn) | h | h | Hartree
| [qm9-property](/home/jvita/scripts/colabfit-tools/colabfit/examples/QM9_filtered/qm9-property.edn) | g | g | Hartree
| [qm9-property](/home/jvita/scripts/colabfit-tools/colabfit/examples/QM9_filtered/qm9-property.edn) | cv | cv | cal/mol/K
| [qm9-property](/home/jvita/scripts/colabfit-tools/colabfit/examples/QM9_filtered/qm9-property.edn) | smiles-relaxed | smiles-relaxed | None
| [qm9-property](/home/jvita/scripts/colabfit-tools/colabfit/examples/QM9_filtered/qm9-property.edn) | inchi-relaxed | inchi-relaxed | None

# Property settings

|ID|Method|Description|Labels|Files|
|---|---|---|---|---|
| -993823251101522014 | DFT/B3LYP/6-31G(2df,p) | QM9 property settings calculation | 6-31G(2df,p), B3LYP, DFT |  |

# Configuration sets

|ID|Description|# of structures| # of atoms|
|---|---|---|---|
| 1074603459859357389 | The subset of all 133,885 species with up to nine heavy atoms (CONF) out of the GDB-17 chemical universe of 166 billion organic molecules | 133779 | 2406719 |
| 865312989011446416 | Isomers of C7O2H10 | 6095 | 115805 |

# Configuration labels

|Labels|Counts|
|---|---|


# Figures
![The results of plot_histograms](histograms.png)
