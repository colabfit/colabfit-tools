
# Name

AlNiTi_CMS2019

# Authors

Josh Vita

# Links

joshvita.com

# Description

A dataset that Josh uploaded

# Storage format

|Elements|File|Format|Name field|
|---|---|---|---|
| Ni, Ti, Al | ./data/uploads/AlNiTi_CMS2019/AlNiTi_CMS2019.xyz | xyz | _name |

# Properties

|Property|KIM field|ASE field|Units
|---|---|---|---|
| [energy-forces-stress](./data/uploads/AlNiTi_CMS2019/energy-forces-stress.edn) | energy | energy | eV

# Property settings

|Method|Description|Labels|Files|
|---|---|---|---|
| VASP | A VASP calculation | PBE, GGA |  |

# Configuration sets

|Query|Description|
|---|---|
| `{'names': {'$regex': 'train_1st_stage'}}` | Configurations used in the first stage of training |

# Configuration labels

|Query|Labels|
|---|---|
| `{}` | active_learning |
