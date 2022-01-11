# Name

Example_Dataset

# Authors

A. B. Cee
E. F. Gee
H. I. Jay

# Links

https://www.google.com

# Description

This is an example dataset

# Storage format

|Elements|File|Format|Name field|
|---|---|---|---|
| In, P | [test_file.extxyz](test_file.extxyz) | xyz | name |

# Properties

|Property|KIM field|ASE field|Units
|---|---|---|---|
| [my-custom-property](test_property.edn) | a-custom-field-name | field-name | None
| [my-custom-property](test_property.edn) | a-custom-1d-array| 1d-array | eV |
| [my-custom-property](test_property.edn) | a-custom-per-atom-array | per-atom-array | eV |

# Property settings

|Property|Method|Description|Labels|Files|
|---|---|---|---|---|
| my-custom-property | VASP | energies/forces/stresses | GGA, PBE | [INCAR1](test_file.INCAR), [INCAR2](test_file.INCAR) |

# Configuration sets

Query|Description|
|---|---|
|`{}` | A default configuration set |

# Configuration labels

|Query|Labels|
|---|---|
| `{}` | water, molecule |