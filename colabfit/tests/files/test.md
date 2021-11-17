
# Name

Example_Dataset

# Authors

A. B. Cee
E. F. Dee
H. I. Jay

# Links

https://www.google.com

# Description

This is an example dataset

# Data
|||
|---|---|
|Elements|In, P|
|File|[test_file.extxyz](test_file.extxyz)|
|Format|xyz|
|Name field|name|

# Properties

|Property|Name|Field|Units|
|---|---|---|---|
|default|energy|energy|eV|
|default|forces|forces|eV/Ang|
|default|stress|virial|kilobar|
|[my-custom-property](test_property.edn)|a-custom-field-name|field-name|None|
|[my-custom-property](test_property.edn)|a-custom-1d-array|1d-array|eV|
|[my-custom-property](test_property.edn)|a-custom-per-atom-array|per-atom-array|eV|

# Property settings

|Regex|Method|Description|Labels|Files|
|---|---|----|----|
|`.*`|VASP|Energies, forces, and stresses computed with VASP using certain settings| PBE, GGA |[INCAR](test_file.INCAR)|

# Configuration sets

|Regex|Description|
|---|---|
|`Default`|A default configuration set is required|

# Configuration labels

|Regex|Labels|
|---|---|
|`.*`|dummy_label1, dummy_label2|