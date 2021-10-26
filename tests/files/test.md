
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
|File|[test_file.extxyz](tests/files/test_file.extxyz)|
|Format|xyz|
|Name field|name|

# Properties

|Name|Field|Units|
|---|---|---|
|energy|energy|eV|
|forces|forces|eV/Ang|
|stress|virial|kilobar|

# Property settings

|Regex|Method|Description|Labels|Files|
|---|---|----|----|
|`.*`|VASP|Energies, forces, and stresses computed with VASP using certain settings| PBE, GGA |[INCAR](tests/files/test_file.INCAR)|

# Configuration sets

|Regex|Description|
|---|---|
|`Default`|A default configuration set is required|

# Configuration labels

|Regex|Labels|
|---|---|
|`.*`|dummy_label1, dummy_label2|