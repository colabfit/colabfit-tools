
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
|File|tests/files/dummy_file.extxyz|
|Format|xyz|
|Name field|name|

# Properties

|Name|Field|Units|
|---|---|---|
|energy|info.energy|eV|
|forces|arrays.forces|eV/Ang|
|stress|info.virial|bar|

# Property sets

|Regex|Description|Files|
|---|---|----|----|
|`.*`|Energies and forces computed with VASP using certain settings|[INCAR](dummy_file.INCAR)|

# Property labels
|Regex|Software|Labels|
|`.*`|VASP|PBE, GGA|

# Configuration sets

|Regex|Description|
|---|---|
|`Default`|A default configuration set is required|

# Configuration labels

|Regex|Labels|
|---|---|
|`.*`|dummy_label1, dummy_label2|