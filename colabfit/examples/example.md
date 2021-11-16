# Summary

This section is ignored during parsing. It is intended to be used as a place to display additional information about a Dataset.

All other sections and tables are required for use with `Dataset.from_markdown()`. Sections should be specified using `<h1>` headers. Exact header spelling and capitalization is required. All table column headers are required unless otherwise specified. Sections may be given in different orders than shown here. Additional sections may be added, but will be ignored by `Dataset.from_markdown()`.

The table below will be automatically generated when using `Dataset.to_markdown()`.

|||
|---|---|
|Chemical systems||
|Element ratios||
|# of configurations||
|# of atoms||

# Name

A single-line name to use for the dataset.

# Authors

Authors of the dataset (one per line)

# Links

Links to associate with the dataset (one per line)

# Description

A multi-line human-readable description of the dataset

# Data

A table for storing the inputs to `load_data()`. First column of each row
('Elements', 'File', 'Format', 'Name field') must be as spelled/capitalized
here. File name must Must have the following rows
(with first columns spelled/capitalized in
this way):

|||
|---|---|
|Elements| a comma-separated list of elements ("Mo, Ni, Cu")|
|File| a hyperlink to the raw data file ([example.extxyz](example.extxyz))|
|Format| a string specifying the file format ('xyz', 'cfg')|
|Name field| the key to use as the name of configurations ('name')|

# Properties

The tabular form of `Dataset.property_map`. For example:

|Name|Field|Units|
|---|---|---|
|energy|energy|eV|
|forces|F|eV/Ang|

# Property settings

The tabular form of `Dataset.property_settings_regexes`. Note that the "Labels" and "Files" columns should be in the table, but may be left empty. For example:

|Regex|Method|Description|Labels|Files|
|---|---|---|---|---|
| `.*` | VASP | energies/forces/stresses | PBE, GGA |  |

# Configuration sets

The tabular form of `Dataset.configuration_set_regexes`. Only the first two columns ("Regex" and "Description") are required. The rest will be filled in programmatically by `Dataset.to_markdown()` and ignored by `Dataset.from_markdown()`. For example:

|Regex|Description|# of structures| # of atoms|
|---|---|---|---|
| `default` | The default CS to use for configurations | | |
| `H2O` | AIMD snapshots of liquid water at 100K | | |

# Configuration labels

The tabular form of `Dataset.configuration_label_regexes`. Only the first two columns ("Regex" and "Labels") are required. The rest will be filled in programmatically by `Dataset.to_markdown()` and ignored by `Dataset.from_markdown()`. For example:

|Regex|Labels|Counts|
|---|---|---|
| `H2O` | water, molecule |  |
| `.*` | 100K |  |
