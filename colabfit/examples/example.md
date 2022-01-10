# Summary

This section is ignored during parsing. It is intended to be used as a place to
display additional information about a Dataset, and will be ignored by `from_markdown()`.

All other sections and tables are required for use with
`from_markdown()`. Sections should be specified using `<h1>` headers.
Exact header spelling and capitalization is required. All table column headers
are required unless otherwise specified. Sections that have tables in them will
only use the table for `from_markdown()`. Sections may be given in different
orders than shown here. Additional sections may be added, but will be ignored by
`from_markdown()`.

Any files should be provided as links in Markdown as described
[here](https://www.markdownguide.org/basic-syntax/#links), using the format
`"[<text>](<path_to_file>)"`.


The table below will be automatically generated when using `to_markdown()`.

|||
|---|---|
|Chemical systems||
|Element ratios||
|# of configurations||
|# of atoms||

This section can also include figures. These figures would typically be the
histograms generated via `MongoDatabase.plot_histograms()`, and will be ignored
by `from_markdown()`.

# Name

A single-line name to use for the dataset.

# Authors

Authors of the dataset (one per line)

# Links

Links to associate with the dataset (one per line)

# Description

A multi-line human-readable description of the dataset

# Storage format

A table for specifying how to load the configurations. First column of each row
('Elements', 'File', 'Format', 'Name field') must be as spelled/capitalized
here. When using `to_markdown()`, if `data_format='mongo'`, then "File" will be
the ID of the Dataset in the Mongo Database.

|||
|---|---|
|Elements| a comma-separated list of elements ("Mo, Ni, Cu")|
|File| a hyperlink to the raw data file ([example.extxyz](example.extxyz))|
|Format| a string specifying the file format ('xyz', 'cfg')|
|Name field| the key to use as the name of configurations ('name')|

# Properties

This section is for storing the tabular form of the `property_map` argument for
the `MongoDatabase.insert_data()` function. The table below should have the
following columns:

* `Property`: An OpenKIM Property Definition ID from [the list of approved OpenKIM Property
   Definitions](https://openkim.org/properties) OR a link to a local EDN file
   (using the notation `[<name_of_property>](<path_to_edn_file>)`) that has been
   formatted according to the [KIM Properties
   Framework](https://openkim.org/doc/schema/properties-framework/)
* `KIM field`: A string that can be used as a key for a dictionary
  representation of an OpenKIM Property Instance
* `ASE field`: A string that can be used as a key for the `info` or `arrays`
  dictionary on a Configuration
* `Units`: A string describing the units of the field (in an ASE-readable
  format), or `None` if the field is unitless

|Property|KIM field|ASE field|Units|
|---|---|---|---|
|an-existing-property|energy|energy|eV|
|an-existing-property|forces|F|eV/Ang|
|[my-custom-property](colabfit/tests/files/test_property.edn)|a-custom-field-name|field-name|None|
|[my-custom-property](colabfit/tests/files/test_property.edn)|a-custom-1d-array|1d-array|eV|
|[my-custom-property](colabfit/tests/files/test_property.edn)|a-custom-per-atom-array|per-atom-array|eV|



# Property settings

The tabular form of the `property_settings` argument to the
`MongoDatabase.insert_data()` function.

The first column should be either "ID"
or "Property". If "ID" , then the Dataset should already exist in the
Database, and "ID" should be the ID of an existing PropertySetting. If
"Property", then this will make it so that the given PropertySettings are linked
to all Properties of the given type.

Property|Method|Description|Labels|Files|
|---|---|---|---|---|
| my-custom-property | VASP | energies/forces/stresses | PBE, GGA |  |

# Configuration sets

A table used for building ConfigurationSets. A ConfigurationSet is built by
writing a Mongo query that returns a list of Configuration IDs to include in the
ConfigurationSet, then providing a human-readable description of the
ConfigurationSet.

The first column should be either "ID" or "Query". If "ID", then all of the
Configurations being added should already exist in the Database, and "ID" should
be the ID of an existing ConfigurationSet. If "Query", then "Query" should be a
Mongo query expressed as a Python dictionary (which will be converted using
`ast.literal_eval()`). For more details on how to write Mongo queries, refer to the "Mongo" example
page in the documentation. `to_markdown()` will use "ID". The rest of the
columns will be filled in programmatically by `to_markdown()` and ignored by
`from_markdown()`.

Note that when using `from_markdown()`, all loaded Configurations must be
contained by at least one ConfigurationSet; if any Configurations don't match
any of the queries provided here, a warning will be raised.

All other text in this section will be ignored.

Query|Description|# of structures| # of atoms|
|---|---|---|---|
| `{'names': {'$regex': 'H20'}}` | AIMD snapshots of liquid water at 100K | | |

# Configuration labels

A table used for applying configuration labels, structured similar to the one in
the "Configuration sets" section. Only the "Labels" column is required. If
using `from_markdown()`, a "Query" column is also required, and will be used to
apply the given labels to Configurations matching the query. `to_markdown()`
will also create a "Counts" column. Labels should be given as a
comma-separated list. When using `to_markdown()`.

All other text in this section will be ignored.

|Query|Labels|Counts|
|---|---|---|
| `{'names': {'$regex': 'H20'}}` | water, molecule |  |

# Figures

This section is used for including any figures. This section is ignored by
`from_markdown()`, and is automatically populated with the results of
`plot_histograms()` for `to_markdown()`.