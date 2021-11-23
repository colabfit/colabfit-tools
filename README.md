# colabfit-tools
Tools for constructing and manipulating datasets for fitting interatomic potentials

# Installation
1. Clone repository
2. `cd` into repository
3. `pip install -e .`

# Basic usage

## Creating a [Dataset](colabfit/tools/dataset.py) from scratch
Import the basic classes and functions.

```python
from colabfit.tools.dataset import Dataset, load_data
```

Initialize the Dataset and add basic metadata.

```python
dataset = Dataset(name='example')

dataset.authors = [
    'J. E. Lennard-Jones',
]

dataset.links = [
    'https://en.wikipedia.org/wiki/John_Lennard-Jones'
]

dataset.description = "This is an example dataset"
```

## Loading configurations
Load the configurations (and the linked properties) onto the Dataset using `load_data()`, which calls a pre-made [Converter](colabfit/tools/converters.py) and returns a list of [Configuration](colabfit/tools/configuration.py) objects. Note that the `file_format` must be specified, and a `name_field` (or `None`) should be provided to specify the name of the loaded configurations.
```python
dataset.configurations = load_data(
    file_path='/path/to/data/file',
    file_format='xyz',
    name_field='name'  # key in ase.Atoms.info to use as the Configuration name
    elements=['H'],    # order matters for CFG files, but not others
    default_name='example'  # default name with `name_field` not found
)
```

## Parsing the data
Parse the properties by specifying a `property_map`, which is a special dictionary on a dataset. Note that the keys should either be 1) the name of an OpenKIM Property Definition from [the list of approved OpenKIM Property Definitions](https://openkim.org/properties), or 2) the name of a locally-defined property (see details below) , or 3) the string `'default'`. Note that `'default'` means that an existing property will be used with support for basic fields like `'energy'`, `'forces'`, and `'stress'`. `'field'` is used to specify the key for extracting the property from `ase.Atoms.info` or `ase.Atoms.arrays`. `load_data()` will extract the fields provided in `property_map` from the configuration, and store them as [Property](colabfit/tools/property.py) objects in the `dataset.data` list. If a custom property is used, the `Dataset.custom_definitions` dictionary must be updated to either point to the local EDN file or a Python dictionary representation of the contents of the EDN file.

```python
dataset.property_map = {
    # ColabFit Property Name: {Property field name: {'field': ASE field name, 'units': ASE units string}}
    'default': {
        'energy': {'field': 'energy', 'units': 'eV'},
        'forces': {'field': 'F',      'units': 'eV/Ang'}
    },
    'my-custom-property': {
        'a-custom-field-name':      {'field': 'field-name',      'units': None},
        'a-custom-1d-array':        {'field': '1d-array',        'units': 'eV'},
        'a-custom-per-atom-array':  {'field': 'per-atom-array',  'units': 'eV'},
    }
}

dataset.custom_definitions = {
    'my-custom-property': 'tests/files/test_property.edn'.
    # OR: 'my-custom-property': {... contents of an EDN file}
}

dataset.parse_data(convert_units=True, verbose=True)
```

## Defining custom properties
Custom properties can be defined by writing an EDN file that has been formatted according to the [KIM Properties Framework](https://openkim.org/doc/schema/properties-framework/). An example EDN file is shown below, and can be modified to fit most use cases.
```
{
  "property-id" "my-custom-property"

  "property-title" "A custom, user-provided Property Definition. See https://openkim.org/doc/schema/properties-framework/ for instructions on how to build these files."

  "property-description" "Some human-readable description"

  "a-custom-field-name" {
    "type"         "string"
    "has-unit"     false
    "extent"       []
    "required"     false
    "description"  "The description of the custom field"
  }
  "a-custom-1d-array" {
    "type"         "float"
    "has-unit"     true
    "extent"       [":"]
    "required"     true
    "description"  "This should be a 1D vector of floats"
  }
  "a-custom-per-atom-array" {
    "type"         "float"
    "has-unit"     true
    "extent"       [":",3]
    "required"     true
    "description"  "This is a 2D array of floats, where the second dimension has a length of 3"
  }
}
```

## Applying labels to configurations
Metadata can be applied to individual configurations using labels. Labels are applied by matching a regular expression to `configuration.info[ASE_NAME_FIELD]` for each configuration. Regex mappings are provided by setting the `configuration_label_regexes` dictionary.

```python

# Labels can be specified as lists or single strings (which will be wrapped in a list).
dataset.configuration_label_regexes = {
    'H2O': ['water', 'molecule'],
    '.*': '100K',
}
```

## Building configuration sets
[ConfigurationSet](colabfit/tools/configuration_sets.py) can be used to create groups of configurations for organizational purposes. This can be done in a similar manner to how configuration labels are applied, but using the `configuration_set_regexes` dictionary. Note that a configuration may exist in multiple sets at the same time. Note that a `default` CS must be provided.
```python
dataset.configuration_set_regexes = {
    'default': "The default CS to use for configurations that don't match anything else",
    'H2O': 'AIMD snapshots of liquid water at 100K',
}
```

## Providing calculation metadata
Metadata for computing properties can be provided by constructing a [PropertySettings](colabfit/tools/property_settings.py) object and matching it to a property by regex matching on the property's linked configurations.
```python
from colabfit.tools.property_settings import PropertySettings

dataset.property_settings_regexes = {
    '.*':
        PropertySettings(
            method='VASP',
            description='energy/force calculations',
            files=['/path/to/INCAR'],
            labels=['PBE', 'GGA'],
        )
}
```

## Synchronizing a dataset
A Dataset is a pool of configurations and properties, where the configurations are further organized by grouping them into configuration sets, and the properties are linked to property settings. A Dataset then aggregates information up from the configurations, properties, and property settings. In order to ensure that the information applied by specifying `configuration_label_regexes`, `configuration_set_regexes`, and `property_settings_regexes` are up-to-date, `dataset.resync()` should be called before performing critical operations like saving a Dataset. Some core functions will call `resync()` automatically.

## Loading from Markdown
Datasets can be easily written/read to/from Markdown files using `Dataset.to_markdown()` and `Dataset.from_markdown()`. This can be a useful way of organizing a Dataset without having to construct it with the programming interface. A description of how to design a proper Markdown file can be found in [colabfit/examples/example.md](colabfit/examples/example.md).

## Data transformations
Data transformations can be applied by supplying
[Transformation](colabfit/tools/transformations.py) objects to `Dataset.apply_transformations()`.

```python
from colabfit.tools.transformations import BaseTransform, Sequential, SubtractDivide, PerAtomEnergies

class ConvertToStress(BaseTransform):
    def __init__(self):
        super(ScaleVirial, self).__init__(
            lambda x, c: (-np.array(x)/c.cell.volume*160.21766208)
        )

reference_energy = -3.14159  # eV/atom

# Keys should match those used in dataset.property_map
dataset.apply_transformations({
    'energy': Sequential(
        PerAtomEnergies(),
        SubtractDivide(sub=reference_energy, div=1)
    ),
    'stress': ConvertToStress(),
})
```

## Exploring the data

Use `Dataset.get_data()` to obtain a list of the given property field where each
element has been wrapped in a numpy array.

```
energies = np.concatenate(dataset.get_data('energy'))
forces   = np.concatenate(dataset.get_data('forces')).ravel()
```

Basic statistics can be obtained using `Dataset.get_statistics()`.

```
# Returns: {'average': ..., 'std': ..., 'min': ..., 'max':, ..., 'average_abs': ...}
dataset.get_statistics('energy')
```

Visualize property distributions using `Dataset.plot_histograms()`.
```
dataset.plot_histograms(['energy', 'forces'], yscale=['linear', 'log'])
```


## Train/test split
Easily generate new train/test datasets using `Dataset.train_test_split()`

```
train_ds, test_ds = low_forces.train_test_split(0.1)
```

# Example data
Example datasets and markdown files can be found at [this (currently private) github repository](https://github.com/jvita/colabfit/tree/tools/data/formatted). To request access, talk to Josh.
