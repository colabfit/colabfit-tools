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
Parse the properties by specifying a `property_map`, which is a special dictionary on a dataset. Note that the key should be the name of an OpenKIM Property Definition (though 'energy', 'forces', and 'stress' are automatically renamed). `'field'` is used to specify the key for extracting the property from `ase.Atoms.info` or `ase.Atoms.arrays`. `load_data()` will extract the fields provided in `property_map` from the configuration, and store them as [Property](colabfit/tools/property.py) objects in the `dataset.data` list.
```python
dataset.property_map = {
    # ColabFit name: {'field': ASE field name, 'units': ASE units string}
    'energy': {'field': 'energy', 'units': 'eV'}
    'forces': {'field': 'F',      'units': 'eV/Ang'}
}

# Parse data, and optionally convert to ColabFit-compliant units
dataset.parse_data(convert_units=True)
```

## Applying labels to configurations
Metadata can be applied to individual configurations using labels. Labels are applied by matching a regular expression to `configuration.info[ASE_NAME_FIELD]` for each configuration. Regex mappings are provided by setting the `co_label_regexes` dictionary.
```python

# Labels can be specified as lists or single strings (which will be wrapped in a list).
dataset.co_label_regexes = {
    'H2O': ['water', 'molecule'],
    '.*': '100K',
}
```

## Building configuration sets
[ConfigurationSet](colabfit/tools/configuration_sets.py) can be used to create groups of configurations for organizational purposes. This can be done in a similar manner to how configuration labels are applied, but using the `cs_regexes` dictionary. Note that a configuration may exist in multiple sets at the same time. Note that a `default` CS must be provided.
```python
dataset.cs_regexes = {
    'default': "The default CS to use for configurations that don't match anything else",
    'H2O': 'AIMD snapshots of liquid water at 100K',
}
```

## Providing calculation metadata
Metadata for computing properties can be provided by constructing a [PropertySettings](colabfit/tools/property_settings.py) object and matching it to a property by regex matching on the property's linked configurations.
```python
from colabfit.tools.property_settings import PropertySettings

dataset.ps_regexes = {
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
A Dataset is a pool of configurations and properties, where the configurations are further organized by grouping them into configuration sets, and the properties are linked to property settings. A Dataset then aggregates information up from the configurations, properties, and property settings. In order to ensure that the information applied by specifying `co_label_regexes`, `cs_regexes`, and `ps_regexes` are up-to-date, `dataset.resync()` should be called before performing critical operations like saving a Dataset. Some core functions will call `resync()` automatically.

## Loading from Markdown
Datasets can be easily written/read to/from Markdown files using `Dataset.to_markdown()` and `Dataset.from_markdown()`. This can be a useful way of organizing a Dataset without having to construct it with the programming interface. A description of how to design a proper Markdown file can be found in [examples/example.md](examples/example.md).

## Data transformations
Data transformations can be applied by supplying
[Transformation](colabfit/transformations.py) objects to `dataset.apply_transformations()`.

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

# Example data
Example datasets and markdown files can be found at [this (currently private) github repository](https://github.com/jvita/colabfit/tree/tools/data/formatted). To request access, talk to Josh.
