import re
import itertools
from html.parser import HTMLParser

from core import ATOMS_ID_FIELD, ATOMS_NAME_FIELD, ATOMS_LABELS_FIELD
from core.configuration_sets import ConfigurationSet
from core.converters import CFGConverter, EXYZConverter
from core.property import Property
from core.property_settings import PropertySettings

# rzm: aggregate info up to a dataset

class Dataset:

    f"""
    Attributes:
        name (str):
            Name of the dataset

        authors (list[str]):
            Authors of the dataset

        links (list[str]):
            External links to associate with the dataset

        description (str):
            A description of the datast

        elements (list):
            A list of strings of element types

        configurations (list):
            A list of Configuration objects. Each element is guaranteed to have
            the following fields:
                `Configuration.atoms.info[{ATOMS_ID_FIELD}]` (ObjectId)
                `Configuration.atoms.info[{ATOMS_NAME_FIELD}]` (str)
                `Configuration.atoms.info[{ATOMS_LABELS_FIELD}]` (str)

        data (list):
            A list of Property objects OR a list of Dataset objects.

        configuration_sets (list):
            List of ConfigurationSet objects defining groups of configurations

        property_info (list):
            A list of named tuples with the structure (name, field, units).
            'name' is the human-readable name of the property; 'field' is a
            string like 'info.energy' or 'arrays.forces' that can be parsed to
            access the property in either the `ase.Atoms.info` dictionary or
            `ase.Atoms.arrays`; 'units' is a string describing the units in ASE
            format (e.g. 'eV/Ang').

        property_settings (list):
            A list of ProprtySetting objects collected by taking the union of
            the settings linked to by all entries in `properties`.

        co_label_regexes (dict):
            A dictionary where the key is a string that will be compiled with
            `re.compile()`, and the value is a list of string labels. Note that
            whenever this dictionary is re-assigned, the labels on all
            configuration in `configurations` are updated.

        cs_regexes (dict):
            A dictionary where the key is a string that will be compiled with
            `re.compile()`, and the value is the description of the
            configuration set. Note that whenever this dictionary is
            re-assigned, `configuration_sets` is re-constructed.
    """

    def __init__(
        self,
        name,
        authors=None,
        links=None,
        description=None,
        configurations=None,
        data=None,
        name_field=None,
        co_label_regexes=None,
        cs_regexes=None,
        ps_regexes=None,
        ):

        self.name           = name
        self.authors        = authors
        self.links          = links
        self.description    = description
        self.name_field     = name_field


        if authors is None: self.authors = []
        if links is None: self.links = []

        if configurations is None: configurations = []
        if data is None: data = []

        self.configurations     = configurations
        self.data               = data

        if co_label_regexes is None: co_label_regexes = {}
        if cs_regexes is None:
            cs_regexes = {'default': 'Default configuration set'}
        if ps_regexes is None: ps_regexes = {}

        self.cs_regexes         = cs_regexes
        self.ps_regexes         = ps_regexes

        self.co_label_regexes   = co_label_regexes

        self.resync()


    def resync(self):
        self.refresh_config_labels()
        self.refresh_config_sets()
        self.refresh_property_settings()


    @property
    def co_label_regexes(self):
        return self._co_label_regexes

    @co_label_regexes.setter
    def co_label_regexes(self, regex_dict):
        """IMPORTANT: use re-assignment instead of `del`, `.pop`, `.update()`"""

        for key, v in regex_dict.items():
            if isinstance(v, str):
                regex_dict[key] = [v]

        self._co_label_regexes = regex_dict
        self.refresh_config_labels()
        self.refresh_config_sets()


    @property
    def cs_regexes(self):
        return self._cs_regexes

    @cs_regexes.setter
    def cs_regexes(self, regex_dict):
        """IMPORTANT: use re-assignment instead of `del`, `.pop`, `.update()`"""
        if len(regex_dict) == 0:
            regex_dict = {'default': 'Default configuration set'}

        self._cs_regexes = regex_dict
        self.refresh_config_sets()


    @property
    def ps_regexes(self):
        return self._ps_regexes

    @ps_regexes.setter
    def ps_regexes(self, regex_dict):
        """IMPORTANT: use re-assignment instead of `del`, `.pop`, `.update()`"""
        self._ps_regexes = regex_dict
        self.refresh_property_settings()


    @classmethod
    def from_markdown(cls, html):
        # Parse information from markdown file
        parser = DatasetParser()
        parser.feed(html)
        parser.data['Name'] = parser.data['Name'][0]

        data_info = dict([l for l in parser.data['Data'] if len(l)])
        elements = [_.strip() for _ in data_info['Elements'].split(',')]

        # Build skeleton Dataset
        dataset = cls(
            name=parser.data['Name'],
            authors=parser.data['Authors'],
            links=parser.data['Links'],
            description=parser.data['Description'][0],
        )

        # Load configurations
        if data_info['Name field'] == 'None':
            data_info['Name field'] = None

        dataset.configurations = load_configurations(
            file_path=data_info['File'][1],
            file_format=data_info['Format'],
            name_field=data_info['Name field'],
            elements=elements,
            default_name=parser.data['Name'],
        )

        # Extract labels and trigger label refresh for configurations
        dataset.co_label_regexes = {
            key: [_.strip() for _ in desc.split(',')]
            for key, desc in parser.data['Configuration labels'][1:]
        }

        # Extract configuration sets and trigger CS refresh
        dataset.cs_regexes = {
            key: desc for key, desc in parser.data['Configuration sets'][1:]
        }

        # Map property fields to supplied names
        for row in parser.data['Properties'][1:]:
            dataset.rename_property(row[1], row[0])

        # Extract computed properties
        units = {}
        for prop in parser.data['Properties'][1:]:
            units[prop[0]] = prop[2]

        dataset.load_data(units)

        # Extract property settings
        ps_regexes = {}
        for row in parser.data['Property settings'][1:]:
            files = []
            for fname in row[3:]:
                with open(fname[1], 'r') as f:
                    files.append('\n'.join([_.strip() for _ in f.readlines()]))

            ps_regexes[row[0]] = PropertySettings(
                method=row[1],
                description=row[2],
                files=files,
            )

        dataset.ps_regexes = ps_regexes


    def add_configurations(self, configurations):
        n = len(self.configurations)
        for ci, conf in enumerate(configurations):
            if ATOMS_NAME_FIELD not in conf.atoms.info:
                conf.atoms.info[ATOMS_NAME_FIELD] = str(n+ci)

        self.configurations += configurations


    def rename_property(self, old_name, new_name):
        """Renames old_name field to new_name in atoms.info and atoms.arrays"""

        if old_name == new_name: return

        for conf in self.configurations:
            if old_name in conf.atoms.info:
                conf.atoms.info[new_name] = conf.atoms.info[old_name]
                del conf.atoms.info[old_name]

            if old_name in conf.atoms.arrays:
                conf.atoms.arrays[new_name] = conf.atoms.arrays[old_name]
                del conf.atoms.arrays[old_name]


    def load_data(self, properties):

        efs_names = {
            'energy', 'forces', 'stress', 'virial',
            'unrelaxed-potential-energy',
            'unrelaxed-potential-forces',
            'unrelaxed-cauchy-stress',
            }

        efs = set(properties.keys()).issubset(efs_names)

        if not efs:
            raise NotImplementedError(
                "Loading from HTML for datasets that contain properties "\
                    "other than 'energy', 'force', 'stress' is not implemented."
            )

        for ci, conf in enumerate(self.configurations):
            self.data.append(Property.EFS(conf, properties, instance_id=ci+1))


    def refresh_config_labels(self):
        """
        Re-applies labels to the `ase.Atoms.info[ATOMS_LABELS_FIELD]` list.
        Note that this overwrites any existing labels on the configurations.
        """

        if self.configurations is None:
            raise RuntimeError(
                "Dataset.configurations is None; must load configurations first"
            )

        # Apply configuration labels
        for conf in self.configurations:
            # Remove old labels
            conf.atoms.info[ATOMS_LABELS_FIELD] = set()

            for co_regex, labels in self.co_label_regexes.items():
                regex = re.compile(co_regex)

                if regex.search(conf.atoms.info[ATOMS_NAME_FIELD]):
                    old_set =  conf.atoms.info[ATOMS_LABELS_FIELD]

                    conf.atoms.info[ATOMS_LABELS_FIELD] = old_set.union(labels)

    def refresh_property_settings(self):
        """
        Refresh property pointers to PSOs by matching on their linked co names
        """

        if self.data is None:
            raise RuntimeError(
                "Dataset.data is None; must load data first"
            )

        # Reset Property PSO pointers
        for prop in self.data:

            if isinstance(prop, Dataset):
                raise RuntimeError(
                    'PropertySettings should not be used with nested datasets'
                )

            # Remove old pointers
            prop.settings = []

            for ps_regex, pso in self.ps_regexes.items():
                regex = re.compile(ps_regex)

                for conf in prop.configurations:
                    if regex.search(conf.atoms.info[ATOMS_NAME_FIELD]):
                        prop.settings.append(pso)


    def refresh_config_sets(self):
        """
        Re-constructs the configuration sets.
        """

        self.configuration_sets = []

        # Build configuration sets
        default_cs_description = None
        assigned_configurations = []
        for cs_regex, cs_desc in self.cs_regexes.items():
            if cs_regex.lower() == 'default':
                default_cs_description = cs_desc
                continue

            regex = re.compile(cs_regex)
            cs_configs = []
            for ai, conf in enumerate(self.configurations):

                if regex.search(conf.atoms.info[ATOMS_NAME_FIELD]):
                    cs_configs.append(conf)

                    assigned_configurations.append(ai)

            self.configuration_sets.append(ConfigurationSet(
                configurations=cs_configs,
                description=cs_desc
            ))

        unassigned_configurations = [
            ii for ii in range(len(self.configurations))
            if ii not in assigned_configurations
        ]

        if unassigned_configurations:

            if default_cs_description is None:
                raise RuntimeError(
                    "Must specify 'default' (or 'Default') configuration set"
                )

            self.configuration_sets.append(ConfigurationSet(
                configurations=[
                    self.configurations[ii]
                    for ii in unassigned_configurations
                ],
                description=default_cs_description
            ))


    def convert_units(self, units):
        """Converts the dataset units to the provided type (e.g., 'OpenKIM'"""
        pass


    def __str__(self):
        return "Dataset(name='{}', authors={}, description='{}')".format(
            self.name,
            self.authors,
            self.description,
        )

    def __repr__(self):
        return str(self)


def load_configurations(
    file_path,
    file_format,
    name_field,
    elements,
    default_name='',
    ):
    """
    Loads configurations as a list of ase.Atoms objects.

    Args:
        file_path (str):
            Path to the file or folder containing the data

        name_field (str):
            Key name to use to access `ase.Atoms.info[<name_field>]` to
            obtain the name of a configuration one the atoms have been
            loaded from the data file. Note that if
            `file_format == 'folder'`, `name_field` will be set to 'name'.

        elements (list):
            A list of strings of element types

        default_name (list):
            Default name to be used if `name_field==None`.
    """

    if file_format in ['xyz', 'extxyz']:
        converter = EXYZConverter()
    elif file_format == 'cfg':
        converter = CFGConverter()

    return converter.load(
        file_path,
        name_field=name_field,
        elements=elements,
        default_name=default_name
    )


class DatasetParser(HTMLParser):

    KNOWN_HEADERS = [
        'Name',
        'Authors',
        'Links',
        'Description',
        'Data',
        'Properties',
        'Property settings',
        'Property labels',
        'Configuration sets',
        'Configuration labels',
    ]

    def __init__(self):
        super().__init__()

        self.data           = {}
        self._t             = None
        self._table         = None
        self._loading_table = False
        self._loading_row   = False
        self._header        = None
        self._href          = None

    def handle_starttag(self, tag, attrs):
        self._t = tag

        if tag == 'table':
            self._table = []
            self._loading_table = True
        elif (tag == 'thead') or ('tag' == 'tbody'):
            pass
        elif tag == 'tr':
            self._loading_row = True
            self._table.append([])

        for att in attrs:
            if att[0] == 'href':
                self._href = att[1]


    def handle_endtag(self, tag):
        if tag == 'table':
            self.data[self._header] = self._table
            self._table = None
        elif tag == 'tr':
            self._loading_row = False

    def handle_data(self, data):
        data = data.strip()

        if data:
            if self._t == 'h1':
                # Begin reading new block
                if data not in self.KNOWN_HEADERS:
                    raise RuntimeError(
                        f"Header '{data}' not in {self.KNOWN_HEADERS}"
                    )
                self._header = data
                self.data[self._header] = []
            else:
                # Add data to current block
                if not self._loading_table:
                    # Adding basic text
                    self.data[self._header] += [_.strip() for _ in data.split('\n')]
                else:
                    # Specifically adding to a table
                    if self._loading_row:
                        if self._href is not None:
                            self._table[-1].append((data, self._href))
                            self._href = None
                        else:
                            self._table[-1].append(data)