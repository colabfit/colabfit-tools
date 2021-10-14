import os
import re
import markdown
import itertools
from ase.io import write
from html.parser import HTMLParser

from core import ATOMS_ID_FIELD, ATOMS_NAME_FIELD, ATOMS_LABELS_FIELD
from core.configuration_sets import ConfigurationSet
from core.converters import CFGConverter, EXYZConverter
from core.property import Property
from core.property_settings import PropertySettings

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

        property_settings (list):
            A list of ProprtySetting objects collected by taking the settings
            linked to by all entries in `properties`.

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
        property_map=None,
        co_label_regexes=None,
        cs_regexes=None,
        ps_regexes=None,
        ):

        self.name           = name
        self.authors        = authors
        self.links          = links
        self.description    = description

        self.is_parent_dataset = False

        if authors is None: self.authors = []
        if links is None: self.links = []

        if configurations is None: configurations = []
        if data is None: data = []

        self.configurations     = configurations
        self.data               = data

        if property_map is None: property_map = {}
        self.property_map = property_map

        if co_label_regexes is None: co_label_regexes = {}
        if cs_regexes is None:
            cs_regexes = {'default': 'Default configuration set'}
        if ps_regexes is None: ps_regexes = {}

        self.cs_regexes         = cs_regexes
        self.ps_regexes         = ps_regexes

        self.co_label_regexes   = co_label_regexes

        self.resync()


    def resync(self):

        self.is_parent_dataset = False
        for data in self.data:
            if isinstance(data, Dataset):
                self.is_parent_dataset = True
            else:
                if self.is_parent_dataset:
                    raise RuntimeError(
                        'Dataset cannot contain Datasets and Properties at '\
                            'the same time.'
                    )

        self.refresh_config_labels()
        self.refresh_config_sets()
        self.refresh_property_settings()
        self.aggregate_metadata()


    @property
    def property_map(self):
        return self._property_map

    @property_map.setter
    def property_map(self, property_map):
        # clean_map = {}
        for key in property_map:
            # clean_map[key] = {}
            for key2 in ['field', 'units']:
                if key2 not in property_map[key]:
                    raise RuntimeError(
                        'Missing "{}" in property_map["{}"]'.format(key2, key)
                    )

                # clean_map[key][key2] = property_map[key][key2]

            # if key == 'energy':
            #     clean_map['unrelaxed-potential-energy'] = property_map['energy']
            #     del clean_map['energy']

            # if key == 'forces':
            #     clean_map['unrelaxed-potential-forces'] = property_map['forces']
            #     del clean_map['forces']

            # if key == 'stress':
            #     clean_map['unrelaxed-cauchy-stress'] = property_map['stress']
            #     del clean_map['stress']

        # self._property_map = clean_map
        self._property_map = property_map


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
        for k, v in regex_dict.items():
            if not isinstance(v, PropertySettings):
                raise RuntimeError(
                    '`ps_regexes` keys must be PropertySettings objects'
                )

        self._ps_regexes = regex_dict
        self.refresh_property_settings()


    def to_markdown(self, base_folder, html_file_name, data_file_name, data_format, name_field=ATOMS_NAME_FIELD):
        """
        Saves a Dataset and writes a properly formatted markdown file. In the
        case of a Dataset that has child Dataset objects, each child Dataset
        is written to a separate sub-folder.

        Args:
            base_folder (str):
                Top-level folder in which to save the markdown and data files

            html_file_name (str):
                Name of file to save markdown to

            data_file_name (str):
                Name of file to save configuration and properties to

            data_format (str):
                Format to use for data file. Default is 'xyz'

            name_field (str):
                The name of the field that should be used to generate
                configuration names

        """
        self.resync()

        template = \
"""
# Name

{}

# Authors

{}

# Links

{}

# Description

{}

# Data

|||
|---|---|
|Elements|{}|
|File|[{}]({})|
|Format|{}|
|Name field|{}|

# Properties

|Name|Field|Units|
|---|---|---|
{}

# Property settings

|Regex|Method|Description|Labels|Files|
|---|---|---|---|---|
{}

# Configuration sets

|Regex|Description|
|---|---|
{}

# Configuration labels

|Regex|Labels|
|---|---|
{}
"""

        if not os.path.isdir(base_folder):
            os.mkdir(base_folder)

        if self.is_parent_dataset:
            for data in self.data:
                subdir = os.path.join(base_folder, data.name)

                if not os.path.isdir(subdir):
                    os.mkdir(subdir)

                data.to_markdown(
                    subdir, html_file_name, data_file_name, data_format
                )

        else:
            html_file_name = os.path.join(base_folder, html_file_name)
            data_file_name = os.path.join(base_folder, data_file_name)

            with open(html_file_name, 'w') as html:
                html.write(
                    template.format(
                        self.name,
                        '\n\n'.join(self.authors),
                        '\n\n'.join(self.links),
                        self.description,
                        ', '.join(self.elements),
                        data_file_name, data_file_name,
                        data_format,
                        name_field,
                        '\n'.join('| {} | {} | {} |'.format(k, v['field'], v['units']) for k,v in self.property_map.items()),
                        '\n'.join('| `{}` | {} | {} | {} | {} |'.format(regex.replace('|', '\|'), pso.method, pso.description, ', '.join(pso.labels), ', '.join('[{}]({})'.format(f, f) for f in pso.files)) for regex, pso in self.ps_regexes.items()),
                        '\n'.join('| `{}` | {} |'.format(regex.replace('|', '\|'), desc) for regex, desc in self.cs_regexes.items()),
                        '\n'.join('| `{}` | {} '.format(regex.replace('|', '\|'), ', '.join(labels)) for regex, labels in self.co_label_regexes.items()),
                    )
                )

            if data_format == 'xyz':
                data_format = 'extxyz'

            write(
                data_file_name,
                [conf.atoms for conf in self.configurations],
                format=data_format,
            )


    @classmethod
    def from_markdown(cls, html_file_path):
        with open(html_file_path, 'r') as f:
            html = markdown.markdown(f.read(), extensions=['tables'])

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
            key.replace('\|', '|'):
                [_.strip() for _ in desc.split(',')]
                for key, desc in parser.data['Configuration labels'][1:]
        }

        # Extract configuration sets and trigger CS refresh
        dataset.cs_regexes = {
            key.replace('\|', '|'):
                desc for key, desc in parser.data['Configuration sets'][1:]
        }

        # Map property fields to supplied names
        for row in parser.data['Properties'][1:]:
            dataset.rename_property(row[1], row[0])

        # Extract computed properties
        property_map = {}
        for prop in parser.data['Properties'][1:]:
            property_map[prop[0]] = {
                'field': prop[1],
                'units': prop[2],
            }

        dataset.property_map = property_map

        dataset.load_data()

        # Extract property settings
        ps_regexes = {}
        for row in parser.data['Property settings'][1:]:
            files = []
            if len(row) > 4:
                for fname in row[4:]:
                    with open(fname[1], 'r') as f:
                        files.append('\n'.join([_.strip() for _ in f.readlines()]))

            ps_regexes[row[0]] = PropertySettings(
                method=row[1],
                description=row[2],
                labels=[_.strip() for _ in row[3].split(',')],
                files=files,
            )

        dataset.ps_regexes = ps_regexes

        dataset.resync()

        return dataset


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


    def load_data(self, convert_units=False):
        if len(self.property_map) == 0:
            raise RuntimeError(
                'Must set `Dataset.property_map first'
            )

        efs_names = {
            'energy', 'forces', 'stress', 'virial',
            'unrelaxed-potential-energy',
            'unrelaxed-potential-forces',
            'unrelaxed-cauchy-stress',
            }

        efs = set(self.property_map.keys()).issubset(efs_names)

        if not efs:
            raise NotImplementedError(
                "Loading from HTML for datasets that contain properties "\
                    "other than 'energy', 'force', 'stress' is not implemented."
            )

        map_copy = {}
        for key in self.property_map:
            map_copy[key] = {}
            for key2 in self.property_map[key]:
                map_copy[key][key2] = self.property_map[key][key2]

        for ci, conf in enumerate(self.configurations):
            self.data.append(Property.EFS(
                conf, map_copy, instance_id=ci+1,
                convert_units=convert_units
            ))

    def refresh_property_map(self):
        property_map = {}

        for data in self.data:
            # if isinstance(data, Dataset):
            #     raise NotImplementedError("Nested datasets not supported yet.")

            for key, val in data.property_map.items():
                if key not in property_map:
                    property_map[key] = dict(val)
                else:
                    if val['units'] != property_map[key]['units']:
                        raise RuntimeError(
                            "Conflicting units found for property "\
                                "'{}': '{}' and '{}'".format(
                                    key, val['units'],
                                    property_map[key]['units']
                                )
                        )

        self.property_map = property_map


    def clear_config_labels(self):
        for conf in self.configurations:
            conf.atoms.info[ATOMS_LABELS_FIELD] = set()


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
        for co_regex, labels in self.co_label_regexes.items():
            regex = re.compile(co_regex)

            for conf in self.configurations:
                # # Remove old labels
                # conf.atoms.info[ATOMS_LABELS_FIELD] = set()

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

        self.property_settings = list(self.ps_regexes.values())

        # Reset Property PSO pointers
        for data in self.data:

            if isinstance(data, Dataset):
                self.property_settings += data.property_settings
                continue

            # Remove old pointers
            data.settings = None

            for ps_regex, pso in self.ps_regexes.items():
                regex = re.compile(ps_regex)

                for conf in data.configurations:
                    if regex.search(conf.atoms.info[ATOMS_NAME_FIELD]):
                        if data.settings is not None:
                            raise RuntimeError(
                                'Properties may only be linked to one PSO'
                            )

                        data.settings = pso

        if self.is_parent_dataset:
            for data in self.data:
                self.property_settings += data.property_settings


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

        if self.is_parent_dataset:
            for data in self.data:
                self.configuration_sets += data.configuration_sets


    def attach_dataset(self, dataset):
        self.data.append(dataset)
        self.configurations += dataset.configurations


    def aggregate_metadata(self):

        self.refresh_property_map()

        elements = {}
        self.chemical_systems = []
        self.property_types = []

        self.n_configurations = 0
        self.n_sites = 0
        co_labels = {}

        if self.is_parent_dataset:
            for ds in self.data:
                self.n_configurations += ds.n_configurations

                for el, er in zip(ds.elements, ds.elements_ratios):
                    if el not in elements:
                        elements[el] = er*ds.n_sites
                    else:
                        elements[el] += er*ds.n_sites

                for l, lc in zip(ds.co_labels, ds.co_labels_counts):
                    if l not in co_labels:
                        co_labels[l] = lc
                    else:
                        co_labels[l] += lc

                self.chemical_systems += ds.chemical_systems
                self.n_sites += ds.n_sites

        else:
            for cs in self.configuration_sets:
                self.n_configurations += cs.n_configurations

                for el, er in zip(cs.elements, cs.elements_ratios):
                    if el not in elements:
                        elements[el] = er*cs.n_sites
                    else:
                        elements[el] += er*cs.n_sites

                for l, lc in zip(cs.labels, cs.labels_counts):
                    if l not in co_labels:
                        co_labels[l] = lc
                    else:
                        co_labels[l] += lc

                self.chemical_systems += cs.chemical_systems
                self.n_sites += cs.n_sites

        self.elements = sorted(list(elements.keys()))
        self.elements_ratios = [
            elements[el]/self.n_sites for el in self.elements
        ]

        self.chemical_systems = sorted(list(set(self.chemical_systems)))

        self.co_labels = sorted(list(co_labels.keys()))
        self.co_labels_counts = [int(co_labels[l]) for l in self.co_labels]

        self.methods = []
        self.property_types = []
        self.n_properties = 0
        prop_counts = {}
        pso_labels = {}

        for data in self.data:
            if self.is_parent_dataset:
                self.methods += data.methods
                self.property_types += data.property_types
                self.n_properties += data.n_properties

                for pname, pc in zip(data.property_types, data.property_counts):
                    if pname not in prop_counts:
                        prop_counts[pname] = pc
                    else:
                        prop_counts[pname] += pc

                for l, lc in zip(data.pso_labels, data.pso_labels_counts):
                    if l not in pso_labels:
                        pso_labels[l] = lc
                    else:
                        pso_labels[l] += lc
            else:
                self.property_types.append(data.edn['property-id'])
                self.n_properties += 1

                if data.edn['property-id'] not in prop_counts:
                    prop_counts[data.edn['property-id']] = 1
                else:
                    prop_counts[data.edn['property-id']] += 1

                if data.settings is not None:
                    for l in data.settings.labels:
                        if l not in pso_labels:
                            pso_labels[l] = 1
                        else:
                            pso_labels[l] += 1

        self.methods = list(set(pso.method for pso in self.property_settings))

        self.property_types = list(prop_counts.keys())
        self.property_counts = [
            int(prop_counts[pname]) for pname in self.property_types
        ]

        self.pso_labels = sorted(list(pso_labels.keys()))
        self.pso_labels_counts = [int(pso_labels[l]) for l in self.pso_labels]


    def convert_units(self, units):
        """Converts the dataset units to the provided type (e.g., 'OpenKIM'"""
        pass


    def __str__(self):
        template = """Dataset
    Name:\n\t{}\n
    Authors:\n\t{}\n
    Description:\n\t{}\n
    Methods:\n\t{}\n
    Units:\n\t{}\n
    Number of configurations:\n\t{}\n
    Number of sites:\n\t{}\n
    Elements:\n\t{}\n
    Chemical systems:\n\t{}\n
    Properties:\n\t{}\n\t{}\n
    Property settings labels:\n\t{}\n
    Configuration labels:\n\t{}\n
    Configuration sets:\n\t{}"""

        return template.format(
            self.name,
            '\n\t'.join(self.authors),
            self.description,
            '\n\t'.join(self.methods),
            '\n\t'.join('{}: {}'.format(k, self.property_map[k]['units']) for k in self.property_map),
            self.n_configurations,
            self.n_sites,
            '\n\t'.join('{}\t({:.1f}% of sites)'.format(e, er*100) for e, er in zip(self.elements, self.elements_ratios)),
            '\n\t'.join(self.chemical_systems),
            'Total: {}'.format(self.n_properties),
            '\n\t'.join('{}: {}'.format(l, lc) for l, lc in zip(self.property_types, self.property_counts)),
            '\n\t'.join('{}: {}'.format(l, lc) for l, lc in zip(self.pso_labels, self.pso_labels_counts)),
            '\n\t'.join('{}: {}'.format(l, lc) for l, lc in zip(self.co_labels, self.co_labels_counts)),
            '\n\t'.join('{}: {}'.format(i, cs.description) for i, cs in enumerate(self.configuration_sets)),
        )


def load_configurations(
    file_path,
    file_format,
    name_field,
    elements,
    default_name='',
    labels_field=None,
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

        labels_field (str):
            Key name to use to access `ase.Atoms.info[<labels_field>]` to
            obtain the labels that should be applied to the configuration. This
            field should contain a comma-separated list of strings
    """

    if file_format in ['xyz', 'extxyz']:
        converter = EXYZConverter()
    elif file_format == 'cfg':
        converter = CFGConverter()

    return converter.load(
        file_path,
        name_field=name_field,
        elements=elements,
        default_name=default_name,
        labels_field=labels_field,
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
        data = data.rstrip('\n')

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