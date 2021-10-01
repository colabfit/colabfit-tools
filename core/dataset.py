import re
import itertools
from html.parser import HTMLParser

from core import ATOMS_ID_FIELD, ATOMS_NAME_FIELD, ATOMS_LABELS_FIELD
from core.configuration_sets import ConfigurationSet
from core.converters import CFGConverter, EXYZConverter
from core.observable import Observable

class WatcherList(list, Observable):
    """
    A helper class for propagating changes up to configuration and
    property sets. Whenever an object that is in the WatcherList calls
    `notify()`, the Watcher list also calls `notify()`.

    TODO: this seems like a bit of overkill. This will make it so that the DS
    and CS re-aggregate information a TON of times. Seems like it would be
    easier to just accept that a user might force a dataset to be out of date,
    and to provide a resync() function that calls everything necessary.
    """

    _observers = []
    _list = []

    def attach(self, observer):
        self._observers.append(observer)

    def detach(self, observer):
        self._observers.remove(observer)

    def update(self):
        self.notify()

    def notify(self):
        for observer in self._observers:
            observer.update()

    def append(self, v):
        v.attach(self)

        self._list.append(v)
        self.notify()

    def remove(self, v):
        v.attach(self)

        self._list.append(v)
        self.notify()

    def __delitem__(self, i):
        self._list[i].detach(self)

        del self._list[i]
        self.notify()


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

        configurations (WatcherList):
            A list of Configuration objects that watches for changes in its
            elements. Its elements are guaranteed to have the following fields:
                `Configuration.atoms.info[{ATOMS_ID_FIELD}]` (ObjectId)
                `Configuration.atoms.info[{ATOMS_NAME_FIELD}]` (str)
                `Configuration.atoms.info[{ATOMS_LABELS_FIELD}]` (str)

        properties (WatcherList):
            A list of Property objects that watches for changes in its elements.

        file_path (str):
            Path to the file or folder containing the data

        file_format (str):
            Format of file containing the data. Currently supported: 'xyz',
            'cfg', and 'folder'

        name_field (str):
            Key name to use to access `ase.Atoms.info[<name_field>]` to obtain
            the name of a configuration one the atoms have been loaded from the
            data file. Note that if `file_format == 'folder'`, `name_field` will
            be set to 'name'.

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
            A list of named tuples with the structure (regex, software,
            description, file). 'regex' is a string that will be passed to
            re.compile() and used to select a subset of the properties based on
            regex matching on their `property.description` field; 'software' is
            the name of the software used to calculate the property;
            'description' is an optional human-readable string used to describe
            the calculations settings; 'file' is an optional path to an example
            template file that was used to run the calculation.

        co_label_regexes (dict):
            A dictionary where the key is a string that will be compiled with
            `re.compile()`, and the value is a list of string labels

        cs_regexes (dict):
            A dictionary where the key is a string that will be compiled with
            `re.compile()`, and the value is the description of the
            configuration set.

    """

    def __init__(
        self,
        name,
        authors=None,
        links=None,
        description=None,
        configurations=None,
        properties=None,
        name_field=None,
        co_label_regexes=None,
        cs_regexes=None,
        ):

        self.name           = name
        self.authors        = authors
        self.links          = links
        self.description    = description
        self.name_field     = name_field


        if authors is None: self.authors = []
        if links is None: self.links = []

        if configurations is None: self._configurations = WatcherList()
        if properties is None: self._properties = WatcherList()

        if co_label_regexes is None: self._co_label_regexes = {}
        if cs_regexes is None:
            self._cs_regexes = {'default': 'Default configuration set'}

    @property
    def configurations(self):
        return self._configurations


    @configurations.setter
    def configurations(self, configurations):
        self._configurations = WatcherList(configurations)


    @property
    def co_label_regexes(self):
        return self._co_label_regexes

    @co_label_regexes.setter
    def co_label_regexes(self, regex_dict):
        """IMPORTANT: use re-assignment instead of `del`, `.pop`, `.update()`"""
        self._co_label_regexes = regex_dict
        self.refresh_config_labels()


    @property
    def cs_regexes(self):
        return self._cs_regexes

    @cs_regexes.setter
    def cs_regexes(self, regex_dict):
        """IMPORTANT: use re-assignment instead of `del`, `.pop`, `.update()`"""
        self._cs_regexes = regex_dict
        self.refresh_config_sets()


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

        dataset.configurations = dataset.load_configurations(
            file_path=data_info['File'],
            file_format=data_info['Format'],
            name_field=data_info['Name field'],
            elements=elements,
            default_name=parser.data['Name'],
        )

        # Extract labels and trigger label refresh for configurations
        dataset.co_label_regexes = {
            key: desc for key, desc in parser.data['Configuration labels']
        }

        # Extract configuration sets and trigger CS refresh
        dataset.cs_regexes = {
            key: desc for key, desc in parser.data['Configuration sets']
        }

        print(Dataset)


    def load_configurations(
        self,
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

            for co_regex, co_labels in self.co_label_regexes.items():
                regex = re.compile(co_regex)

                if regex.search(conf.atoms.info[ATOMS_NAME_FIELD]):
                    old_set =  conf.atoms.info[ATOMS_LABELS_FIELD]

                    conf.update_labels(old_set.union(co_labels))


    def refresh_property_labels(self):
        """
        Re-applies labels to the `property[PROP_LABELS_FIELD]` list.
        Note that this overwrites any existing labels on the configurations.
        """

        if self.configurations is None:
            raise RuntimeError(
                "Dataset.properties is None; must load configurations first"
            )

        # Apply configuration labels
        for atoms in self.configurations:
            # Remove old labels
            atoms.info[ATOMS_LABELS_FIELD] = set()

            for co_regex, co_labels in self.co_label_regexes.items():
                regex = re.compile(co_regex)

                if regex.search(atoms.info[ATOMS_NAME_FIELD]):
                    old_set =  atoms.info[ATOMS_LABELS_FIELD]
                    atoms.info[ATOMS_LABELS_FIELD] = old_set.union(co_labels)


    def refresh_config_sets(self):
        """
        Re-constructs the configuration sets.
        """

        self.configuration_sets = WatcherList()

        # Build configuration sets
        default_cs_description = None
        assigned_configurations = []
        for cs_regex, cs_desc in self.cs_regexes.items():
            if cs_regex.lower() == 'default':
                default_cs_description = cs_desc
                continue

            regex = re.compile(cs_regex)
            cs_configs = []
            for ai, atoms in enumerate(self.configurations):
                if regex.search(atoms.info[ATOMS_NAME_FIELD]):
                    cs_configs.append(atoms)
                    atoms.attach(cs_configs)

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


    def aggregate(self):
        pass


class DatasetParser(HTMLParser):

    KNOWN_HEADERS = [
        'Name',
        'Authors',
        'Links',
        'Description',
        'Data',
        'Properties',
        'Property sets',
        'Property labels',
        'Configuration sets',
        'Configuration labels',
    ]

    def __init__(self):
        super().__init__()

        self.data = {}
        self._t = None
        self._table = None
        self._loading_table = False
        self._loading_row = False
        self._header = None

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
                        self._table[-1].append(data)