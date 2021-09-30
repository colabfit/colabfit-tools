import re
import itertools
from html.parser import HTMLParser

from core import ATOMS_NAME_FIELD, ATOMS_LABELS_FIELD
from core.configuration_sets import ConfigurationSet
from core.converters import CFGConverter, EXYZConverter


class Dataset:

    """
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
            A list of ase.Atoms objects

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

        TODO: don't need property_settings; a PropertySet will store this info.

    """

    def __init__(
        self,
        name=None,
        authors=None,
        links=None,
        description=None,
        configurations=None,
        name_field=None,
        configuration_sets=None,
        ):

        self.name           = name
        self.authors        = authors
        self.links          = links
        self.description    = description
        self.name_field     = name_field

        if authors is None: self.authors = []
        if links is None: self.links = []
        if configurations is None: self._configurations = []
        if configuration_sets is None: self.configuration_sets = []

        # TODO: a Dataset shouldn't store things like file_path and file_format.
        # it should just be passed the CO/CS/PR/PS objects


    @property
    def configurations(self):
        return self._configurations

    @configurations.setter
    def configurations(self, configurations):
        self.elements = sorted(list(set(itertools.chain.from_iterable(
            a.get_chemical_symbols() for a in configurations
        ))))

        self._configurations = configurations


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
            # file_path=data_info['File'],
            # file_format=data_info['Format'],
        )

        # Load configurations
        if data_info['Name field'] == 'None':
            data_info['Name field'] = None

        if data_info['Format'] in ['xyz', 'extxyz']:
            converter = EXYZConverter()
            dataset.configurations = converter.load(
                data_info['File'],
                name_field=data_info['Name field'],
                elements=elements,
                default_name=parser.data['Name'],
            )
        elif data_info['Format'] == 'cfg':
            converter = CFGConverter()
            dataset.configurations = converter.load(
                data_info['File'],
                name_field=data_info['Name field'],
                elements=elements,
                default_name=parser.data['Name'],
            )

        # Apply configuration labels
        for co_regex, co_labels in parser.data['Configuration labels'][1:]:
            regex = re.compile(co_regex)
            labels = [_.strip() for _ in co_labels.split(',')]

            for atoms in dataset.configurations:
                if regex.search(atoms.info[ATOMS_NAME_FIELD]):
                    atoms.info[ATOMS_LABELS_FIELD] = labels

        # Build configuration sets
        default_cs_description = None
        unassigned_configurations = list(range(len(dataset.configurations)))
        for cs_regex, cs_desc in parser.data['Configuration sets'][1:]:
            if cs_regex.lower() == 'default':
                default_cs_description = cs_desc
                continue

            regex = re.compile(cs_regex)
            cs_configs = []
            for ai, atoms in enumerate(dataset.configurations):
                if regex.search(atoms.info[ATOMS_NAME_FIELD]):
                    cs_configs.append(atoms)
                    del unassigned_configurations[ai]

            dataset.configuration_sets.append(ConfigurationSet(
                configurations=cs_configs,
                description=cs_desc
            ))

        if default_cs_description is None:
            raise RuntimeError(
                "Must specify 'default' (or 'Default') configuration set"
            )

        if unassigned_configurations:
            dataset.configuration_sets.append(ConfigurationSet(
                configurations=[
                    dataset.configurations[ii]
                    for ii in unassigned_configurations
                ],
                description=default_cs_description
            ))

        print(Dataset)


    def convert_units(self, units):
        """Converts the dataset units to the provided type (e.g., 'OpenKIM'"""
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
                        f"Header '{self._data[0]}' not in {self.KNOWN_HEADERS}"
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