import os
import re
import json
import shutil
import random
from kim_property.create import PROPERTY_NAME_TO_PROPERTY_ID
import markdown
import traceback
import warnings
import itertools
import numpy as np
from tqdm import tqdm
from hashlib import sha512
from bson import ObjectId
from copy import deepcopy
# import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from ase.io import write

from kim_property.create import KIM_PROPERTIES
# from kim_property import get_properties
# available_kim_properties = get_properties()

from colabfit import (
    HASH_SHIFT,
    ATOMS_CONSTRAINTS_FIELD, ATOMS_ID_FIELD, ATOMS_NAME_FIELD,
    ATOMS_LABELS_FIELD, EDN_KEY_MAP,
    OPENKIM_PROPERTY_UNITS, DEFAULT_PROPERTY_NAME
)
from colabfit.tools.configuration_sets_old import ConfigurationSet
from colabfit.tools.converters import CFGConverter, EXYZConverter, FolderConverter
from colabfit.tools.property import (
    Property, PropertyParsingError
)
from colabfit.tools.property_settings import PropertySettings
from colabfit.tools.transformations import BaseTransform
from colabfit.tools.dataset_parser import (
    DatasetParser, MarkdownFormatError, BadTableFormatting
)


__all__ = [
    'Dataset',
    'load_data',
]


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

        configurations (list):
            A list of Configuration objects.

        data (list):
            A list of Property objects OR a list of Dataset objects.

        property_fields (list):
            A list of strings specifying the property fields that can be
            accessed using things like get_data() and apply_transformation()

        property_counts (list):
            A list of integers counting the number of instances of each property
            in `property_fields`. Matches the order of `property_fields`

        configuration_sets (list):
            List of ConfigurationSet objects defining groups of configurations

        property_settings (list):
            A list of ProprtySetting objects collected by taking the settings
            linked to by all entries in `properties`.

        configuration_labels (list):
            A list of all configuration labels. Same as
            `sorted(list(self.configuration_label_regexes.keys()))`

        configuration_labels_counts (list):
            A list of integers indicating the number of occurrences of each
            configuration label. Matches the order of
            `self.configuration_labels`

        custom_definitions (dict):
            key = the name of a locally-defined property

            value = the path to an EDN file containing the property definition

        property_settings_labels (list):
            A list of all property settings labels.

        property_settings_labels_counts (list):
            A list of integers indicating the number of occurrences of each
            property settings label. Matches the order of
            `self.property_settings_labels`

        elements (list):
            A list of strings of element names present in the dataset

        elements_ratios (list):
            A list of floats; the total concentration of each element, given as
            a fraction of the total number of atoms in the dataset

        chemical_systems (list):
            A list of strings of chemical systems present in the dataset

        n_configurations (int):
            `len(self.configurations)`

        n_sites (int):
            `sum([len(co) for co in self.configurations])`

        methods (list):
            `[data.methods for self.data]`

        n_properties (int):
            Total number of properties in the dataset.
    """

    def __init__(
        self,
        name='',
        authors=None,
        links=None,
        description='',
        configurations=None,
        data=None,
        property_map=None,
        configuration_label_regexes=None,
        configuration_set_regexes=None,
        property_settings_regexes=None,
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

        self.configuration_sets = []

        self.custom_definitions = {}

        self._definitions_added_to_kim = []

        if property_map is None: property_map = {}
        self.property_map = property_map

        if configuration_label_regexes is None: configuration_label_regexes = {}
        if configuration_set_regexes is None:
            configuration_set_regexes  = {'default': 'Default configuration set'}
        if property_settings_regexes is None: property_settings_regexes = {}

        self.configuration_set_regexes = configuration_set_regexes
        self.property_settings_regexes = property_settings_regexes
        self.property_settings_labels = []
        self.property_settings_labels_counts = []

        self.configuration_label_regexes   = configuration_label_regexes
        self.configuration_labels = []
        self.configuration_labels_counts = []

        if self.configurations or self.data:
            self.resync()


    def check_if_is_parent_dataset(self):
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


    def resync(self, verbose=False):

        self.check_if_is_parent_dataset()
        self.refresh_config_labels(verbose)
        self.refresh_config_sets(verbose)
        self.refresh_property_settings(verbose)
        self.aggregate_metadata(verbose)


    @property
    def property_map(self):
        """
        A dictionary with the following structure.

        .. code-block:: python

            {
                <property_name>: {
                    <property_field>: {
                        'field': <key_for_info_or_arrays>,
                        'units': <ase_readable_units>
                    }
                }
            }

        See :ref:`Parsing data` for more details.
        """
        return self._property_map


    @property_map.setter
    def property_map(self, property_map):
        """IMPORTANT: call :meth:`resync` after modifying"""

        for pname, pdict in property_map.items():
            for fname, fdict in pdict.items():
                for key in ['field', 'units']:
                    if key not in fdict:
                        raise RuntimeError(
                            'Missing "{}" in property_map["{}"]["{}"}'.format(
                                key, pname, fname
                            )
                    )

        self._property_map = property_map


    @property
    def property_settings_regexes(self):
        """
        A dictionary where the key is a string that will be compiled with
        `re.compile()`, and the value is a PropertySettings object.
        configuration set. Note that whenever :meth:`Dataset.resync` is called,
        `property_settings` is re-constructed and property links to
        PropertySettings objects are re-assigned.
        """

    @property
    def configuration_label_regexes(self):
        """
        A dictionary where the key is a string that will be compiled with
        `re.compile()`, and the value is a list of string labels. Note that
        whenever :meth:`Dataset.resync` is called, the labels provided in this
        dictionary are re-applied to the matching Configurations.
        """

        return self._configuration_label_regexes

    @configuration_label_regexes.setter
    def configuration_label_regexes(self, regex_dict):
        """IMPORTANT: call :meth:`resync` after modifying"""

        self._configuration_label_regexes = regex_dict

        for regex, labels in regex_dict.items():
            if isinstance(labels, str):
                regex_dict[regex] = set([labels])

        # self.refresh_config_labels()


    @property
    def configuration_set_regexes(self):
        """
        A dictionary where the key is a string that will be compiled with
        `re.compile()`, and the value is a description of a
        :class:`~colabfit.tools.configuration_sets.ConfigurationSet`. Note that
        whenever :meth:`Dataset.resync` is called,
        :attr:`self.configuration_sets` sets will be reconstructed.
        """
        return self._configuration_set_regexes

    @configuration_set_regexes.setter
    def configuration_set_regexes(self, regex_dict):
        """IMPORTANT: call :meth:`resync` after modifying"""

        if len(regex_dict) == 0:
            regex_dict = {'default': 'Default configuration set'}

        self._configuration_set_regexes = regex_dict
        # self.refresh_config_sets()


    @property
    def property_settings_regexes(self):
        """
        A dictionary where the key is a string that will be compiled with
        `re.compile()`, and the value is the description of the
        configuration set. Note that whenever this dictionary is
        re-assigned, `configuration_sets` is re-constructed.
        """
        return self._property_settings_regexes

    @property_settings_regexes.setter
    def property_settings_regexes(self, regex_dict):
        """IMPORTANT: call :meth:`resync` after modifying"""
        for k, v in regex_dict.items():
            if not isinstance(v, PropertySettings):
                raise RuntimeError(
                    '`_property_settings_regexes` keys must be PropertySettings objects'
                )

        self._property_settings_regexes = regex_dict
        # self.refresh_property_settings()


    def to_markdown(
        self,
        base_folder,
        html_file_name,
        data_file_name,
        data_format,
        name_field=ATOMS_NAME_FIELD,
        ):
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
# Summary
|||
|---|---|
|Chemical systems|{}|
|Element ratios|{}|
|# of configurations|{}|
|# of atoms|{}|

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

|Property|KIM field|ASE field|Units
|---|---|---|---|
{}

# Property settings

|Regex|Method|Description|Labels|Files|
|---|---|---|---|---|
{}

# Configuration sets

|Regex|Description|# of structures| # of atoms|
|---|---|---|---|
{}

# Configuration labels

|Regex|Labels|Counts|
|---|---|---|
{}
"""

        if not os.path.isdir(base_folder):
            os.mkdir(base_folder)

        self.check_if_is_parent_dataset()

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

            definition_files = {}
            definition_files['default'] = DEFAULT_PROPERTY_NAME
            for pid, definition in self.custom_definitions.items():
                if isinstance(definition, dict):
                    def_fpath = os.path.join(base_folder, f'{pid}.edn')

                    json.dump(definition, open(def_fpath, 'w'))

                    definition_files[pid] = def_fpath
                else:
                    definition_files[pid] = definition


            # Write the markdown file
            with open(html_file_name, 'w') as html:
                html.write(
                    template.format(
                        ', '.join(self.chemical_systems),
                        ', '.join(['{} ({:.1f}%)'.format(e, er*100) for e, er in zip(self.elements, self.elements_ratios)]),
                        len(self.configurations),
                        self.n_sites,
                        self.name,
                        '\n\n'.join(self.authors),
                        '\n\n'.join(self.links),
                        self.description,
                        ', '.join(self.elements),
                        data_file_name, data_file_name,
                        data_format,
                        name_field,
                        # '\n'.join('| {} | {} | {} | {} |'.format('[{}]({})'.format(pid, definition_files[pid]), fdict['field'], fdict['units']) for pid, fdict in self.property_map.items()),
                        '\n'.join('\n'.join('| {} | {} | {} | {}'.format( '[{}]({})'.format(pid, definition_files[pid]), kim_field, field_map['field'], field_map['units']) for kim_field, field_map in fdict.items()) for pid, fdict in self.property_map.items()),
                        '\n'.join('| `{}` | {} | {} | {} | {} |'.format(regex.replace('|', '\|'), pso.method, pso.description, ', '.join(pso.labels), ', '.join('[{}]({})'.format(f, f) for f in pso.files)) for regex, pso in self.property_settings_regexes.items()),
                        '\n'.join('| `{}` | {} | {} | {} |'.format(regex.replace('|', '\|'), desc, cs.n_configurations, cs.n_sites) for cs, (regex, desc) in zip(self.configuration_sets, self.configuration_set_regexes.items())),
                        '\n'.join('| `{}` | {} | {} |'.format(regex.replace('|', '\|'), ', '.join(labels), ', '.join([str(self.configuration_labels_counts[self.configuration_labels.index(l)]) for l in labels])) for regex, labels in self.configuration_label_regexes.items()),
                    )
                )

            data_file_name = os.path.join(base_folder, data_file_name)

            # Copy any PSO files
            all_file_names = []
            for pso in self.property_settings:
                for fi, f in enumerate(pso.files):
                    new_name = os.path.join(base_folder, os.path.split(f)[-1])
                    shutil.copyfile(f, new_name)

                    if new_name in all_file_names:
                        raise RuntimeError(
                            "PSO file name {} is used more than once."\
                            "Use unique file names to avoid errors".format(f)
                        )

                    all_file_names.append(new_name)
                    pso.files[fi] = new_name


            if data_format == 'xyz':
                data_format = 'extxyz'

            # Store the labels as a string, since sets arent' hashable
            images = []
            for conf in self.configurations:
                conf.info[ATOMS_LABELS_FIELD] = tuple(
                    conf.info[ATOMS_LABELS_FIELD]
                )

                conf.info[ATOMS_ID_FIELD] = str(
                    conf.info[ATOMS_ID_FIELD]
                )

                conf.info[ATOMS_CONSTRAINTS_FIELD] = tuple(
                    conf.info[ATOMS_CONSTRAINTS_FIELD]
                )

                images.append(conf)

            # Write to the data file.
            # TODO: this should use converter.write()


            write(
                data_file_name,
                images=images,
                format=data_format,
            )

            # Make sure the labels on the Dataset are still sets
            for conf in self.configurations:
                conf.info[ATOMS_LABELS_FIELD] = set(
                    conf.info[ATOMS_LABELS_FIELD]
                )

                conf.info[ATOMS_ID_FIELD] = ObjectId(
                    conf.info[ATOMS_ID_FIELD]
                )

                conf.info[ATOMS_CONSTRAINTS_FIELD] = set(
                    conf.info[ATOMS_CONSTRAINTS_FIELD]
                )


    @classmethod
    def from_markdown(
        cls, html_file_path, convert_units=False, verbose=False
        ):
        """
        Loads a Dataset from a markdown file.
        """
        base_path = os.path.split(html_file_path)[0]

        with open(html_file_path, 'r') as f:
            try:
                html = markdown.markdown(f.read(), extensions=['tables'])
            except:
                raise MarkdownFormatError(
                    "Markdown file could not be read by markdown.markdown()"
                )

        # Parse information from markdown file
        parser = DatasetParser()
        parser.feed(html)

        parser.data['Name'] = parser.get_data('Name')[0]

        data_info = dict([l for l in parser.get_data('Data') if len(l)])

        try:
            elements = [_.strip() for _ in data_info['Elements'].split(',')]
        except:
            raise BadTableFormatting(
                "Error trying to access 'Elements' row in 'Data' table."
            )

        for key in ['File', 'Format', 'Name field']:
            try:
                data_info[key]
            except KeyError:
                raise BadTableFormatting(
                    f"Could not find key '{key}' in table 'Data'"
                )
        # Build skeleton Dataset
        dataset = cls(
            name=parser.get_data('Name'),
            authors=parser.get_data('Authors'),
            links=parser.get_data('Links'),
            description=parser.get_data('Description')[0],
        )

        # Load configurations
        if data_info['Name field'] == 'None':
            data_info['Name field'] = None

        # Extract labels and trigger label refresh for configurations
        labels = parser.get_data('Configuration labels')
        dataset.configuration_label_regexes = {
            l[0].replace('\|', '|'):
                [_.strip() for _ in l[1].split(',')]
                for l in labels[1:] if l
        }

        # Extract configuration sets and trigger CS refresh
        config_sets = parser.get_data('Configuration sets')

        dataset.configuration_set_regexes = {
            l[0].replace('\|', '|'):
                l[1]for l in config_sets[1:] if l
        }

        property_map = {}
        for prop in parser.get_data('Properties')[1:]:
            pid, kim_field, ase_field, units = prop

            if pid == 'default':
                pname = pid
            elif pid in KIM_PROPERTIES:
                pname = pid
            elif isinstance(pid, tuple):
                pname = pid[0]

                edn_path = os.path.abspath(pid[1])
                dataset.custom_definitions[pname] = edn_path
                # dataset._custom_definitions[pname] = os.path.abspath(edn_path)
                # dataset._custom_definitions[pname] = os.path.join(os.getcwd(), edn_path)

            pid_dict = property_map.setdefault(pname, {})

            if kim_field in pid_dict:
                raise BadTableFormatting(
                    "Duplicate property field found"
                )

            pid_dict[kim_field] = {
                'field': ase_field,
                'units': units
            }

        dataset.property_map = property_map

        dataset.configurations = load_data(
            file_path=os.path.join(base_path, data_info['File'][1]),
            file_format=data_info['Format'],
            name_field=data_info['Name field'],
            elements=elements,
            default_name=parser.data['Name'],
            verbose=verbose
        )

        dataset.parse_data(convert_units=convert_units, verbose=verbose)

        # Extract property settings
        ps_regexes = {}
        for row in parser.get_data('Property settings')[1:]:
            files = []
            if len(row) > 4:
                for fname in row[4:]:
                    with open(fname[1], 'r') as f:
                        files.append('\n'.join([_.strip() for _ in f.readlines()]))

            ps_regexes[row[0]] = PropertySettings(
                method=row[1],
                description=row[2],
                labels=[_.strip() for _ in row[3].split(',')] if len(row)>3 else [],
                files=files,
            )

        dataset.property_settings_regexes = ps_regexes

        dataset.resync(verbose=verbose)

        return dataset

    # @profile
    def clean(self, verbose=False):
        """
        Uses hashing to compare the configurations of all properties and check
        for duplicate configurations.
        """

        duplicate_checker = {}

        self.check_if_is_parent_dataset()

        if self.is_parent_dataset:
            raise RuntimeError(
                "Can't clean parent datasets. Use `flatten()` first."
            )
        else:
            n_duplicates = 0
            for data in tqdm(
                self.data,
                desc='Cleaning data',
                disable=not verbose
                ):

                conf_hash = sha512()
                for ch in sorted([hash(c) for c in data.configurations]):
                    conf_hash.update(str(ch))

                conf_hash = str(int(conf_hash.hexdigest()[:16], 16)-HASH_SHIFT)

                if conf_hash not in duplicate_checker:
                    # Data with the same configurations doesn't exist yet
                    duplicate_checker[conf_hash] = [data]
                else:
                    # Possible data matches exist
                    duplicate = False
                    for existing in duplicate_checker[conf_hash]:
                        if existing == data:
                            duplicate = True

                    # Only add the data if it's not a duplicate
                    if not duplicate:
                        duplicate_checker[conf_hash].append(data)
                    else:
                        n_duplicates += 1

            self.data = list(itertools.chain.from_iterable(
                duplicate_checker.values()
            ))

            self.configurations = list(itertools.chain.from_iterable(
                [data.configurations for data in self.data]
            ))

            if n_duplicates:
                warnings.warn(f"Removed {n_duplicates} duplicate data entries")

            return n_duplicates


    # @profile
    def merge(self, other, clean=False):
        """
        Merges the new and current Datasets. Note that the current data
        supersedes any incoming data. This means that if incoming data points
        to an existing configuration, the incoming data is not added to the
        Dataset because it is assumed that the existing data pointing to the
        same configuration is the more important version.

        The following additional changes are made to the incoming data:
            - Configurations are renamed to prepend the name of their datasets
            - All regexes are renamed as f"^{other.name}_.*{regex}"

        Args:
            other (Dataset):
                The new dataset to be added to the existing one.

            clean (bool):
                If True, checks for duplicates after merging. Default is False.
        """

        self.check_if_is_parent_dataset()
        other.check_if_is_parent_dataset()

        if not self.is_parent_dataset:
            if not other.is_parent_dataset:
                # Both datasets are children, so just merge directly

                # Rename configurations for the default CS
                for regex, cs in zip(other.configuration_set_regexes.keys(), other.configuration_sets):
                    if regex == 'default':
                        self.configuration_set_regexes[f'{other.name}_default'] = cs.description

                        for conf in cs.configurations:
                            # Make sure the configuration still matches the regex
                            conf.info[ATOMS_NAME_FIELD] = 'default_{}'.format(
                                conf.info[ATOMS_NAME_FIELD]
                            )

                        break

                # Prepend dataset names to avoid name collisions
                for conf in other.configurations:
                    conf.info[ATOMS_NAME_FIELD] = '{}_{}'.format(
                        other.name, conf.info[ATOMS_NAME_FIELD]
                    )

                self.data += deepcopy(other.data)
                self.configurations += deepcopy(other.configurations)

                # Update data and configurations, removing duplicates
                if clean:
                    self.clean()

                # Modify regex maps to avoid collisions
                new_cs_regexes = {}
                for regex, desc in self.configuration_set_regexes.items():
                    if regex[0] == '^': regex = regex[1:]

                    # new_cs_regexes[f'^{self.name}_' + regex] = desc
                    new_cs_regexes[regex] = desc

                for regex, desc in other.configuration_set_regexes.items():
                    if regex[0] == '^': regex = regex[1:]

                    new_cs_regexes[f'^{other.name}_.*{regex}'] = desc

                new_cs_regexes['default'] = f'Merged {self.name} and {other.name}'

                self.configuration_set_regexes = new_cs_regexes

                new_co_label_regexes = {}
                for regex, labels in self.configuration_label_regexes.items():
                    if regex[0] == '^': regex = regex[1:]

                    # new_co_label_regexes[f'^{self.name}_' + regex] = labels
                    new_co_label_regexes[regex] = labels

                for regex, labels in other.configuration_label_regexes.items():
                    if regex[0] == '^': regex = regex[1:]

                    new_co_label_regexes[f'^{other.name}_.*{regex}'] = deepcopy(labels)

                self.configuration_label_regexes = new_co_label_regexes

                new_ps_regexes = {}
                for regex, pso in self.property_settings_regexes.items():
                    if regex[0] == '^': regex = regex[1:]

                    # new_ps_regexes[f'^{self.name}_' + regex] = pso
                    new_ps_regexes[regex] = pso

                for regex, pso in other.property_settings_regexes.items():
                    if regex[0] == '^': regex = regex[1:]

                    new_ps_regexes[f'^{other.name}_.*{regex}'] = deepcopy(pso)

                self.property_settings_regexes = new_ps_regexes

                # self.resync()

                self.name           = f'{self.name}_{other.name}'
                self.authors        = deepcopy(list(set(self.authors + other.authors)))
                self.links          = deepcopy(list(set(self.links + other.links)))
                self.description    = f'Merged {self.name} and {other.name}'

            else:
                # Other is a parent dataset
                raise RuntimeError(
                    "Incoming dataset is a nested dataset, but current is not."
                )
        else:  # Self is parent
            if not other.is_parent_dataset:
                raise RuntimeError(
                    "Current dataset is a nested dataset, but incoming is not."
                )
            else:
                # Merging two parent datasets
                raise NotImplementedError(
                    "Merging two nested datasets is not available yet."
                )


    def flatten(self):
        """
        Attempts to "flatten" a parent dataset by merging the Property and
        Configuration lists of all of its children.

        Author lists, reference links, and descriptions are also merged.
        """
        self.check_if_is_parent_dataset()

        if not self.is_parent_dataset:
            raise RuntimeError(
                'Cannot flatten. Dataset has no attached Datasets.'
            )

        self.resync()

        flat = Dataset(self.name)

        flat.authors        = deepcopy(self.authors)
        flat.links          = deepcopy(self.links)
        flat.description    = self.description

        for data in self.data:
            flat.merge(data, clean=True)

        flat.clean()

        return flat


    def add_configurations(self, configurations):
        n = len(self.configurations)
        for ci, conf in enumerate(configurations):
            if ATOMS_NAME_FIELD not in conf.info:
                conf.info[ATOMS_NAME_FIELD] = str(n+ci)

        self.configurations += configurations

    def rename_configuration_field(self, old_name, new_name):
        """
        Renames fields in :attr:`Configuration.info` and
        :attr:`Configuration.arrays`.

        Args:
            old_name (str): the original name of the field
            new_name (str): the new name of the field

        Returns:
            None. Modifies configuration fields in-place.
        """

        for conf in self.configurations:
            if old_name in conf.info:
                conf.info[new_name] = conf.info[old_name]
                del conf.info[old_name]
            if old_name in conf.arrays:
                conf.arrays[new_name] = conf.arrays[old_name]
                del conf.arrays[old_name]


    def rename_property(self, old_name, new_name):
        """Renames old_name field to new_name in each Property"""

        """
        This should also work for CO fields.

        TODO:
        Here's what I would actually want this function to do:
            - update the key used in property_map
            - make sure get_data and apply_transform can use the new name
            - probably update property_map everywhere
        """

        edn_key = EDN_KEY_MAP.get(old_name, old_name)

        for data in self.data:
            if old_name in data:
                data[new_name] = data[edn_key]
                del data[edn_key]


    def apply_transformation(self, field_name, tform):
        """
        Args:
            field_name (str):
                The property field name to applyl the transformation to

            tform (callable):
                A BaseTransform object or a lambda function. If a lambda
                function is supplied, it must accept a 2-tuple as input, where
                the first value will be the property field data, and the second
                value will be the list of configurations linked to the property.
        """

        if not isinstance(tform, BaseTransform):
            tform = BaseTransform(tform)

        if self.is_parent_dataset:
            for data in self.data:
                data.apply_transformation(field_name, tform)
        else:
            for data in self.data:
                edn_key = EDN_KEY_MAP.get(field_name, field_name)

                if edn_key in data.edn:
                    data.edn[edn_key]['source-value'] = tform(
                        data.edn[edn_key]['source-value'],
                        data.configurations
                    )


    def parse_data(self, convert_units=False, verbose=False):
        """
        Re-constructs :attr:`self.data` by building a list of Property objects using
        :attr:`self.property_map` and :attr:`self.configurations`. Modifies :attr:`self.data` in
        place. If :code:`convert_units==True`, then the units in :attr:`self.property_map`
        are also updated.
        """

        if len(self.property_map) == 0:
            raise RuntimeError(
                'Must set `Dataset.property_map first'
            )

        map_copy = deepcopy(self.property_map)

        self.data = []

        id_counter = 1
        for ci, conf in enumerate(tqdm(
            self.configurations,
            desc='Parsing data',
            disable=not verbose
            )):

            for pid in self.property_map:
                if pid != 'default':
                    try:
                        # if pid in KIM_PROPERTIES:
                        if pid in PROPERTY_NAME_TO_PROPERTY_ID:
                            # Existing property in OpenKIM
                            definition = PROPERTY_NAME_TO_PROPERTY_ID[pid]
                        elif pid in EDN_KEY_MAP:
                            # Existing property with a pseudonym
                            definition = EDN_KEY_MAP[pid]
                            definition = PROPERTY_NAME_TO_PROPERTY_ID[definition]
                        elif pid in self._definitions_added_to_kim:
                            # Recently added local definition with spoofing
                            definition = pid
                        elif os.path.isfile(pid):
                            # Existing local file that hasn't been added
                            definition = pid
                        else:
                            # Completely new definition
                            definition = self.custom_definitions[pid]
                            self._definitions_added_to_kim.append(pid)

                        prop = Property.from_definition(
                            pid, definition, conf, map_copy[pid],
                            instance_id=id_counter,
                            convert_units=convert_units
                        )

                        self.data.append(prop)
                        id_counter += 1
                    # except MissingPropertyFieldWarning as e:
                    #     warnings.warn(
                    #         f'Skipping configuration {ci}. {e.message}',
                    #         category=MissingPropertyFieldWarning
                    #     )
                    except Exception as e:
                        traceback.format_exc()
                        raise PropertyParsingError(
                            'Caught exception while parsing data entry {}: {}\n{}'.format(
                                ci, e, conf.info
                            )
                        )
                else:

                    try:
                        # This will raise an error if energy isnt found on the conf
                        prop = Property.Default(
                            conf, map_copy['default'], instance_id=id_counter,
                            convert_units=convert_units
                        )

                        self.data.append(prop)
                        id_counter += 1
                    except Exception as e:
                        traceback.format_exc()
                        raise PropertyParsingError(
                            'Caught exception while parsing data entry {}: {}\n{}'.format(
                                ci, e, conf.info
                            )
                        )

                if convert_units:
                    for key in self.property_map[pid]:
                        self.property_map[pid][key]['units'] = OPENKIM_PROPERTY_UNITS[key]

        self.resync()


    def refresh_property_map(self, verbose=False):

        if len(self.data) == 0:
            # Avoid erasing property_map if data hasn't been loaded yet
            return

        property_map = {}

        for data in tqdm(
            self.data,
            desc='Refreshing properties',
            disable=not verbose
            ):
            # if isinstance(data, Dataset):
            #     raise NotImplementedError("Nested datasets not supported yet.")

            property_map.setdefault(data.name, {})
            for key, val in data.property_map.items():
                if key not in property_map[data.name]:
                    property_map[data.name][key] = dict(val)
                else:
                    if val['units'] != property_map[data.name][key]['units']:
                        raise RuntimeError(
                            "Conflicting units found for property "\
                                "'{}': '{}' and '{}'".format(
                                    key, val['units'],
                                    property_map[data.name][key]['units']
                                )
                        )

        self.property_map = property_map


    def clear_config_labels(self):
        for conf in self.configurations:
            conf.info[ATOMS_LABELS_FIELD] = set()

    def delete_config_label_regex(self, regex):
        label_set = self.configuration_label_regexes.pop(regex)

        regex = re.compile(regex)
        for conf in self.configurations:
            if regex.search(conf.info[ATOMS_NAME_FIELD]):
                conf.info[ATOMS_LABELS_FIELD] -= label_set


    def refresh_config_labels(self, verbose=False):
        """
        Re-applies labels to the `ase.Atoms.info[ATOMS_LABELS_FIELD]` list.
        Note that this overwrites any existing labels on the configurations.
        """

        # TODO: one problem with the current way of displaying the labels is
        # that it shows the total count even if multiple regexes have the same
        # label

        if self.configurations is None:
            raise RuntimeError(
                "Dataset.configurations is None; must load configurations first"
            )

        # Apply configuration labels
        for co_regex, labels in tqdm(
            self.configuration_label_regexes.items(),
            desc='Refreshing configuration labels',
            disable=not verbose
            ):
            regex = re.compile(co_regex)
            used = False

            for conf in self.configurations:
                # # Remove old labels
                # conf.info[ATOMS_LABELS_FIELD] = set()

                if regex.search(conf.info[ATOMS_NAME_FIELD]):
                    used = True
                    old_set =  conf.info[ATOMS_LABELS_FIELD]

                    conf.info[ATOMS_LABELS_FIELD] = old_set.union(labels)

            if not used:
                no_labels = 'Labels regex "{}" did not match any '\
                    'configurations.'.format(regex)

                warnings.warn(no_labels)


    def refresh_property_settings(self, verbose=False):
        """
        Refresh property pointers to PSOs by matching on their linked co names
        """

        if self.data is None:
            raise RuntimeError(
                "Dataset.data is None; must load data first"
            )

        self.property_settings = list(self.property_settings_regexes.values())

        used = {ps_regex: False for ps_regex in self.property_settings_regexes}

        # Reset Property PSO pointers
        for data in tqdm(
            self.data,
            desc='Refreshing property settings',
            disable=not verbose
            ):

            if isinstance(data, Dataset):
                data.refresh_property_settings()
                self.property_settings += data.property_settings
                continue

            # Remove old pointers
            data.settings = None

            for conf in data.configurations:
                match = None
                for ps_regex, pso in self.property_settings_regexes.items():
                    regex = re.compile(ps_regex)

                    if regex.search(conf.info[ATOMS_NAME_FIELD]):

                        used[ps_regex] = True

                        if data.settings is not None:
                            raise RuntimeError(
                                'Configuration name {} matches multiple PSO '\
                                'regexes: {}'.format(
                                    conf.info[ATOMS_NAME_FIELD],
                                    [match, ps_regex]
                                )
                            )

                        match = ps_regex

                        data.settings = pso

        for regex, u in used.items():
            if not u:
                no_ps = 'PS regex "{}" did not match any '\
                    'configurations.'.format(re.compile(regex))

                warnings.warn(no_ps)



    def refresh_config_sets(self, verbose=False):
        """
        Re-constructs the configuration sets.
        """

        self.configuration_sets = []

        if not self.is_parent_dataset:
            # Build configuration sets
            default_cs_description = None
            assigned_configurations = []
            for cs_regex, cs_desc in tqdm(
                self.configuration_set_regexes.items(),
                desc='Refreshing configuration sets',
                disable=not verbose
                ):
                if cs_regex.lower() == 'default':
                    default_cs_description = cs_desc
                    continue

                regex = re.compile(cs_regex)
                cs_configs = []
                for ai, conf in enumerate(self.configurations):

                    if regex.search(conf.info[ATOMS_NAME_FIELD]):
                        cs_configs.append(conf)

                        assigned_configurations.append(ai)

                self.configuration_sets.append(ConfigurationSet(
                    configurations=cs_configs,
                    description=cs_desc
                ))

                if len(cs_configs) == 0:
                    empty_cs = 'CS regex "{}" did not match any '\
                        'configurations.'.format(regex)

                    warnings.warn(empty_cs)

            unassigned_configurations = [
                ii for ii in range(len(self.configurations))
                if ii not in assigned_configurations
            ]

            if unassigned_configurations:

                if default_cs_description is None:
                    raise RuntimeError(
                        "Must specify 'default' (or 'Default') configuration "\
                        "set if given regexes don't encompass all configurations"
                    )

                self.configuration_sets.append(ConfigurationSet(
                    configurations=[
                        self.configurations[ii]
                        for ii in unassigned_configurations
                    ],
                    description=default_cs_description
                ))
            else:

                if 'default' in self.configuration_set_regexes:
                    no_default_configs = 'No configurations were added to the default '\
                        'CS. "default" was removed from the regexes.'
                    warnings.warn(no_default_configs)

                    del self.configuration_set_regexes['default']

        else:
            for data in self.data:
                data.refresh_config_sets()
                self.configuration_sets += data.configuration_sets


    def define_configuration_set(self, fxn, desc, regex):
        """
        Args:
            fxn (callable):
                A function that is called for every item in
                `self.configurations`, returning True if the PSO should be
                applied to the data entry.

            desc (str):
                The description of the new configuration set.

            regex (str):
                The string that will be used for regex matching to identify
                the updated configurations in the future. `regex` will be
                appended to the name of each matching configuration
        """

        if regex in self.configuration_set_regexes:
            raise RuntimeError(
                "Regex '{}' already exists in `self.configuration_set_regexes`."\
                    "Update the ConfigurationSet description there instead"
            )

        if self.is_parent_dataset:
            for data in self.data:
                data.define_configuration_set(fxn, desc, regex)
        else:

            matching_cos = []
            for co in self.configurations:
                if fxn(co):
                    matching_cos.append(co)
                    co.info[ATOMS_NAME_FIELD] += regex

            self.configuration_sets.append(ConfigurationSet(
                configurations=matching_cos,
                description=desc
            ))

        self.configuration_set_regexes[regex] = desc


    def attach_configuration_labels(self, fxn, labels, regex):
        """
        Args:
            fxn (callable):
                A function that is called for every item in
                `self.configurations`, returning True if the PSO should be
                applied to the data entry.

            labels (str or list):
                The labels to be applied

            regex (str):
                The string that will be used for regex matching to identify
                the updated configurations in the future. `regex` will be
                appended to the name of each matching configuration
        """

        if regex in self.configuration_label_regexes:
            raise RuntimeError(
                "Regex '{}' already exists in `self.configuration_label_regexes`."\
                    "Update the Configuration labels there instead"
            )


        if isinstance(labels, str):
            labels = [labels]
            labels = set(labels)

        if self.is_parent_dataset:
            for data in self.data:
                data.attach_configuration_labels(fxn, labels, regex)
        else:
            for co in self.configurations:
                if fxn(co):
                    co.info[ATOMS_LABELS_FIELD] = co.info[ATOMS_LABELS_FIELD].union(labels)
                    co.info[ATOMS_NAME_FIELD] += regex

        self.configuration_label_regexes[regex] = regex


    def attach_property_settings(self, fxn, pso, regex):
        """
        Args:
            fxn (callable):
                A function that is called for every item in `self.data`,
                returning True if the PSO should be applied to the data entry.

            pso (PropertySettings):
                A property settings object

            regex (str):
                The string that will be used for regex matching to identify
                the updated data entries in the future. `regex` will be
                appended to the name of each matching configuration
        """

        if regex in self.property_settings_regexes:
            raise RuntimeError(
                "Regex '{}' already exists in `self.property_settings_regexes`."\
                    "Update the PropertySettings object there instead"
            )


        if self.is_parent_dataset:
            for data in self.data:
                data.attach_property_settings(fxn, pso, regex)
        else:
            for data in self.data:
                if fxn(data):
                    data.settings = pso

                    for co in data.configurations:
                        co.info[ATOMS_NAME_FIELD] += regex

        self.property_settings_regexes[regex] = pso



    def attach_dataset(self, dataset, supersede_existing=False):
        """
        Attaches a child dataset. Use this method instead of directly modifying
        :attr:`Dataset.data`.

        Args:
            dataset (Dataset):
                The new dataset to be added

            supersede_existing (bool):
                If True, any new data that is being added will be used to
                overwrite existing duplicate data when calling clean() or
                merge(). This is important for preserving Property metadata.
                Default is False.
        """
        if supersede_existing:
            self.data = [dataset] + self.data
            self.configurations = dataset.configurations + self.configurations
        else:
            self.data.append(dataset)
            self.configurations += dataset.configurations


    def aggregate_metadata(self, verbose=False):

        self.refresh_property_map()

        elements = {}
        self.chemical_systems = []
        self.property_fields = []

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

                for l, lc in zip(ds.configuration_labels, ds.configuration_labels_counts):
                    if l not in co_labels:
                        co_labels[l] = lc
                    else:
                        co_labels[l] += lc

                self.chemical_systems += ds.chemical_systems
                self.n_sites += ds.n_sites

            self.authors = deepcopy(list(set(itertools.chain.from_iterable(
                ds.authors for ds in self.data
            ))))

            self.links = deepcopy(list(set(itertools.chain.from_iterable(
                ds.links for ds in self.data
            ))))

            self.description = '\n'.join(
                f'{ii}: {data.description.strip()}' for ii, data in enumerate(self.data)
            )

        else:
            # Separate counter to normalize ratios taking overlap into account
            dummy_count = 0
            for cs in self.configuration_sets:
                dummy_count += cs.n_sites

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

        self.n_configurations = len(self.configurations)
        self.n_sites = sum([len(c) for c in self.configurations])

        self.elements = sorted(list(elements.keys()))
        self.elements_ratios = [
            elements[el]/dummy_count for el in self.elements
        ]

        self.chemical_systems = sorted(list(set(self.chemical_systems)))

        self.configuration_labels = sorted(list(co_labels.keys()))
        self.configuration_labels_counts = [int(co_labels[l]) for l in self.configuration_labels]

        self.methods = []
        self.n_properties = 0
        prop_counts = {}
        pso_labels = {}

        for data in tqdm(
            self.data, desc='Aggregating metadata', disable=not verbose
            ):
            if self.is_parent_dataset:
                self.methods += data.methods
                self.property_fields += data.property_fields
                self.n_properties += data.n_properties

                for pname, pc in zip(data.property_fields, data.property_counts):
                    if pname not in prop_counts:
                        prop_counts[pname] = pc
                    else:
                        prop_counts[pname] += pc

                for l, lc in zip(data.property_settings_labels, data.property_settings_labels_counts):
                    if l not in pso_labels:
                        pso_labels[l] = lc
                    else:
                        pso_labels[l] += lc
            else:
                # self.property_types.append(data.edn['property-id'])
                self.n_properties += 1

                for field in data.property_fields:
                    if field not in prop_counts:
                        prop_counts[field] = 1
                    else:
                        prop_counts[field] += 1

                if data.settings is not None:
                    for l in data.settings.labels:
                        if l not in pso_labels:
                            pso_labels[l] = 1
                        else:
                            pso_labels[l] += 1

        self.methods = list(set(pso.method for pso in self.property_settings))

        self.property_fields = list(prop_counts.keys())
        self.property_counts = [
            int(prop_counts[pname]) for pname in self.property_fields
        ]

        self.property_settings_labels = sorted(list(pso_labels.keys()))
        self.property_settings_labels_counts = [int(pso_labels[l]) for l in self.property_settings_labels]


    def convert_units(self):
        """Converts the dataset units to the provided type (e.g., 'OpenKIM'"""
        for data in self.data:
            data.convert_units()


    def isdisjoint(self, other, configurations_only=False):

        if len(self.data) == 0:
            raise RuntimeError(
                "Must load data before performing set operations"
            )

        if len(other.data) == 0:
            raise RuntimeError(
                "Must load data before performing set operations"
            )

        self.check_if_is_parent_dataset()
        other.check_if_is_parent_dataset()

        if configurations_only:
            if self.is_parent_dataset:
                super_configs1 = set(itertools.chain.from_iterable([
                    ds.configurations for ds in self.data
                ]))
            else:
                super_configs1 = set(self.configurations)

            if other.is_parent_dataset:
                super_configs2 = set(itertools.chain.from_iterable([
                    ds.configurations for ds in other.data
                ]))
            else:
                super_configs2 = set(other.configurations)

            return super_configs1.isdisjoint(super_configs2)


        else:
            if self.is_parent_dataset:
                super_data1 = set(itertools.chain.from_iterable([
                    ds.data for ds in self.data
                ]))
            else:
                super_data1 = set(self.data)

            if other.is_parent_dataset:
                super_data2 = set(itertools.chain.from_iterable([
                    ds.data for ds in other.data
                ]))
            else:
                super_data2 = set(other.data)

            return super_data1.isdisjoint(super_data2)


    def issubset(self, other, configurations_only=False):

        if len(self.data) == 0:
            raise RuntimeError(
                "Must load data before performing set operations"
            )

        if len(other.data) == 0:
            raise RuntimeError(
                "Must load data before performing set operations"
            )

        self.check_if_is_parent_dataset()
        other.check_if_is_parent_dataset()

        if configurations_only:
            if self.is_parent_dataset:
                super_configs1 = set(itertools.chain.from_iterable([
                    ds.configurations for ds in self.data
                ]))
            else:
                super_configs1 = set(self.configurations)

            if other.is_parent_dataset:
                super_configs2 = set(itertools.chain.from_iterable([
                    ds.configurations for ds in other.data
                ]))
            else:
                super_configs2 = set(other.configurations)

            return super_configs1.issubset(super_configs2)

        else:

            if self.is_parent_dataset:
                super_data1 = set(itertools.chain.from_iterable([
                    ds.data for ds in self.data
                ]))
            else:
                super_data1 = set(self.data)

            if other.is_parent_dataset:
                super_data2 = set(itertools.chain.from_iterable([
                    ds.data for ds in other.data
                ]))
            else:
                super_data2 = set(other.data)

            return super_data1.issubset(super_data2)


    def issuperset(self, other, configurations_only=False):
        return other.issubset(self, configurations_only=configurations_only)


    def __eq__(self, other):
        return self.issubset(other) and other.issubset(self)


    def get_available_configuration_fields(self):
        fields = {'info': set(), 'arrays': set()}

        for conf in self.configurations:
            for k in conf.info:
                fields['info'].add(k)
            for k in conf.arrays:
                fields['arrays'].add(k)

        return fields


    def get_configuration_field(self, field):
        results = []
        for conf in self.configurations:
            if field in conf.info:
                results.append(conf.info[field])
            elif field in conf.arrays:
                results.append(conf.arrays[field])
            else:
                continue

        return results


    def get_data(self, property_field, cs_ids=None, exclude=False, concatenate=False, ravel=False):
        """
        Returns a list of properties obtained by looping over `self.data` and
        extracting the desired field if it exists on the property.

        Note that if the field does not exist on the property, that property
        will be skipped. This means that if there are multiple properties linked
        to a single configuration, `len(get_data(...))` will not be the same as
        `len(self.configurations)`

        Args:
            property_field (str):
                The string key used to extract the property values from the
                Property objects

            cs_ids (int or list):
                The integers specifying the configuration sets to obtain the
                data from. Default is None, which returns data from all
                configuration sets.

                If `self` is a base dataset, then `cs_ids` should be a list of
                integers used for indexing `self.configuration_sets`.

                If `self` is a parent dataset, then `cs_ids` should be a list of
                2-tuples `(i, j)` where a configuration set will be indexed
                using `self.data[i].configuration_sets[j]`.

            exclude (bool):
                Only to be used when `cs_ids` is not None. If `exclude==True`,
                then data is only returned for the configuration sets that are
                _not_ in `cs_ids`.

            concatenate (bool):
                If True, calls np.concatenate() on the list before returning.
                Default is False.

            ravel (bool):
                If True, calls np.concatenate() on the list before returning.
                Default is False.

        Returns a list of Numpy arrays that were constructed by calling
        [np.atleast_1d(d[property_field]['source-value']) for d in self.data]
        """

        if cs_ids is None:
            if self.is_parent_dataset:
                cs_ids = list(itertools.chain.from_iterable([
                    [(i, j) for j in range(len(self.data[i].configuration_sets))]
                    for i in range(len(self.data))
                ]))
            else:
                cs_ids = list(range(len(self.configuration_sets)))

        if isinstance(cs_ids, int) or isinstance(cs_ids, tuple):
            cs_ids = [cs_ids]

        self.check_if_is_parent_dataset()

        if self.is_parent_dataset:
            # Break the 2-tuples into separate lists
            ds_ids, sub_cs_ids = list(zip(*cs_ids))

            rebuilt_cs_ids = {ds_i: [] for ds_i in ds_ids}

            for ds_i, cs_j in zip(ds_ids, sub_cs_ids):
                rebuilt_cs_ids[ds_i].append(cs_j)


            # if concatenate=true, each one will be an n-d array
            # if ravel=true, each one will be a 1-d array
            # otherwise, it will be a list of arbitrary objects

            tmp = [
                self.data[i].get_data(
                    property_field,
                    cs_ids=sub_list,
                    exclude=exclude,
                    concatenate=concatenate,
                    ravel=ravel,
                ) for i, sub_list in rebuilt_cs_ids.items()
            ]

            if (not concatenate) and (not ravel):
                # Then tmp will be a list of arbitrary objects
                tmp = list(itertools.chain.from_iterable(tmp))
            else:
                # tmp will either be an n-d or 1-d array; either way, concat

                tmp = np.concatenate(tmp)

                if ravel:
                    tmp = tmp.ravel()

            return tmp
        else:

            if exclude:
                cs_ids = [
                    _ for _ in range(len(self.configuration_sets))
                    if _ not in cs_ids
                ]

            # property_field = EDN_KEY_MAP.get(property_field, property_field)

            # Extract only the data from the given CSs
            config_sets = []
            for j, cs in enumerate(self.configuration_sets):
                add = (j in cs_ids and not exclude) or (j not in cs_ids and exclude)
                if add:
                    config_sets.append(cs)

            configurations = itertools.chain.from_iterable([
                cs.configurations for cs in config_sets
            ])

            quickset = set(configurations)

            tmp = []
            for d in self.data:
                add = True

                for conf in d.configurations:
                    if conf not in quickset:
                        add = False

                if add:
                    v = d.get_data(property_field)
                    if v is not np.nan:
                        tmp.append(v)

            if concatenate:
                tmp = np.concatenate(tmp)

            if ravel:
                tmp = np.concatenate(tmp).ravel()

            return tmp


    def get_statistics(self, property_field):
        """
        Builds an list by extracting the values of `property_field` for
        each entry in the dataset, wrapping them in a numpy array, and
        concatenating them all together. Then returns statistics on the
        resultant array.

        Returns:
            results (dict)::
                ..code-block::
                    {'average': np.average(data), 'std': np.std(data), 'min': np.min(data), 'max': np.max(data), 'average_abs': np.average(np.abs(data))}
        """

        data = np.concatenate(self.get_data(property_field))

        return {
            'average': np.average(data),
            'std': np.std(data),
            'min': np.min(data),
            'max': np.max(data),
            'average_abs': np.average(np.abs(data)),
        }


    def plot_histograms(self, fields=None, cs_ids=None, xscale='linear', yscale='linear'):
        """
        Generates histograms of the given fields.
        """

        if fields is None:
            fields = self.property_fields

        nfields = len(fields)

        nrows = max(1, int(np.ceil(nfields/3)))
        ncols = max(3, nfields%3)

        fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=fields)

        for i, prop in enumerate(fields):
            data = self.get_data(prop, cs_ids=cs_ids, ravel=True)

            nbins = max(data.shape[0]//1000, 100)

            c = i % 3
            r = i // 3

            if nrows > 1:
                fig.add_trace(
                    go.Histogram(x=data, nbinsx=nbins),
                    row=r+1, col=c+1,
                )
            else:
                fig.add_trace(
                    go.Histogram(x=data, nbinsx=nbins),
                    row=1, col=c+1
                )

        fig.update_layout(showlegend=False)
        fig.update_xaxes(type=xscale)
        fig.update_yaxes(type=yscale)

        return fig


    def dataset_from_config_sets(self, cs_ids, exclude=False, verbose=False):
        """
        Returns a new dataset that only contains the specified configuration
        sets.

        Args:
            cs_ids (int or list):
                The index of the configuration set(s) to use for building the
                new dataset. If `self` is a parent dataset, then `cs_ids` shoud
                either be a 2-tuple or a list of 2-tuples (i, j), where the
                configuration sets will be indexed as
                `dataset.data[i].configuration_sets[j]`.

            exclude (bool):
                If False, builds a new dataset using all of the configuration
                sets _except_ those specified by `cs_ids`. Default is False.

            verbose (bool):
                If True, prints progress. Default is False


        Returns:
            ds (Dataset):
                The new dataset. If `self` is a parent dataset, then the new
                dataset will also be a parent dataset.
        """

        if isinstance(cs_ids, int) or isinstance(cs_ids, tuple):
            cs_ids = [cs_ids]

        self.check_if_is_parent_dataset()

        ds = Dataset('{} configuration sets: {}'.format(self.name, cs_ids))
        ds.description = ds.name

        if self.is_parent_dataset:
            # Break the 2-tuples into separate lists
            ds_ids, sub_cs_ids = list(zip(*cs_ids))

            rebuilt_cs_ids = {ds_i: [] for ds_i in ds_ids}

            for ds_i, cs_j in zip(ds_ids, sub_cs_ids):
                rebuilt_cs_ids[ds_i].append(cs_j)

            for ds_i, sub_list in rebuilt_cs_ids.items():
                ds.attach_dataset(self.data[ds_i].dataset_from_config_sets(
                    cs_ids=sub_list, exclude=exclude, verbose=verbose
                ))
        else:

            if exclude:
                cs_ids = [
                    _ for _ in range(len(self.configuration_sets))
                    if _ not in cs_ids
                ]

            cs_regexes = {}
            config_sets = []
            for j, (regex, cs) in enumerate(zip(self.configuration_set_regexes, self.configuration_sets)):
                add = (j in cs_ids and not exclude) or (j not in cs_ids and exclude)
                if add:
                    cs_regexes[regex] = cs.description
                    config_sets.append(cs)

            cs_regexes['default'] = '\n'.join(
                '{}: {}'.format(i, cs.description) for i, cs in enumerate(config_sets)
            )

            ds.configuration_set_regexes = cs_regexes

            ds.configurations = deepcopy(list(itertools.chain.from_iterable([
                cs.configurations for cs in config_sets
            ])))

            sub_data = []

            quickset = set(ds.configurations)

            for data in tqdm(
                self.data,
                desc='Extracting configuration sets',
                disable=not verbose
                ):
                add = True
                for conf in data.configurations:
                    if conf not in quickset:
                        add = False

                if add:
                    sub_data.append(data)

            ds.data = deepcopy(sub_data)

        ds.authors        = list(self.authors)
        ds.links          = list(self.links)
        ds.property_map   = deepcopy(self.property_map)

        ds.resync()

        return ds


    def print_configuration_sets(self):
        if self.is_parent_dataset:
            for i, ds in enumerate(self.data):
                for j, (regex, cs) in enumerate(zip(
                    ds.configuration_set_regexes, ds.configuration_sets
                    )):

                    print(f'DS={i}, CS={j} (n_configurations={cs.n_configurations}, n_sites={cs.n_sites}, regex="{regex}"):\n{cs.description}\n')
        else:
            for i, (regex, cs) in enumerate(zip(
                self.configuration_set_regexes, self.configuration_sets
                )):

                print(f'CS={i} (n_configurations={cs.n_configurations}, n_sites={cs.n_sites}, regex="{regex}"):\n{cs.description}\n')


    def train_test_split(self, train_frac, copy=False):
        # Initialize the train/test datasets
        train = Dataset(self.name+'-train')
        test  = Dataset(self.name+'-test')

        # Copy over all of the important information
        train.configuration_set_regexes        = dict(self.configuration_set_regexes)
        train.configuration_label_regexes  = dict(self.configuration_label_regexes)
        train.property_settings_regexes        = dict(self.property_settings_regexes)

        test.configuration_set_regexes        = dict(self.configuration_set_regexes)
        test.configuration_label_regexes  = dict(self.configuration_label_regexes)
        test.property_settings_regexes        = dict(self.property_settings_regexes)

        train.authors        = list(self.authors)
        train.links          = list(self.links)
        train.description    = self.description
        train.property_map   = deepcopy(self.property_map)

        test.authors        = list(self.authors)
        test.links          = list(self.links)
        test.description    = self.description
        test.property_map   = deepcopy(self.property_map)

        # Shuffle/split the data and add it to the train/test datasets
        n = len(self.data)

        indices = np.arange(n)
        random.shuffle(indices)

        train_num = int(train_frac*n)
        train_indices = indices[:train_num]
        test_indices  = indices[train_num:]

        train.data = [self.data[i] for i in train_indices]
        test.data  = [self.data[i] for i in test_indices]

        if copy:
            train.data = deepcopy(train.data)
            test.data  = deepcopy(test.data)

        # Extract the configurations
        train.configurations = list(itertools.chain.from_iterable([
            d.configurations for d in train.data
        ]))

        test.configurations = list(itertools.chain.from_iterable([
            d.configurations for d in test.data
        ]))

        return train, test


    def filter(self, filter_type, filter_fxn, copy=False, verbose=False):
        """
        A helper function for filtering on a Dataset. A filter is specified by
        providing  a `filter_type` and a `filter_fxn`. In the case of a parent
        dataset, the filter is applied to each of the children individually.

        Examples::

            # Filter based on configuration name
            regex = re.compile('example_name.*')

            filtered_dataset = dataset.filter(
                'configurations',
                lambda c: regex.search(c.info[ATOMS_NAME_FIELD])
            )

            # Filter based on maximum force component
            import numpy as np

            filtered_dataset = dataset.filter(
                'data',
                lambda p: np.max(np.abs(p.edn['unrelaxed-potential-forces']['source-value'])) < 1.0
            )

        Args:
            filter_type (str):
                One of 'configurations' or 'data'.

                If `filter_type == 'configurations'`:
                    Filters on configurations, and returns a dataset with only
                    the configurations and their linked properties.

                If `filter_type == 'data'`:
                    Filters on properties, and returns a dataset with only the
                    properties and their linked configurations.

            filter_fxn (callable):
                A callable function to use as `filter(filter_fxn)`.

            copy (bool):
                If True, deep copies all dataset attributes before returning
                filtered results. Default is False.

        Returns:
            dataset (Dataset):
                A Dataset object constructed by applying the specified filter,
                extracting any objects linked to the filtered object, then
                copying over `property_map`, `configuration_label_regexes`,
                `configuration_set_regexes`, and `property_settings_regexes`.
        """

        if self.is_parent_dataset:
            # TODO: this is completely untested

            parent = Dataset('filtered')

            for ds in self.data:
                parent.attach_dataset(ds.filter(filter_type, filter_fxn))

            ds.property_map = self.property_map
            ds.configuration_set_regexes = self.configuration_set_regexes
            ds.configuration_label_regexes = self.configuration_label_regexes
            ds.ps_regexes = self.property_settings_regexes

            if copy:
                ds.property_map = deepcopy(ds.property_map)
                ds.configuration_set_regexes = deepcopy(ds.configuration_set_regexes)
                ds.configuration_label_regexes = deepcopy(ds.configuration_label_regexes)
                ds.property_settings_regexes = deepcopy(ds.property_settings_regexes)

            parent.resync()

            return parent

        ds = Dataset('filtered')

        data = []
        configurations = set()

        if filter_type == 'data':
            # Append any matching data entries, and their linked configurations

            for d in tqdm(
                self.data,
                desc='Filtering data',
                disable=not verbose
                ):

                if filter_fxn(d):
                    data.append(d)

                    for c in d.configurations:
                        configurations.add(c)

        elif filter_type == 'configurations':
            # Append any matching configurations, and their linked data
            if len(self.data) == 0:
                configurations = filter(filter_fxn, self.configurations)
            else:

                for d in tqdm(
                    self.data,
                    desc='Filtering data',
                    disable=not verbose
                    ):
                    for c in d.configurations:
                        if filter_fxn(c):
                            data.append(d)
                            configurations.add(c)
                            break

        configurations = list(configurations)

        property_map = self.property_map
        configuration_set_regexes = self.configuration_set_regexes
        configuration_label_regexes = self.configuration_label_regexes
        property_settings_regexes = self.property_settings_regexes


        if copy:
            data = deepcopy(data)
            configurations = deepcopy(configurations)

            property_map = deepcopy(property_map)
            configuration_set_regexes = deepcopy(configuration_set_regexes)
            configuration_label_regexes = deepcopy(configuration_label_regexes)
            property_settings_regexes = deepcopy(property_settings_regexes)

        ds.data = data
        ds.configurations = configurations

        ds.property_map = property_map
        ds.configuration_set_regexes = configuration_set_regexes
        ds.configuration_label_regexes = configuration_label_regexes
        ds.property_settings_regexes = property_settings_regexes

        ds.resync(verbose=verbose)

        return ds


    def __str__(self):
        return f"Dataset(name={self.name})"


    def summary(self):
        template = """Dataset
    Name:\n\t{}\n
    Authors:\n\t{}\n
    Links:\n\t{}\n
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



        print(template.format(
            self.name,
            '\n\t'.join(self.authors),
            '\n\t'.join(self.links),
            self.description,
            '\n\t'.join(self.methods),
            '\n\t'.join( '{}:\n\t{}'.format( pid, '\n\t\t'.join( '{}: {}'.format(field, fdict[field]['units']) for field in fdict)) for pid, fdict in self.property_map.items()),
            self.n_configurations,
            self.n_sites,
            '\n\t'.join('{}\t({:.1f}% of sites)'.format(e, er*100) for e, er in zip(self.elements, self.elements_ratios)),
            '\n\t'.join(self.chemical_systems),
            'Total: {}'.format(self.n_properties),
            '\n\t'.join('{}: {}'.format(l, lc) for l, lc in zip(self.property_fields, self.property_counts)),
            '\n\t'.join('{}: {}'.format(l, lc) for l, lc in zip(self.property_settings_labels, self.property_settings_labels_counts)),
            '\n\t'.join('{}: {}'.format(l, lc) for l, lc in zip(self.configuration_labels, self.configuration_labels_counts)),
            '\n\t'.join('{}: {}'.format(i, cs.description) for i, cs in enumerate(self.configuration_sets)),
        ))




def load_data(
    file_path,
    file_format,
    name_field,
    elements,
    default_name='',
    labels_field=None,
    reader=None,
    glob_string=None,
    verbose=False,
    **kwargs,
    ):
    """
    Loads configurations as a list of ase.Atoms objects.

    Args:
        file_path (str):
            Path to the file or folder containing the data

        file_format (str):
            A string for specifying the type of Converter to use when loading
            the configurations. Allowed values are 'xyz', 'extxyz', 'cfg', or
            'folder'.

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

        reader (callable):
            An optional function for loading configurations from a file. Only
            used for `file_format == 'folder'`

        glob_string (str):
            A string to use with `Path(file_path).rglob(glob_string)` to
            generate a list of files to be passed to `self.reader`. Only used
            for `file_format == 'folder'`.

        verbose (bool):
            If True, prints progress bar.

    All other keyword arguments will be passed with
    `converter.load(..., **kwargs)`
    """

    if file_format == 'folder':
        if reader is None:
            raise RuntimeError(
                "Must provide a `reader` function when `file_format=='folder'`"
            )

        if glob_string is None:
            raise RuntimeError(
                "Must provide `glob_string` when `file_format=='folder'`"
            )


        converter = FolderConverter(reader)

        return converter.load(
            file_path,
            name_field=name_field,
            elements=elements,
            default_name=default_name,
            labels_field=labels_field,
            glob_string=glob_string,
            verbose=verbose,
            **kwargs,
        )


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
        verbose=verbose,
    )

