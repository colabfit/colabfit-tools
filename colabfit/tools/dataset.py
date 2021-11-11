import os
import re
import shutil
import random
import markdown
import warnings
import itertools
import numpy as np
from tqdm import tqdm
from bson import ObjectId
from copy import deepcopy
import matplotlib.pyplot as plt
from html.parser import HTMLParser

from ase.io import write

from colabfit import (
    ATOMS_ID_FIELD, ATOMS_NAME_FIELD, ATOMS_LABELS_FIELD, EDN_KEY_MAP, OPENKIM_PROPERTY_UNITS
)
from colabfit.tools.configuration_sets import ConfigurationSet
from colabfit.tools.converters import CFGConverter, EXYZConverter, FolderConverter
from colabfit.tools.property import Property
from colabfit.tools.property_settings import PropertySettings


__all__ = [
    'Dataset',
    'load_data',
]


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
                `Configuration.info[{ATOMS_ID_FIELD}]` (ObjectId)
                `Configuration.info[{ATOMS_NAME_FIELD}]` (str)
                `Configuration.info[{ATOMS_LABELS_FIELD}]` (str)

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

        ps_regexes (dict):
            A dictionary where the key is a string that will be compiled with
            `re.compile()`, and the value is a PropertySettings object.
            configuration set. Note that whenever this dictionary is
            re-assigned, `property_settings` is re-constructed and property
            links to PropertySettings objects are re-assigned.
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


    # @property
    # def property_map(self):
    #     return self._property_map

    # @property_map.setter
    # def property_map(self, property_map):
    #     # clean_map = {}
    #     for key in property_map:
    #         # clean_map[key] = {}
    #         for key2 in ['field', 'units']:
    #             if key2 not in property_map[key]:
    #                 raise RuntimeError(
    #                     'Missing "{}" in property_map["{}"]'.format(key2, key)
    #                 )

    #     self._property_map = property_map


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
        # self.refresh_config_labels()


    @property
    def cs_regexes(self):
        return self._cs_regexes

    @cs_regexes.setter
    def cs_regexes(self, regex_dict):
        """IMPORTANT: use re-assignment instead of `del`, `.pop`, `.update()`"""
        if len(regex_dict) == 0:
            regex_dict = {'default': 'Default configuration set'}

        self._cs_regexes = regex_dict
        # self.refresh_config_sets()


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
# Name

{}

# Authors

{}

# Links

{}

# Description

{}

# Summary
|||
|---|---|
|Chemical systems|{}|
|Element ratios|{}|
|# of configurations|{}|
|# of atoms|{}|

# Data

|||
|---|---|
|Elements|{}|
|File|[{}]({})|
|Format|{}|
|Name field|{}|

# Properties

|Name|Field|Units
|---|---|---|
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

            # Write the markdown file
            with open(html_file_name, 'w') as html:
                html.write(
                    template.format(
                        self.name,
                        '\n\n'.join(self.authors),
                        '\n\n'.join(self.links),
                        self.description,
                        ', '.join(self.chemical_systems),
                        ', '.join(['{} ({:.1f}%)'.format(e, er*100) for e, er in zip(self.elements, self.elements_ratios)]),
                        len(self.configurations),
                        self.n_sites,
                        ', '.join(self.elements),
                        data_file_name, data_file_name,
                        data_format,
                        name_field,
                        '\n'.join('| {} | {} | {} |'.format(k, v['field'], v['units']) for k,v in self.property_map.items()),
                        '\n'.join('| `{}` | {} | {} | {} | {} |'.format(regex.replace('|', '\|'), pso.method, pso.description, ', '.join(pso.labels), ', '.join('[{}]({})'.format(f, f) for f in pso.files)) for regex, pso in self.ps_regexes.items()),
                        '\n'.join('| `{}` | {} | {} | {} |'.format(regex.replace('|', '\|'), desc, cs.n_configurations, cs.n_sites) for cs, (regex, desc) in zip(self.configuration_sets, self.cs_regexes.items())),
                        '\n'.join('| `{}` | {} | {} |'.format(regex.replace('|', '\|'), ', '.join(labels), ', '.join([str(self.co_labels_counts[self.co_labels.index(l)]) for l in labels])) for regex, labels in self.co_label_regexes.items()),
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
                conf.info[ATOMS_LABELS_FIELD] = ' '.join(
                    conf.info[ATOMS_LABELS_FIELD]
                )

                conf.info[ATOMS_ID_FIELD] = str(
                    conf.info[ATOMS_ID_FIELD]
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
                    conf.info[ATOMS_LABELS_FIELD].split(' ')
                )

                conf.info[ATOMS_ID_FIELD] = ObjectId(
                    conf.info[ATOMS_ID_FIELD]
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

        dataset.configurations = load_data(
            file_path=os.path.join(base_path, data_info['File'][1]),
            file_format=data_info['Format'],
            name_field=data_info['Name field'],
            elements=elements,
            default_name=parser.data['Name'],
            verbose=verbose
        )

        # Extract labels and trigger label refresh for configurations
        dataset.co_label_regexes = {
            l[0].replace('\|', '|'):
                [_.strip() for _ in l[1].split(',')]
                for l in parser.data['Configuration labels'][1:]
        }

        # Extract configuration sets and trigger CS refresh
        dataset.cs_regexes = {
            l[0].replace('\|', '|'):
                l[1]for l in parser.data['Configuration sets'][1:]
        }

        # # Map property fields to supplied names
        # for row in parser.data['Properties'][1:]:
        #     dataset.rename_property(row[1], row[0])

        # Extract computed properties
        property_map = {}
        for prop in parser.data['Properties'][1:]:
            property_map[prop[0]] = {
                'field': prop[1],
                'units': prop[2],
            }

        dataset.property_map = property_map

        dataset.parse_data(convert_units=convert_units, verbose=verbose)

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
                labels=[_.strip() for _ in row[3].split(',')] if len(row)>3 else [],
                files=files,
            )

        dataset.ps_regexes = ps_regexes

        dataset.resync()

        return dataset

    # @profile
    def clean(self, verbose=False):
        """
        Pseudocode:
            - For each property
                - add the property to a dictionary where the key is the hashed
                configuration
                - if the key already exists in the dictionary, check the new
                property against all existing ones.
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

                conf_hash = hash(tuple(sorted(
                    hash(c) for c in data.configurations
                )))

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
                for regex, cs in zip(other.cs_regexes.keys(), other.configuration_sets):
                    if regex == 'default':
                        self.cs_regexes[f'{other.name}_default'] = cs.description

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
                for regex, desc in self.cs_regexes.items():
                    if regex[0] == '^': regex = regex[1:]

                    # new_cs_regexes[f'^{self.name}_' + regex] = desc
                    new_cs_regexes[regex] = desc

                for regex, desc in other.cs_regexes.items():
                    if regex[0] == '^': regex = regex[1:]

                    new_cs_regexes[f'^{other.name}_.*{regex}'] = desc

                new_cs_regexes['default'] = f'Merged {self.name} and {other.name}'

                self.cs_regexes = new_cs_regexes

                new_co_label_regexes = {}
                for regex, labels in self.co_label_regexes.items():
                    if regex[0] == '^': regex = regex[1:]

                    # new_co_label_regexes[f'^{self.name}_' + regex] = labels
                    new_co_label_regexes[regex] = labels

                for regex, labels in other.co_label_regexes.items():
                    if regex[0] == '^': regex = regex[1:]

                    new_co_label_regexes[f'^{other.name}_.*{regex}'] = deepcopy(labels)

                self.co_label_regexes = new_co_label_regexes

                new_ps_regexes = {}
                for regex, pso in self.ps_regexes.items():
                    if regex[0] == '^': regex = regex[1:]

                    # new_ps_regexes[f'^{self.name}_' + regex] = pso
                    new_ps_regexes[regex] = pso

                for regex, pso in other.ps_regexes.items():
                    if regex[0] == '^': regex = regex[1:]

                    new_ps_regexes[f'^{other.name}_.*{regex}'] = deepcopy(pso)

                self.ps_regexes = new_ps_regexes

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
        Pseudocode:
            - convert everything to the same units
            - merge authors/links
            - warn if overlap (maybe this should be done in attach()?)
                - tell me which dataset has an overlap with which other (disjoint)
                - optionally merge subset datasets
            - check if conflicting CO labels, CS regexes, or PS regexes
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


    def rename_property(self, old_name, new_name):
        """Renames old_name field to new_name in atoms.info and atoms.arrays"""

        if old_name == new_name: return

        for conf in self.configurations:
            if old_name in conf.info:
                conf.info[new_name] = conf.info[old_name]
                del conf.info[old_name]

            if old_name in conf.arrays:
                conf.arrays[new_name] = conf.arrays[old_name]
                del conf.arrays[old_name]


    def apply_transformations(self, tform_dict):
        """
        Args:
            tform_dict (dict):
                key = property field to apply transformations to
                value = Transformation object
        """

        if self.is_parent_dataset:
            for data in self.data:
                data.apply_transformations(tform_dict)
        else:
            for data in self.data:
                for key in tform_dict:
                    edn_key = EDN_KEY_MAP.get(key, key)

                    if edn_key in data.edn:
                        data.edn[edn_key]['source-value'] = tform_dict[key](
                            data.edn[edn_key]['source-value'],
                            data.configurations
                        )


    def parse_data(self, convert_units=False, verbose=False):
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

        for ci, conf in enumerate(tqdm(
            self.configurations,
            desc='Parsing data',
            disable=not verbose
            )):

            try:
                self.data.append(Property.EFS(
                    conf, map_copy, instance_id=ci+1,
                    convert_units=convert_units
                ))
            except Exception as e:
                raise RuntimeError(
                    'Caught exception while parsing data entry {}: {}\n{}'.format(
                        ci, e, conf.info
                    )
                )

        if convert_units:
            for key in self.property_map:
                self.property_map[key]['units'] = OPENKIM_PROPERTY_UNITS[key]


    def refresh_property_map(self, verbose=False):
        property_map = {}

        for data in tqdm(
            self.data,
            desc='Refreshing properties',
            disable=not verbose
            ):
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
            conf.info[ATOMS_LABELS_FIELD] = set()

    def delete_config_label_regex(self, regex):
        label_set = self.co_label_regexes.pop(regex)

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
            self.co_label_regexes.items(),
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

        self.property_settings = list(self.ps_regexes.values())

        used = {ps_regex: False for ps_regex in self.ps_regexes}

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
                for ps_regex, pso in self.ps_regexes.items():
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
                self.cs_regexes.items(),
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
                        "Must specify 'default' (or 'Default') configuration set"
                    )

                self.configuration_sets.append(ConfigurationSet(
                    configurations=[
                        self.configurations[ii]
                        for ii in unassigned_configurations
                    ],
                    description=default_cs_description
                ))
            else:

                no_default_configs = 'No configurations were added to the default '\
                    'CS. "default" was removed from the regexes.'
                warnings.warn(no_default_configs)

                if 'default' in self.cs_regexes:
                    del self.cs_regexes['default']

        else:
            for data in self.data:
                data.refresh_config_sets()
                self.configuration_sets += data.configuration_sets


    def attach_dataset(self, dataset, supersede_existing=False):
        """
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

                for l, lc in zip(data.pso_labels, data.pso_labels_counts):
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

        self.pso_labels = sorted(list(pso_labels.keys()))
        self.pso_labels_counts = [int(pso_labels[l]) for l in self.pso_labels]


    def convert_units(self):
        """Converts the dataset units to the provided type (e.g., 'OpenKIM'"""
        for data in self.data:
            data.convert_units()


    def isdisjoint(self, other):

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


    def issubset(self, other):

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


    def issuperset(self, other):
        return other.issubset(self)


    def __eq__(self, other):
        return self.issubset(other) and other.issubset(self)


    def __ne__(self, other):
        return not self == other


    def get_data(self, property_field):
        if property_field == 'energy':
            property_field = 'unrelaxed-potential-energy'
        if property_field == 'forces':
            property_field = 'unrelaxed-potential-forces'
        if property_field == 'stress':
            property_field = 'unrelaxed-cauchy-stress'

        if self.is_parent_dataset:
            return list(itertools.chain.from_iterable(
                data.get_data(property_field) for data in self.data
            ))
        else:
            return [
                np.atleast_1d(d[property_field]['source-value']) for d in self.data
            ]


    def get_statistics(self, property_field):
        """
        Builds a `data` array by extracting the values of `property_field' for
        each entry in the dataset, wrapping them in a numpy array, and
        concatenating them all together. Then returns statistics on the
        resultant array.

        Returns:
            results (dict):
                {
                    'average': np.average(data),
                    'std': np.std(data),
                    'min': np.min(data),
                    'max': np.max(data),
                    'average_abs': np.average(np.abs(data))
                }
        """

        data = np.concatenate(self.get_data(property_field))

        return {
            'average': np.average(data),
            'std': np.std(data),
            'min': np.min(data),
            'max': np.max(data),
            'average_abs': np.average(np.abs(data)),
        }


    def plot_histograms(self, fields, yscale=None):
        """
        Generates histograms of the given fields.
        """

        nfields = len(fields)

        if yscale is None:
            yscale = ['linear']*nfields

        nrows = max(1, nfields//3)
        ncols = min(3, nfields%3)

        fig, ax = plt.subplots(nrows, ncols, figsize=(6*ncols, 4*nrows))

        for i, (prop, ys) in enumerate(zip(fields, yscale)):
            data = np.concatenate(self.get_data(prop)).ravel()

            c = i % 3
            r = i // 3

            if nrows > 1:
                ax[r][c].hist(data, bins=100)
                ax[r][c].set_title(prop)
                ax[r][c].set_yscale(ys)
            else:
                ax[c].hist(data, bins=100)
                ax[c].set_title(prop)
                ax[c].set_yscale(ys)

        # return fig


    def dataset_from_config_sets(self, cs_ids, exclude=False, verbose=False):
        if isinstance(cs_ids, int):
            cs_ids = [cs_ids]

        self.check_if_is_parent_dataset()

        if self.is_parent_dataset:
            raise RuntimeError(
                "Can't extract CS from parent Dataset"
            )

        if exclude:
            cs_ids = [
                _ for _ in range(len(self.configuration_sets))
                if _ not in cs_ids
            ]

        cs_regexes = {}
        config_sets = []
        for j, (regex, cs) in enumerate(zip(self.cs_regexes, self.configuration_sets)):
            add = (j in cs_ids and not exclude) or (j not in cs_ids and exclude)
            if add:
                cs_regexes[regex] = cs.description
                config_sets.append(cs)

        cs_regexes['default'] = '\n'.join(
            '{}: {}'.format(i, cs.description) for i, cs in enumerate(config_sets)
        )

        ds = Dataset('{} configuration sets: {}'.format(self.name, cs_ids))

        ds.cs_regexes = cs_regexes

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

        return ds


    def print_config_sets(self):
        for i, (regex, cs) in enumerate(zip(
            self.cs_regexes, self.configuration_sets
            )):

            print(f'{i} (n_sites={cs.n_sites}, regex="{regex}"): {cs.description}')


    def train_test_split(self, train_frac):
        # Initialize the train/test datasets
        train = Dataset(self.name+'-train')
        test  = Dataset(self.name+'-test')

        # Copy over all of the important information
        train.cs_regexes        = dict(self.cs_regexes)
        train.co_label_regexes  = dict(self.co_label_regexes)
        train.ps_regexes        = dict(self.ps_regexes)

        test.cs_regexes        = dict(self.cs_regexes)
        test.co_label_regexes  = dict(self.co_label_regexes)
        test.ps_regexes        = dict(self.ps_regexes)

        train.authors        = list(self.authors)
        train.links          = list(self.links)
        train.description    = self.description
        train.property_map = dict(self.property_map)

        test.authors        = list(self.authors)
        test.links          = list(self.links)
        test.description    = self.description
        test.property_map = dict(self.property_map)

        # Shuffle/split the data and add it to the train/test datasets
        n = len(self.data)

        indices = np.arange(n)
        random.shuffle(indices)

        train_num = int(train_frac*n)
        train_indices = indices[:train_num]
        test_indices  = indices[train_num:]

        train.data = deepcopy([self.data[i] for i in train_indices])
        test.data  = deepcopy([self.data[i] for i in test_indices])

        # Extract the configurations
        train.configurations = list(itertools.chain.from_iterable([
            d.configurations for d in train.data
        ]))

        test.configurations = list(itertools.chain.from_iterable([
            d.configurations for d in test.data
        ]))

        return train, test


    def filter(self, filter_type, filter_fxn):
        """
        A helper function for filtering on a Dataset. A filter is specified by
        providing  a `filter_type` and a `filter_fxn`. In the case of a parent
        dataset, the filter is applied to each of the children individually.

        Examples:

        ```
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
        ```

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

        Returns:
            dataset (Dataset):
                A Dataset object constructed by applying the specified filter,
                extracting any objects linked to the filtered object, then
                copying over `property_map`, `co_label_regexes`, `cs_regexes`,
                and `ps_regexes`.
        """

        if self.is_parent_dataset:
            # TODO: this is completely untested

            parent = Dataset('filtered')

            for ds in self.data:
                parent.attach_dataset(ds.filter(filter_type, filter_fxn))

            ds.property_map = self.property_map
            ds.cs_regexes = self.cs_regexes
            ds.co_label_regexes = self.co_label_regexes
            ds.ps_regexes = self.ps_regexes

            parent.resync()

            return parent

        ds = Dataset('filtered')

        data = []
        configurations = set()

        if filter_type == 'data':
            # Append any matching data entries, and their linked configurations

            for d in self.data:
                if filter_fxn(d):
                    data.append(d)
                    for c in d.configurations:
                        configurations.add(c)

        elif filter_type == 'configurations':
            # Append any matching configurations, and their linked data
            if len(self.data) == 0:
                configurations = filter(filter_fxn, self.configurations)
            else:
                for d in self.data:
                    for c in d.configurations:
                        if filter_fxn(c):
                            data.append(d)
                            configurations.add(c)

        ds.data = deepcopy(data)
        ds.configurations = deepcopy(list(configurations))

        ds.property_map = deepcopy(self.property_map)
        ds.cs_regexes = deepcopy(self.cs_regexes)
        ds.co_label_regexes = deepcopy(self.co_label_regexes)
        ds.ps_regexes = deepcopy(self.ps_regexes)

        ds.resync()

        return ds


    def __str__(self):
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

        return template.format(
            self.name,
            '\n\t'.join(self.authors),
            '\n\t'.join(self.links),
            self.description,
            '\n\t'.join(self.methods),
            '\n\t'.join('{}: {}'.format(k, self.property_map[k]['units']) for k in self.property_map),
            self.n_configurations,
            self.n_sites,
            '\n\t'.join('{}\t({:.1f}% of sites)'.format(e, er*100) for e, er in zip(self.elements, self.elements_ratios)),
            '\n\t'.join(self.chemical_systems),
            'Total: {}'.format(self.n_properties),
            '\n\t'.join('{}: {}'.format(l, lc) for l, lc in zip(self.property_fields, self.property_counts)),
            '\n\t'.join('{}: {}'.format(l, lc) for l, lc in zip(self.pso_labels, self.pso_labels_counts)),
            '\n\t'.join('{}: {}'.format(l, lc) for l, lc in zip(self.co_labels, self.co_labels_counts)),
            '\n\t'.join('{}: {}'.format(i, cs.description) for i, cs in enumerate(self.configuration_sets)),
        )


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


class DatasetParser(HTMLParser):

    KNOWN_HEADERS = [
        'Name',
        'Authors',
        'Links',
        'Description',
        'Data',
        'Summary',
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