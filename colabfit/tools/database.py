import os
import json
import shutil
import markdown
import datetime
import warnings
import itertools
import numpy as np
from tqdm import tqdm
import multiprocessing
from copy import deepcopy
from hashlib import sha512
from getpass import getpass
from ast import literal_eval
from functools import partial
from pymongo import MongoClient, UpdateOne
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from ase.io import write as ase_write

import kim_edn
from kim_property.definition import check_property_definition
from kim_property.definition import PROPERTY_ID as VALID_KIM_ID
from kim_property.create import KIM_PROPERTIES

from colabfit import (
    HASH_LENGTH,
    HASH_SHIFT,
    ID_FORMAT_STRING,
    _CONFIGS_COLLECTION, _PROPS_COLLECTION, _PROPSETTINGS_COLLECTION,
    _CONFIGSETS_COLLECTION, _PROPDEFS_COLLECTION, _DATASETS_COLLECTION,
    ATOMS_NAME_FIELD, ATOMS_LABELS_FIELD, ATOMS_LAST_MODIFIED_FIELD
)
from colabfit.tools.configuration import Configuration, process_species_list
from colabfit.tools.property import Property
from colabfit.tools.configuration_set import ConfigurationSet
from colabfit.tools.converters import CFGConverter, EXYZConverter, FolderConverter
from colabfit.tools.dataset import Dataset
from colabfit.tools.property_settings import PropertySettings
from colabfit.tools.dataset_parser import (
    DatasetParser, MarkdownFormatError, BadTableFormatting
)

class MongoDatabase(MongoClient):
    """
    A MongoDatabase stores all of the data in Mongo documents, and
    provides additinal functionality like filtering and optimized queries.

    The Mongo database has the following structure

    .. code-block:: text

        /configurations
            _id
            atomic_numbers
            positions
            cell
            pbc
            names
            labels
            elements
            nelements
            elements_ratios
            chemical_formula_reduced
            chemical_formula_anonymous
            chemical_formula_hill
            nsites
            dimension_types
            nperiodic_dimensions
            latice_vectors
            last_modified
            relationships
                properties
                configuration_sets

        /property_definitions
            _id
            definition

        /properties
            _id
            type
            property_name
                each field in the property definition
            methods
            labels
            last_modified
            relationships
                property_settings
                configurations

        /property_settings
            _id
            method
            decription
            labels
            files
                file_name
                file_contents
            relationships
                properties

        /configuration_sets
            _id
            last_modified
            aggregated_info
                (from configurations)
                nconfigurations
                nsites
                nelements
                chemical_systems
                elements
                individual_elements_ratios
                total_elements_ratios
                labels
                labels_counts
                chemical_formula_reduced
                chemical_formula_anonymous
                chemical_formula_hill
                nperiodic_dimensions
                dimension_types
            relationships
                configurations
                datasets

        /datasets
            _id
            last_modified
            aggregated_info
                (from configuration sets)
                nconfigurations
                nsites
                nelements
                chemical_systems
                elements
                individual_elements_ratios
                total_elements_ratios
                configuration_labels
                configuration_labels_counts
                chemical_formula_reduced
                chemical_formula_anonymous
                chemical_formula_hill
                nperiodic_dimensions
                dimension_types

                (from properties)
                property_types
                property_fields
                methods
                methods_counts
                property_labels
                property_labels_counts
            relationships
                properties
                configuration_sets

    Attributes:

        database_name (str):
            The name of the Mongo database

        configurations (Collection):
            A Mongo collection of configuration documents

        properties (Collection):
            A Mongo collection of property documents

        property_definitions (Collection):
            A Mongo collection of property definitions

        property_settings (Collection):
            A Mongo collection of property setting documents

        configuration_sets (Collection):
            A Mongo collection of configuration set documents

        datasets (Collection):
            A Mongo collection of dataset documents

    """
    def __init__(
        self, database_name, nprocs=1,
        drop_database=False, user=None, pwrd=None, port=27017
        ):
        """
        Args:

            database_name (str):
                The name of the database

            nprocs (int):
                The size of the processor pool

            drop_database (bool, default=False):
                If True, deletes the existing Mongo database.

            user (str, default=None):
                Mongo server username

            pwrd (str, default=None):
                Mongo server password

            port (int, default=27017):
                Mongo server port number

        """

        self.user = user
        self.pwrd = pwrd
        self.port = port

        if user is None:
            super().__init__('localhost', self.port)
        else:
            super().__init__(
                'mongodb://{}:{}@localhost:{}/'.format(
                    self.user, self.pwrd, self.port
                )
            )

        self.database_name = database_name

        if drop_database:
            self.drop_database(database_name)

        self.configurations         = self[database_name][_CONFIGS_COLLECTION]
        self.properties             = self[database_name][_PROPS_COLLECTION]
        self.property_definitions   = self[database_name][_PROPDEFS_COLLECTION]
        self.property_settings      = self[database_name][_PROPSETTINGS_COLLECTION]
        self.configuration_sets     = self[database_name][_CONFIGSETS_COLLECTION]
        self.datasets               = self[database_name][_DATASETS_COLLECTION]

        self.nprocs = nprocs


    def insert_data(
        self,
        configurations,
        property_map=None,
        property_settings=None,
        transform=None,
        generator=False,
        verbose=True
        ):
        """
        A wrapper to Database.insert_data() which also adds important queryable
        metadata about the configurations into the Client's server.

        Note that when adding the data, the Mongo server will store the
        bi-directional relationships between the data. For example, a property
        will point to its configurations, but those configurations will also
        point back to any linked properties.

        Args:

            configurations (list or Configuration):
                The list of configurations to be added.

            property_map (dict):
                A dictionary that is used to specify how to load a defined
                property off of a configuration. Note that the top-level keys in
                the map must be the names of properties that have been
                previously defined using
                :meth:`~colabfit.tools.database.Database.add_property_definition`.

                Example:

                    .. code-block:: python

                        property_map = {
                            'energy-forces-stress': {
                                # ColabFit name: {'field': ASE field name, 'units': str}
                                'energy':   {'field': 'energy',  'units': 'eV'},
                                'forces':   {'field': 'forces',  'units': 'eV/Ang'},
                                'stress':   {'field': 'virial',  'units': 'GPa'},
                                'per-atom': {'field': 'per-atom', 'units': None},
                            
                                '_settings': {
                                    '_method': 'VASP',
                                    '_description': 'A static VASP calculation',
                                    '_files': None,
                                    '_labels': ['Monkhorst-Pack'],
                                }
                            }
                        }


                If None, only loads the configuration information (atomic
                numbers, positions, lattice vectors, and periodic boundary
                conditions).

                The '_settings' key is a special key that can be used to specify
                the contents of a PropertySettings object that will be
                constructed and linked to each associated property instance.

            property_settings (list)
                A list of PropertySettings objects.

            transform (callable, default=None):
                If provided, `transform` will be called on each configuration in
                :code:`configurations` as :code:`transform(configuration)`.
                Note that this happens before anything else is done. `transform`
                should modify the Configuration in-place.

            generator (bool, default=False):
                If True, returns a generator of the results; otherwise returns
                a list. If True, uses :code:`update_one` instead of
                :code:`bulk_write` to avoid having to store update documents in
                memory.

            verbose (bool, default=False):
                If True, prints a progress bar

        Returns:

            ids (list):
                A list of (config_id, property_id) tuples of the inserted data.
                If no properties were inserted, then property_id will be None.

        """

        if self.user is None:
            mongo_login = self.port
        else:
            mongo_login = 'mongodb://{}:{}@localhost:{}/'.format(
                self.user, self.pwrd, self.port
            )

        if property_map is None:
            property_map = {}

        ignore_keys = {
            'property-id', 'property-title', 'property-description',
            'last_modified', 'definition', '_id', '_settings'
        }

        # Sanity checks for property map
        for pname, pdict in property_map.items():
            pd_doc = self.property_definitions.find_one({'_id': pname})

            if pd_doc:
                # property_field_name, {'ase_field': ..., 'units': ...}
                for k, pd in pdict.items():
                    if k in ignore_keys:
                        continue

                    if k not in pd_doc['definition']:
                        warnings.warn(
                            'Provided field "{}" in property_map does not match '\
                            'property definition'.format(k)
                        )

                    if ('field' not in pd) or (pd['field'] is None):
                        raise RuntimeError(
                            "Must specify all 'field' sections in property_map"
                        )

                    if 'units' not in pd:
                        raise RuntimeError(
                            "Must specify all 'units' sections in "\
                            "property_map. Set value to None if no units."
                        )
            else:
                warnings.warn(
                    'Property name "{}" in property_map does not have an '\
                    'existing definition in the database.'.format(pname)
                )

        # for pso in property_settings.values():
        #     self.insert_property_settings(pso)

        if generator:
            return self._insert_data_generator(
                mongo_login=mongo_login,
                database_name=self.database_name,
                configurations=configurations,
                property_map=property_map,
                transform=transform,
                verbose=verbose
            )
        else:
            configurations = list(configurations)

            n = len(configurations)
            k = self.nprocs

            split_configs = [
                configurations[i*(n//k)+min(i, n%k):(i+1)*(n//k)+min(i+1, n%k)]
                for i in range(k)
            ]

            pfunc = partial(
                self._insert_data,
                mongo_login=mongo_login,
                database_name=self.database_name,
                property_map=property_map,
                transform=transform,
                verbose=verbose
            )

            pool = multiprocessing.Pool(self.nprocs)

            return list(itertools.chain.from_iterable(
                pool.map(pfunc, split_configs)
            ))


    @staticmethod
    def _insert_data_generator(
        configurations, database_name, mongo_login,
        property_map=None, property_settings=None, transform=None,
        verbose=False
        ):

        if isinstance(mongo_login, int):
            client = MongoClient('localhost', mongo_login)
        else:
            client = MongoClient(mongo_login)

        coll_configurations         = client[database_name][_CONFIGS_COLLECTION]
        coll_properties             = client[database_name][_PROPS_COLLECTION]
        coll_property_definitions   = client[database_name][_PROPDEFS_COLLECTION]
        coll_property_settings      = client[database_name][_PROPSETTINGS_COLLECTION]

        if isinstance(configurations, Configuration):
            configurations = [configurations]

        if property_map is None:
            property_map = {}

        if property_settings is None:
            property_settings = []

        property_definitions = {
            pname: coll_property_definitions.find_one({'_id': pname})['definition']
            for pname in property_map
        }

        ignore_keys = {
            'property-id', 'property-title', 'property-description',
            'last_modified', 'definition', '_id'
        }

        expected_keys = {
            pname: set(
                property_map[pname][f]['field']
                for f in property_definitions[pname].keys() - ignore_keys
                if property_definitions[pname][f]['required']
            )
            for pname in property_map
        }

        settings_docs   = {}

        # Add all of the configurations into the Mongo server
        ai = 1
        for atoms in tqdm(
            configurations,
            desc='Preparing to add configurations to Database',
            disable=not verbose,
            ):

            if transform:
                transform(atoms)

            cid = ID_FORMAT_STRING.format('CO', str(hash(atoms)), 0)

            processed_fields = process_species_list(atoms)

            # Add if doesn't exist, else update (since last-modified changed)
            c_update_doc =  {  # update document
                    '$setOnInsert': {
                        '_id': cid,
                        'atomic_numbers': atoms.get_atomic_numbers().tolist(),
                        'positions': atoms.get_positions().tolist(),
                        'cell': np.array(atoms.get_cell()).tolist(),
                        'pbc': atoms.get_pbc().astype(int).tolist(),
                        'elements': processed_fields['elements'],
                        'nelements': processed_fields['nelements'],
                        'elements_ratios': processed_fields['elements_ratios'],
                        'chemical_formula_reduced': processed_fields['chemical_formula_reduced'],
                        'chemical_formula_anonymous': processed_fields['chemical_formula_anonymous'],
                        'chemical_formula_hill': atoms.get_chemical_formula(),
                        'nsites': len(atoms),
                        'dimension_types': atoms.get_pbc().astype(int).tolist(),
                        'nperiodic_dimensions': int(sum(atoms.get_pbc())),
                        'lattice_vectors': np.array(atoms.get_cell()).tolist(),
                    },
                    '$set': {
                        'last_modified': datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
                    },
                    '$addToSet': {
                        'names': {
                            '$each': list(atoms.info[ATOMS_NAME_FIELD])
                        },
                        'labels': {
                            '$each': list(atoms.info[ATOMS_LABELS_FIELD])
                        },
                        'relationships.properties': {
                            '$each': []
                        }
                    }
                }

            available_keys = set().union(atoms.info.keys(), atoms.arrays.keys())

            pid = None

            for pname, pmap in property_map.items():

                # Pre-check to avoid having to delete partially-added properties
                missing_keys = expected_keys[pname] - available_keys
                if missing_keys:
                    warnings.warn(
                        "Configuration is missing keys {} during "\
                        "insert_data. Available keys: {}. Skipping".format(
                            missing_keys, available_keys
                        )
                        # "Configuration {} is missing keys {} during "\
                        # "insert_data. Available keys: {}. Skipping".format(
                        #     ai, missing_keys, available_keys
                        # )
                    )
                    continue

                prop = Property.from_definition(
                    definition=property_definitions[pname],
                    configuration=atoms,
                    property_map=pmap
                )

                # NOTE: property ID does not depend upon linked settings
                pid = ID_FORMAT_STRING.format('PI', str(hash(prop)), 0)

                # Attach property settings, if any were given
                labels = []
                methods = []
                settings_id = []
                if pname in property_settings:
                    settings_id = str(hash(property_settings[pname]))

                    labels = list(property_settings[pname].labels)
                    methods = [property_settings[pname].method]

                    # Tracker for updating PSO->PR relationships
                    if settings_id in settings_docs:
                        settings_docs[settings_id].append(pid)
                    else:
                        settings_docs[settings_id] = [pid]

                    settings_id = [settings_id]

                # Prepare the EDN document
                setOnInsert = {}
                for k in property_map[pname]:
                    if k not in prop.keys():
                        # To allow for missing non-required keys.
                        # Required keys checked for in Property.from_definition
                        continue

                    if isinstance(prop[k]['source-value'], (int, float, str)):
                        # Add directly
                        setOnInsert[k] = {
                            'source-value': prop[k]['source-value']
                        }
                    else:
                        # Then it's array-like and should be converted to a list
                        setOnInsert[k] = {
                            'source-value': np.atleast_1d(
                                prop[k]['source-value']
                            ).tolist()
                        }

                    if 'source-unit' in prop[k]:
                        setOnInsert[k]['source-unit'] = prop[k]['source-unit']

                    p_update_doc = {
                        '$addToSet': {
                            'methods': {'$each': methods},
                            'labels': {'$each': labels},
                            # PR -> PSO pointer
                            'relationships.property_settings': {
                                '$each': settings_id
                            },
                            'relationships.configurations': cid,
                        },
                        '$setOnInsert': {
                            '_id': pid,
                            'type': pname,
                            pname: setOnInsert
                        },
                        '$set': {
                            'last_modified': datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
                        }
                    }

                coll_properties.update_one(
                    {'_id': pid},
                    p_update_doc,
                    upsert=True
                )

                c_update_doc['$addToSet']['relationships.properties']['$each'].append(
                    pid
                )

                yield (cid, pid)

            coll_configurations.update_one(
                {'_id': cid},
                c_update_doc,
                upsert=True
            )

            if not pid:
                # Only yield if something wasn't yielded earlier
                yield (cid, pid)

            ai += 1

        if settings_docs:
            coll_property_settings.bulk_write(
                [
                    UpdateOne(
                        {'_id': sid},
                        {'$addToSet': {'relationships.properties': {'$each': lst}}}
                    ) for sid, lst in settings_docs.items()
                ],
                ordered=False
            )

        client.close()


    @staticmethod
    def _insert_data(
        configurations, database_name, mongo_login,
        property_map=None, transform=None,
        verbose=False
        ):

        if isinstance(mongo_login, int):
            client = MongoClient('localhost', mongo_login)
        else:
            client = MongoClient(mongo_login)

        coll_configurations         = client[database_name][_CONFIGS_COLLECTION]
        coll_properties             = client[database_name][_PROPS_COLLECTION]
        coll_property_definitions   = client[database_name][_PROPDEFS_COLLECTION]
        coll_property_settings      = client[database_name][_PROPSETTINGS_COLLECTION]

        if isinstance(configurations, Configuration):
            configurations = [configurations]

        if property_map is None:
            property_map = {}

        # if property_settings is None:
        #     property_settings = []

        property_definitions = {
            pname: coll_property_definitions.find_one({'_id': pname})['definition']
            for pname in property_map
        }

        ignore_keys = {
            'property-id', 'property-title', 'property-description',
            'last_modified', 'definition', '_id', '_settings'
        }

        expected_keys = {
            pname: set(
                property_map[pname][f]['field']
                for f in property_definitions[pname].keys() - ignore_keys
                if property_definitions[pname][f]['required']
            )
            for pname in property_map
        }

        insertions = []

        config_docs     = []
        property_docs   = []
        settings_docs   = []

        # Add all of the configurations into the Mongo server
        ai = 1
        for atoms in tqdm(
            configurations,
            desc='Preparing to add configurations to Database',
            disable=not verbose,
            ):

            if transform:
                transform(atoms)

            cid = ID_FORMAT_STRING.format('CO', hash(atoms), 0)

            processed_fields = process_species_list(atoms)

            # Add if doesn't exist, else update (since last-modified changed)
            c_update_doc =  {  # update document
                    '$setOnInsert': {
                        '_id': cid,
                        'atomic_numbers': atoms.get_atomic_numbers().tolist(),
                        'positions': atoms.get_positions().tolist(),
                        'cell': np.array(atoms.get_cell()).tolist(),
                        'pbc': atoms.get_pbc().astype(int).tolist(),
                        'elements': processed_fields['elements'],
                        'nelements': processed_fields['nelements'],
                        'elements_ratios': processed_fields['elements_ratios'],
                        'chemical_formula_reduced': processed_fields['chemical_formula_reduced'],
                        'chemical_formula_anonymous': processed_fields['chemical_formula_anonymous'],
                        'chemical_formula_hill': atoms.get_chemical_formula(),
                        'nsites': len(atoms),
                        'dimension_types': atoms.get_pbc().astype(int).tolist(),
                        'nperiodic_dimensions': int(sum(atoms.get_pbc())),
                        'lattice_vectors': np.array(atoms.get_cell()).tolist(),
                    },
                    '$set': {
                        'last_modified': datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
                    },
                    '$addToSet': {
                        'names': {
                            '$each': list(atoms.info[ATOMS_NAME_FIELD])
                        },
                        'labels': {
                            '$each': list(atoms.info[ATOMS_LABELS_FIELD])
                        },
                        'relationships.properties': {
                            '$each': []
                        }
                    }
                }

            available_keys = set().union(atoms.info.keys(), atoms.arrays.keys())

            pid = None

            new_pids = []
            for pname, pmap in property_map.items():
                pmap_copy = dict(pmap)
                if '_settings' in pmap_copy:
                    del pmap_copy['_settings']

                # Pre-check to avoid having to delete partially-added properties
                missing_keys = expected_keys[pname] - available_keys
                if missing_keys:
                    warnings.warn(
                        "Configuration is missing keys {} for Property"\
                        "Instance construction. Available keys: {}. "\
                        "Skipping".format(
                            missing_keys, available_keys
                        )
                    )
                    continue

                prop = Property.from_definition(
                    definition=property_definitions[pname],
                    configuration=atoms,
                    property_map=pmap_copy
                )

                pid = ID_FORMAT_STRING.format('PI', hash(prop), 0)

                new_pids.append(pid)

                labels = []
                methods = []
                settings_ids = []

                # Attach property settings, if any were given
                if '_settings' in pmap:
                    pso_map = pmap['_settings']

                    all_ps_fields = set(pso_map.keys()) - {
                        '_method', '_description', '_files', '_labels'
                    }

                    ps_not_required = {
                        psk for psk in all_ps_fields if not pso_map[psk]['required']
                    }

                    ps_missing_keys = all_ps_fields  - available_keys - ps_not_required

                    if not ps_missing_keys:
                        # Has all of the required PS keys

                        gathered_fields = {}
                        for ps_field in all_ps_fields:
                            psf_key = pso_map[ps_field]['field']
                            psf_units = pso_map[ps_field]['field']

                            if ps_field in atoms.info:
                                v = atoms.info[psf_key]
                            elif ps_field in atoms.arrays:
                                v = atoms.arrays[psf_units]
                            else:
                                # Then this key is not required
                                continue

                            gathered_fields[ps_field] = {
                                'source-value': v,
                                'source-unit': psf_units
                            }

                        ps = PropertySettings(
                            method=pso_map['_method'] if '_method' in pso_map else None,
                            description=pso_map['_description'] if '_description' in pso_map else None,
                            files=pso_map['_files'] if '_files' in pso_map else None,
                            labels=pso_map['_labels'] if '_labels' in pso_map else None,
                            fields=gathered_fields,
                        )

                        ps_id = ID_FORMAT_STRING.format('PS', hash(ps), 0)

                        ps_update_doc =  {  # update document
                                '$setOnInsert': {
                                    '_id': ps_id,
                                    'method':       ps.method,
                                    '_description': ps.description,
                                    '_files':       ps.files,
                                },
                                '$set': {
                                    'last_modified': datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
                                },
                                '$addToSet': {
                                    'labels': {
                                        '$each': list(ps.labels)
                                    },
                                    'relationships.properties': {
                                        '$each': [pid]
                                    }
                                }
                            }

                        settings_docs.append(UpdateOne(
                            {'_id': ps_id},
                            ps_update_doc,
                            upsert=True,
                        ))

                        methods.append(ps.method)
                        labels += list(ps.labels)
                        settings_ids.append(ps_id)

                # Prepare the property instance EDN document
                setOnInsert = {}
                for k in property_map[pname]:
                    if k not in prop.keys():
                        # To allow for missing non-required keys.
                        # Required keys checked for in Property.from_definition
                        continue

                    if isinstance(prop[k]['source-value'], (int, float, str)):
                        # Add directly
                        setOnInsert[k] = {
                            'source-value': prop[k]['source-value']
                        }
                    else:
                        # Then it's array-like and should be converted to a list
                        setOnInsert[k] = {
                            'source-value': np.atleast_1d(
                                prop[k]['source-value']
                            ).tolist()
                        }

                    if 'source-unit' in prop[k]:
                        setOnInsert[k]['source-unit'] = prop[k]['source-unit']

                    p_update_doc = {
                        '$addToSet': {
                            'methods': {'$each': methods},
                            'labels': {'$each': labels},
                            # PR -> PSO pointer
                            'relationships.property_settings': {
                                '$each': settings_ids
                            },
                            'relationships.configurations': cid,
                        },
                        '$setOnInsert': {
                            '_id': pid,
                            'type': pname,
                            pname: setOnInsert
                        },
                        '$set': {
                            'last_modified': datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
                        }
                    }

                property_docs.append(UpdateOne(
                    {'_id': pid},
                    p_update_doc,
                    upsert=True,
                ))

                c_update_doc['$addToSet']['relationships.properties']['$each'].append(
                    pid
                )

                insertions.append((cid, pid))

            config_docs.append(
                UpdateOne({'_id': cid}, c_update_doc, upsert=True)
            )

            if not pid:
                # Only yield if something wasn't yielded earlier
                insertions.append((cid, pid))

            ai += 1

        if config_docs:
            res = coll_configurations.bulk_write(config_docs, ordered=False)
            nmatch = res.bulk_api_result['nMatched']
            if nmatch:
                warnings.warn(
                    '{} duplicate configurations detected'.format(nmatch)
                )
        if property_docs:
            res = coll_properties.bulk_write(property_docs, ordered=False)
            nmatch = res.bulk_api_result['nMatched']
            if nmatch:
                warnings.warn(
                    '{} duplicate properties detected'.format(nmatch)
                )

        if settings_docs:
            res = coll_property_settings.bulk_write(
                settings_docs,
                # [
                #     UpdateOne(
                #         {'_id': sid},
                #         {'$addToSet': {'relationships.properties': {'$each': lst}}}
                #     ) for sid, lst in settings_docs.items()
                # ],
                ordered=False
            )
            nmatch = res.bulk_api_result['nMatched']
            if nmatch:
                warnings.warn(
                    '{} duplicate property settings detected'.format(nmatch)
                )


        client.close()
        return insertions


    def insert_property_definition(self, definition):
        """
        Inserts a new property definition into the database. Checks that
        definition is valid, then builds all necessary groups in
        :code:`/root/properties`. Throws an error if the property already
        exists.

        Args:

            definition (dict or string):
                The map defining the property. See the example below, or the
                `OpenKIM Properties Framework <https://openkim.org/doc/schema/properties-framework/>`_
                for more details. If a string is provided, it must be the name
                of an existing property definition from the
                `OpenKIM Properties List <https://openkim.org/properties>`_.

        Example definition:

        .. code-block:: python

            property_definition = {
                'property-id': 'default',
                'property-title': 'A default property used for testing',
                'property-description': 'A description of the property',
                'energy': {'type': 'float', 'has-unit': True, 'extent': [], 'required': True, 'description': 'empty'},
                'stress': {'type': 'float', 'has-unit': True, 'extent': [6], 'required': True, 'description': 'empty'},
                'name': {'type': 'string', 'has-unit': False, 'extent': [], 'required': True, 'description': 'empty'},
                'nd-same-shape': {'type': 'float', 'has-unit': True, 'extent': [2,3,5], 'required': True, 'description': 'empty'},
                'nd-diff-shape': {'type': 'float', 'has-unit': True, 'extent': [":", ":", ":"], 'required': True, 'description': 'empty'},
                'forces': {'type': 'float', 'has-unit': True, 'extent': [":", 3], 'required': True, 'description': 'empty'},
                'nd-same-shape-arr': {'type': 'float', 'has-unit': True, 'extent': [':', 2, 3], 'required': True, 'description': 'empty'},
                'nd-diff-shape-arr': {'type': 'float', 'has-unit': True, 'extent': [':', ':', ':'], 'required': True, 'description': 'empty'},
            }

        """

        if isinstance(definition, str):
            definition = KIM_PROPERTIES[definition]

        if self.property_definitions.count_documents(
            {'_id': definition['property-id']}
            ):
            warnings.warn(
                "Property definition with name '{}' already exists. "\
                "Using existing definition.".format(
                    definition['property-id']
                )
            )

        dummy_dict = deepcopy(definition)

        # Spoof if necessary
        if VALID_KIM_ID.match(dummy_dict['property-id']) is None:
            # Invalid ID. Try spoofing it
            dummy_dict['property-id'] = 'tag:@,0000-00-00:property/'
            dummy_dict['property-id'] += definition['property-id']
            warnings.warn(
                "Invalid KIM property-id; "\
                "Temporarily renaming to {}. "\
                "See https://openkim.org/doc/schema/properties-framework/ "\
                "for more details.".format(dummy_dict['property-id'])
            )

        check_property_definition(dummy_dict)

        self.property_definitions.update_one(
            {'_id': definition['property-id']},
            {
                '$setOnInsert': {
                    '_id': definition['property-id'],
                    'definition': dummy_dict
                }
            },
            upsert=True
        )


    def get_property_definition(self, name):
        return self.property_definitions.find_one({'_id': name})


    def insert_property_settings(self, ps_object):
        """
        Inserts a new property settings object into the database by creating
        and populating the necessary groups in :code:`/root/property_settings`.

        Args:

            ps_object (PropertySettings)
                The :class:`~colabfit.tools.property_settings.PropertySettings`
                object to insert into the database.


        Returns:

            ps_id (str):
                The ID of the inserted property settings object. Equals the hash
                of the object.
        """

        ps_id = ID_FORMAT_STRING.format('PS', hash(ps_object), 0)

        self.property_settings.update_one(
            {'_id': ps_id},
            {
                '$addToSet': {
                    'labels': {'$each': list(ps_object.labels)}
                },
                '$setOnInsert': {
                    '_id': ps_id,
                    'method': ps_object.method,
                    'description': ps_object.description,
                    'files': [
                        {
                            'file_name': ftup[0],
                            'file_contents': ftup[1],
                        } for ftup in ps_object.files
                    ],
                }
            },
            upsert=True
        )

        return ps_id


    def get_property_settings(self, pso_id):
        pso_doc = self.property_settings.find_one({'_id': pso_id})

        return PropertySettings(
                method=pso_doc['method'],
                description=pso_doc['description'],
                labels=set(pso_doc['labels']),
                files=[
                    (d['file_name'], d['file_contents'])
                    for d in pso_doc['files']
                ]
            )


    # @staticmethod
    def get_data(
        self, collection_name,
        fields,
        query=None,
        ids=None,
        keep_ids=False,
        concatenate=False,
        vstack=False,
        ravel=False,
        unpack_properties=True,
        verbose=False,
        ):
        """
        Queries the database and returns the fields specified by `keys` as a
        list or an array of values. Returns the results in memory.

        Example:

        .. code-block:: python

            data = database.get_data(
                collection_name='properties',
                query={'_id': {'$in': <list_of_property_IDs>}},
                fields=['property_name_1.energy', 'property_name_1.forces'],
                cache=True
            )

        Args:

            collection_name (str):
                The name of a collection in the database.

            fields (list or str):
                The fields to return from the documents. Sub-fields can be
                returned by providing names separated by periods ('.')

            query (dict, default=None):
                A Mongo query dictionary. If None, returns the data for all of
                the documents in the collection.

            ids (list):
                The list of IDs to return the data for. If None, returns the
                data for the entire collection. Note that this information can
                also be provided using the :code:`query` argument.

            keep_ids (bool, default=False):
                If True, includes the '_id' field as one of the returned values.

            concatenate (bool, default=False):
                If True, concatenates the data before returning.

            vstack (bool, default=False):
                If True, calls np.vstack on data before returning.

            ravel (bool, default=False):
                If True, concatenates and ravels the data before returning.

            unpack_properties (bool, default=True):
                If True, returns only the contents of the :code:`'source-value'`
                key for each field in :attr:`fields` (assuming
                :code:`'source-value'` exists). Users who wish to return the
                full dictionaries for fields should set
                :code:`unpack_properties=False`.

            verbose (bool, default=False):
                If True, prints a progress bar

        Returns:

            data (dict):
                key = k for k in keys. val = in-memory data
        """

        if query is None:
            query = {}

        if ids is not None:
            if isinstance(ids, str):
                ids = [ids]
            elif isinstance(ids, np.ndarray):
                ids = ids.tolist()

            query = {'_id': {'$in': ids}}

        if isinstance(fields, str):
            fields = [fields]

        retfields = {k: 1 for k in fields}

        if keep_ids:
            retfields['_id'] = 1

        collection = self[self.database_name][collection_name]

        cursor = collection.find(query, retfields)

        data = {
            k: [] for k in retfields
        }

        for doc in tqdm(cursor, desc='Getting data', disable=not verbose):
            for k in retfields:
                # For figuring out if document has the data

                # Handle keys like "property-name.property-field"
                missing = False
                v = doc
                for kk in k.split('.'):
                    if kk in v:
                        v = v[kk]
                    else:
                        # Missing something
                        missing = True

                if not missing:
                    if isinstance(v, dict):
                        if unpack_properties and ('source-value' in v):
                            v = v['source-value']

                    data[k].append(v)

        for k,v in data.items():
            # data[k] = np.array(data[k])

            if concatenate or ravel:
                try:
                    data[k] = np.concatenate(v)
                except:
                    data[k] = np.array(v)

            if vstack:
                data[k] = np.vstack(v)

        if ravel:
            for k,v in data.items():
                data[k] = v.ravel()

        if len(retfields) == 1:
            return data[list(retfields.keys())[0]]
        else:
            return data


    def get_configuration(self, i, property_ids=None, attach_properties=False):
        """
        Returns a single configuration by calling :meth:`get_configurations`
        """
        return self.get_configurations(
            [i], property_ids=property_ids, attach_properties=attach_properties
        )[0]


    def get_configurations(
        self, configuration_ids,
        property_ids=None,
        attach_properties=False,
        generator=False,
        verbose=False
        ):
        """
        A generator that returns in-memory Configuration objects one at a time
        by loading the atomic numbers, positions, cells, and PBCs.

        Args:

            configuration_ids (list or 'all'):
                A list of string IDs specifying which Configurations to return.
                If 'all', returns all of the configurations in the database.

            property_ids (list, default=None):
                A list of Property IDs. Used for limiting searches when
                :code:`attach_properties==True`.  If None,
                :code:`attach_properties` will attach all linked Properties.
                Note that this only attaches one property per Configuration, so
                if multiple properties point to the same Configuration, that
                Configuration will be returned multiple times.

            attach_properties (bool, default=False):
                If True, attaches all the data of any linked properties from
                :code:`property_ids`. The property data will either be added to
                the :code:`arrays` dictionary on a Configuration (if it can be
                converted to a matrix where the first dimension is the same
                as the number of atoms in the Configuration) or the :code:`info`
                dictionary (if it wasn't added to :code:`arrays`). Property
                fields in a list to accomodate the possibility of multiple
                properties of the same type pointing to the same configuration.
                WARNING: don't use this option if multiple properties of the
                same type point to the same Configuration, but the properties
                don't have values for all of their fields.

            generator (bool, default=False):
                If True, this function returns a generator of the
                configurations. This is useful if the configurations can't all
                fit in memory at the same time.

            verbose (bool):
                If True, prints progress bar

        Returns:

            configurations (iterable):
                A list or generator of the re-constructed configurations
        """

        if configuration_ids == 'all':
            query = {}
        else:
            if isinstance(configuration_ids, str):
                configuration_ids = [configuration_ids]

            query = {'_id': {'$in': configuration_ids}}

        if generator:
            return self._get_configurations(
                query=query,
                property_ids=property_ids,
                attach_properties=attach_properties,
                verbose=verbose
            )
        else:
            return list(self._get_configurations(
                query=query,
                property_ids=property_ids,
                attach_properties=attach_properties,
                verbose=verbose
            ))


    def _get_configurations(self, query, property_ids, attach_properties, verbose=False):
        if not attach_properties:
            for co_doc in tqdm(
                self.configurations.find(
                    query,
                    {
                        'atomic_numbers', 'positions', 'cell', 'pbc', 'names',
                        'labels'
                    }
                ),
                desc='Getting configurations',
                disable=not verbose
                ):
                c = Configuration(
                    symbols=co_doc['atomic_numbers'],
                    positions=co_doc['positions'],
                    cell=co_doc['cell'],
                    pbc=co_doc['pbc'],
                )

                c.info['_id'] = co_doc['_id']
                c.info[ATOMS_NAME_FIELD] = co_doc['names']
                c.info[ATOMS_LABELS_FIELD] = co_doc['labels']

                yield c
        else:

            # property_match = { 'relationships.configurations': query['_id']}

            # if property_ids is not None:
            #     property_match['_id'] = {'$in': property_ids}

            for co_doc in tqdm(self.configurations.aggregate([
                    {'$match': query},
                    {'$lookup': {
                        'from': 'properties',
                        'localField': 'relationships.properties',
                        'foreignField': '_id',
                        'as': 'linked_properties'
                    }},
                    # {'$match': {'linked_properties._id': property_match}},
                    {'$match': {'linked_properties._id': {'$in': property_ids}}},
                    ]),
                    desc='Getting configurations',
                    disable=not verbose
                ):

                c = Configuration(
                    symbols=co_doc['atomic_numbers'],
                    positions=co_doc['positions'],
                    cell=co_doc['cell'],
                    pbc=co_doc['pbc'],
                )

                c.info['_id'] = co_doc['_id']
                c.info[ATOMS_NAME_FIELD] = co_doc['names']
                c.info[ATOMS_LABELS_FIELD] = co_doc['labels']

                n = len(c)

                for pr_doc in co_doc['linked_properties']:
                    for field_name, field in pr_doc[pr_doc['type']].items():
                        v = np.atleast_1d(field['source-value'])

                        if (v.dtype == 'O') or v.shape[0] != n:
                            dct = c.info
                        else:
                            dct = c.arrays

                        field_name = f'{pr_doc["type"]}.{field_name}'

                        if field_name in dct:
                            # Then this is a duplicate property
                            dct[field_name].append(v)
                        else:
                            # Then this is the first time
                            # the property of this type is being added
                            dct[field_name] = [v]

                yield c


    def concatenate_configurations(self):
        """
        Concatenates the atomic_numbers, positions, cells, and pbcs groups in
        /configurations.
        """
        self.database.concatenate_configurations()


    def insert_configuration_set(self, ids, description='', verbose=False):
        """
        Inserts the configuration set of IDs to the database.

        Args:

            ids (list or str):
                The IDs of the configurations to include in the configuartion
                set.

            description (str, optional):
                A human-readable description of the configuration set.

            verbose (bool, default=False):
                If True, prints a progress bar
        """

        if isinstance(ids, str):
            ids = [ids]

        cs_hash = sha512()
        cs_hash.update(description.encode('utf-8'))
        for i in sorted(ids):
            cs_hash.update(str(i).encode('utf-8'))

        cs_hash = int(cs_hash.hexdigest()[:HASH_LENGTH], 16)-HASH_SHIFT
        cs_id = ID_FORMAT_STRING.format('CS', cs_hash, 0)

        # Check for duplicates
        if self.configuration_sets.count_documents({'_id': cs_id}):
            return cs_id

        # Make sure all of the configurations exist
        if self.configurations.count_documents({'_id': {'$in': ids}}) != len(ids):
            raise MissingEntryError(
                "Not all of the IDs provided to insert_configuration_set exist"\
                " in the database."
            )

        aggregated_info = self.aggregate_configuration_info(ids, verbose=verbose)

        self.configuration_sets.update_one(
            {'_id': cs_id},
            {
                '$addToSet': {
                    'relationships.configurations': {'$each': ids}
                },
                '$setOnInsert': {
                    '_id': cs_id,
                    'description': description,
                },
                '$set': {
                    'aggregated_info': aggregated_info,
                    'last_modified': datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
                },
            },
            upsert=True
        )

        # Add the backwards relationships CO->CS
        config_docs = []
        for cid in ids:
            config_docs.append(UpdateOne(
                {'_id': cid},
                {
                    '$addToSet': {
                        'relationships.configuration_sets': cs_id
                    }
                }
            ))

        self.configurations.bulk_write(config_docs)

        return cs_id


    def get_configuration_set(self, cs_id, resync=False):
        """
        Returns the configuration set with the given ID.

        Args:

            cs_ids (str):
                The ID of the configuration set to return

            resync (bool):
                If True, re-aggregates the configuration set information before
                returning. Default is False.

        Returns:

            A dictionary with two keys:
                'last_modified': a datetime string
                'configuration_set': the configuration set object
        """


        if resync:
            self.resync_configuration_set(cs_id)

        cs_doc = self.configuration_sets.find_one({'_id': cs_id})

        return {
            'last_modified': cs_doc['last_modified'],
            'configuration_set': ConfigurationSet(
                configuration_ids=cs_doc['relationships']['configurations'],
                description=cs_doc['description'],
                aggregated_info=cs_doc['aggregated_info']
            )
        }


    def resync_configuration_set(self, cs_id, verbose=False):
        """
        Re-synchronizes the configuration set by re-aggregating the information
        from the configurations.

        Args:

            cs_id (str):
                The ID of the configuration set to update

            verbose (bool, default=False):
                If True, prints a progress bar

        Returns:

            None; updates the configuration set document in-place

        """

        cs_doc = self.configuration_sets.find_one({'_id': cs_id})

        aggregated_info = self.aggregate_configuration_info(
            cs_doc['relationships']['configurations'], verbose=verbose
        )

        self.configuration_sets.update_one(
            {'_id': cs_id},
            {'$set': {'aggregated_info': aggregated_info}},
            upsert=True,
        )


    def resync_dataset(self, ds_id, verbose=False):
        """
        Re-synchronizes the dataset by aggregating all necessary data from
        properties and configuration sets. Note that this also calls
        :meth:`colabfit.tools.client.resync_configuration_set`

        Args:

            ds_id (str):
                The ID of the dataset to update

            verbose (bool, default=False):
                If True, prints a progress bar

        Returns:

            None; updates the dataset document in-place
        """

        ds_doc = self.datasets.find_one({'_id': ds_id})

        cs_ids = ds_doc['relationships']['configuration_sets']
        pr_ids = ds_doc['relationships']['properties']

        for csid in cs_ids:
            self.resync_configuration_set(csid, verbose=verbose)

        aggregated_info = {}
        for k,v in self.aggregate_configuration_set_info(cs_ids).items():
            if k == 'labels':
                k = 'configuration_labels'
            elif k == 'labels_counts':
                k = 'configuration_labels_counts'

            aggregated_info[k] = v

        for k,v in self.aggregate_property_info(pr_ids, verbose=verbose).items():
            if k in {
                'labels', 'labels_counts',
                'types',  'types_counts',
                'fields', 'fields_counts'
                }:
                k = 'property_' + k

            aggregated_info[k] = v

        self.datasets.update_one(
            {'_id': ds_id},
            {'$set': {'aggregated_info': aggregated_info}},
            upsert=True
        )


    def aggregate_configuration_info(self, ids, verbose=False):
        """
        Gathers the following information from a collection of configurations:

        * :code:`nconfigurations`: the total number of configurations
        * :code:`nsites`: the total number of sites
        * :code:`nelements`: the total number of unique element types
        * :code:`elements`: the element types
        * :code:`individual_elements_ratios`: a set of elements ratios generated
          by looping over each configuration, extracting its concentration of
          each element, and adding the tuple of concentrations to the set
        * :code:`total_elements_ratios`: the ratio of the total count of atoms
            of each element type over :code:`nsites`
        * :code:`labels`: the union of all configuration labels
        * :code:`labels_counts`: the total count of each label
        * :code:`chemical_formula_reduced`: the set of all reduced chemical
            formulae
        * :code:`chemical_formula_anonymous`: the set of all anonymous chemical
            formulae
        * :code:`chemical_formula_hill`: the set of all hill chemical formulae
        * :code:`nperiodic_dimensions`: the set of all numbers of periodic
            dimensions
        * :code:`dimension_types`: the set of all periodic boundary choices

        Returns:

            aggregated_info (dict):
                All of the aggregated info

            verbose (bool, default=False):
                If True, prints a progress bar
        """

        aggregated_info = {
            'nconfigurations': len(ids),
            'nsites': 0,
            'nelements': 0,
            'chemical_systems': set(),
            'elements': [],
            'individual_elements_ratios': {},
            'total_elements_ratios': {},
            'labels': [],
            'labels_counts': [],
            'chemical_formula_reduced': set(),
            'chemical_formula_anonymous': set(),
            'chemical_formula_hill': set(),
            'nperiodic_dimensions': set(),
            'dimension_types': set(),
        }

        for doc in tqdm(
            self.configurations.find({'_id': {'$in': ids}}),
            desc='Aggregating configuration info',
            disable=not verbose,
            total=len(ids),
            ):
            aggregated_info['nsites'] += doc['nsites']

            aggregated_info['chemical_systems'].add(''.join(doc['elements']))

            for e, er in zip(doc['elements'], doc['elements_ratios']):
                if e not in aggregated_info['elements']:
                    aggregated_info['nelements'] += 1
                    aggregated_info['elements'].append(e)
                    aggregated_info['total_elements_ratios'][e] = er*doc['nsites']
                    aggregated_info['individual_elements_ratios'][e] = set(
                        [np.round_(er, decimals=2)]
                    )
                else:
                    aggregated_info['total_elements_ratios'][e] += er*doc['nsites']
                    aggregated_info['individual_elements_ratios'][e].add(
                        np.round_(er, decimals=2)
                    )

            for l in doc['labels']:
                if l not in aggregated_info['labels']:
                    aggregated_info['labels'].append(l)
                    aggregated_info['labels_counts'].append(1)
                else:
                    idx = aggregated_info['labels'].index(l)
                    aggregated_info['labels_counts'][idx] += 1

            aggregated_info['chemical_formula_reduced'].add(doc['chemical_formula_reduced'])
            aggregated_info['chemical_formula_anonymous'].add(doc['chemical_formula_anonymous'])
            aggregated_info['chemical_formula_hill'].add(doc['chemical_formula_hill'])

            aggregated_info['nperiodic_dimensions'].add(doc['nperiodic_dimensions'])
            aggregated_info['dimension_types'].add(tuple(doc['dimension_types']))

        for e in aggregated_info['elements']:
            aggregated_info['total_elements_ratios'][e] /= aggregated_info['nsites']
            aggregated_info['individual_elements_ratios'][e] = list(aggregated_info['individual_elements_ratios'][e])

        aggregated_info['chemical_systems'] = list(aggregated_info['chemical_systems'])

        aggregated_info['chemical_formula_reduced'] = list(aggregated_info['chemical_formula_reduced'])
        aggregated_info['chemical_formula_anonymous'] = list(aggregated_info['chemical_formula_anonymous'])
        aggregated_info['chemical_formula_hill'] = list(aggregated_info['chemical_formula_hill'])
        aggregated_info['nperiodic_dimensions'] = list(aggregated_info['nperiodic_dimensions'])
        aggregated_info['dimension_types'] = list(aggregated_info['dimension_types'])

        return aggregated_info


    def aggregate_property_info(self, pr_ids, verbose=False):
        """
        Aggregates the following information from a list of properties:

            * types
            * labels
            * labels_counts

        Args:

            pr_ids (list or str):
                The IDs of the configurations to aggregate information from

            verbose (bool, default=False):
                If True, prints a progress bar

        Returns:

            aggregated_info (dict):
                All of the aggregated info
        """

        if isinstance(pr_ids, str):
            pr_ids = [pr_ids]

        aggregated_info = {
            'types': [],
            'types_counts': [],
            'fields': [],
            'fields_counts': [],
            'methods': [],
            'methods_counts': [],
            'labels': [],
            'labels_counts': []
        }

        ignore_keys = {
            'property-id', 'property-title', 'property-description', '_id',
        }

        for doc in tqdm(
            self.properties.find({'_id': {'$in': pr_ids}}),
            desc='Aggregating property info',
            disable=not verbose,
            total=len(pr_ids)
            ):
            if doc['type'] not in aggregated_info['types']:
                aggregated_info['types'].append(doc['type'])
                aggregated_info['types_counts'].append(1)
            else:
                idx = aggregated_info['types'].index(doc['type'])
                aggregated_info['types_counts'][idx] += 1

            for l in doc[doc['type']]:
                if l in ignore_keys: continue

                l = '.'.join([doc['type'], l])

                if l not in aggregated_info['fields']:
                    aggregated_info['fields'].append(l)
                    aggregated_info['fields_counts'].append(1)
                else:
                    idx = aggregated_info['fields'].index(l)
                    aggregated_info['fields_counts'][idx] += 1

            for l in doc['labels']:
                if l not in aggregated_info['labels']:
                    aggregated_info['labels'].append(l)
                    aggregated_info['labels_counts'].append(1)
                else:
                    idx = aggregated_info['labels'].index(l)
                    aggregated_info['labels_counts'][idx] += 1


            for l in doc['methods']:
                if l not in aggregated_info['methods']:
                    aggregated_info['methods'].append(l)
                    aggregated_info['methods_counts'].append(1)
                else:
                    idx = aggregated_info['methods'].index(l)
                    aggregated_info['methods_counts'][idx] += 1

        return aggregated_info


    def aggregate_configuration_set_info(self, cs_ids, resync=False, verbose=False):
        """
        Aggregates the following information from a list of configuration sets:

            * nconfigurations
            * nsites
            * chemical_systems
            * nelements
            * elements
            * individual_elements_ratios
            * total_elements_ratios
            * labels
            * labels_counts
            * chemical_formula_reduced
            * chemical_formula_anonymous
            * chemical_formula_hill
            * nperiodic_dimensions
            * dimension_types

        Args:

            cs_ids (list or str):
                The IDs of the configurations to aggregate information from

            resync (bool, default=False):
                If True, re-synchronizes each configuration set before
                aggregating the information.

            verbose (bool, default=False):
                If True, prints a progress bar

        Returns:

            aggregated_info (dict):
                All of the aggregated info
        """

        if isinstance(cs_ids, str):
            cs_ids = [cs_ids]

        if resync:
            for csid in cs_ids:
                self.resync_configuration_set(csid, verbose=verbose)

        co_ids = list(set(itertools.chain.from_iterable(
            cs_doc['relationships']['configurations'] for cs_doc in
            self.configuration_sets.find({'_id': {'$in': cs_ids}})
        )))

        return self.aggregate_configuration_info(co_ids, verbose=verbose)


    def insert_dataset(
        self, cs_ids, pr_ids, name,
        authors=None,
        links=None,
        description='',
        resync=False,
        verbose=False,
        ):
        """
        Inserts a dataset into the database.

        Args:

            cs_ids (list or str):
                The IDs of the configuration sets to link to the dataset.

            pr_ids (list or str):
                The IDs of the properties to link to the dataset

            name (str):
                The name of the dataset

            authors (list or str or None):
                The names of the authors of the dataset. If None, then no
                authors are added.

            links (list or str or None):
                External links (e.g., journal articles, Git repositories, ...)
                to be associated with the dataset. If None, then no links are
                added.

            description (str or None):
                A human-readable description of the dataset. If None, then not
                description is added.

            resync (bool):
                If True, re-synchronizes the configuration sets and properties
                before adding to the dataset. Default is False.

            verbose (bool, default=False):
                If True, prints a progress bar

        Returns:

            ds_id (str):
                The ID of the inserted dataset
        """

        if isinstance(cs_ids, str):
            cs_ids = [cs_ids]

        if isinstance(pr_ids, str):
            pr_ids = [pr_ids]

        if isinstance(authors, str):
            authors = [authors]

        if isinstance(links, str):
            links = [links]

        ds_hash = sha512()
        for ci in sorted(cs_ids):
            ds_hash.update(str(ci).encode('utf-8'))
        for pi in sorted(pr_ids):
            ds_hash.update(str(pi).encode('utf-8'))

        ds_hash = int(ds_hash.hexdigest()[:HASH_LENGTH], 16)-HASH_SHIFT
        ds_id = ID_FORMAT_STRING.format('DS', ds_hash, 0)

        # Check for duplicates
        if self.datasets.count_documents({'_id': ds_id}):
            if resync:
                self.resync_dataset(ds_id)

            return ds_id

        aggregated_info = {}
        for k,v in self.aggregate_configuration_set_info(
            cs_ids, verbose=verbose).items():
            if k == 'labels':
                k = 'configuration_labels'
            elif k == 'labels_counts':
                k = 'configuration_labels_counts'

            aggregated_info[k] = v

        for k,v in self.aggregate_property_info(
            pr_ids, verbose=verbose).items():
            if k in {
                'labels', 'labels_counts',
                'types',  'types_counts',
                'fields', 'fields_counts'
                }:
                k = 'property_' + k

            aggregated_info[k] = v

        self.datasets.update_one(
            {'_id': ds_id},
            {
                '$addToSet': {
                    'relationships.configuration_sets': {'$each': cs_ids},
                    'relationships.properties': {'$each': pr_ids},
                },
                '$setOnInsert': {
                    '_id': ds_id,
                    'name': name,
                    'authors': authors,
                    'links': links,
                    'description': description,
                },
                '$set': {
                    'aggregated_info': aggregated_info,
                    'last_modified': datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
                },
            },
            upsert=True
        )

        # Add the backwards relationships CS->DS
        config_set_docs = []
        for csid in cs_ids:
            config_set_docs.append(UpdateOne(
                {'_id': csid},
                {'$addToSet': {'relationships.datasets': ds_id}}
            ))

        self.configuration_sets.bulk_write(config_set_docs)

        # Add the backwards relationships PR->DS
        property_docs = []
        for pid in tqdm(pr_ids, desc='Updating PR->DS relationships'):
            property_docs.append(UpdateOne(
                {'_id': pid},
                {'$addToSet': {'relationships.datasets': ds_id}}
            ))

        self.properties.bulk_write(property_docs)

        return ds_id


    def get_dataset(self, ds_id, resync=False, verbose=False):
        """
        Returns the dataset with the given ID.

        Args:

            ds_ids (str):
                The ID of the dataset to return

            resync (bool):
                If True, re-aggregates the configuration set and property
                information before returning. Default is False.

            verbose (bool, default=True):
                If True, prints a progress bar. Only used if
                :code:`resync=False`.

        Returns:

            A dictionary with two keys:
                'last_modified': a datetime string
                'dataset': the dataset object
        """


        if resync:
            self.resync_dataset(ds_id, verbose=verbose)

        ds_doc = self.datasets.find_one({'_id': ds_id})

        return {
            '_id': ds_id,
            'last_modified': ds_doc['last_modified'],
            'dataset': Dataset(
                configuration_set_ids=ds_doc['relationships']['configuration_sets'],
                property_ids=ds_doc['relationships']['properties'],
                name=ds_doc['name'],
                authors=ds_doc['authors'],
                links=ds_doc['links'],
                description=ds_doc['description'],
                aggregated_info=ds_doc['aggregated_info']
            )
        }



    def aggregate_dataset_info(self, ds_ids):
        """
        Aggregates information from a list of datasets.

        NOTE: this will face all of the same challenges as
        aggregate_configuration_set_info()

            * you need to find the overlap of COs and PRs.
        """
        pass


    def apply_labels(
        self, dataset_id, collection_name, query, labels, verbose=False
        ):
        """
        Applies the given labels to all objects in the specified collection that
        match the query and are linked to the given dataset.

        Args:

            dataset_id (str):
                The ID of the dataset. Used as a safety measure to only update
                entries for the given dataset.

            collection_name (str):
                One of 'configurations' or 'properties'.

            query (dict):
                A Mongo-style query for filtering the collection. For
                example: :code:`query = {'nsites': {'$lt': 100}}`.

            labels (set or str):
                A set of labels to apply to the matching entries.

            verbose (bool):
                If True, prints progress bar.

        Pseudocode:
            * Get the IDs of the configurations that match the query
            * Use updateMany to update the MongoDB
            * Iterate over the HDF5 entries.
        """

        dataset = self.get_dataset(dataset_id)['dataset']

        if collection_name == 'configurations':
            collection = self.configurations

            cs_ids = dataset.configuration_set_ids

            all_co_ids = list(set(itertools.chain.from_iterable(
                cs_doc['relationships']['configurations'] for cs_doc in
                self.configuration_sets.find({'_id': {'$in': cs_ids}})
            )))

            query['_id'] = {'$in': all_co_ids}
        elif collection_name == 'properties':
            collection = self.properties

            query['_id'] = {'$in': dataset.property_ids}
        else:
            raise RuntimeError(
                "collection_name must be 'configurations' or 'properties'"
            )

        if isinstance(labels, str):
            labels = {labels}

        for doc in tqdm(
            collection.find(query, {'_id': 1}),
            desc='Applying configuration labels',
            disable=not verbose
            ):
            doc_id = doc['_id']

            collection.update_one(
                {'_id': doc_id},
                {'$addToSet': {'labels': {'$each': list(labels)}}}
            )


    def plot_histograms(
        self,
        fields=None,
        query=None,
        ids=None,
        verbose=False,
        nbins=100,
        xscale='linear',
        yscale='linear',
        method='matplotlib'
        ):
        """
        Generates histograms of the given fields.

        Args:

            fields (list or str):
                The names of the fields to plot

            query (dict, default=None):
                A Mongo query dictionary. If None, returns the data for all of
                the documents in the collection.

            ids (list or str):
                The IDs of the objects to plot the data for

            verbose (bool, default=False):
                If True, prints progress bar

            nbins (int):
                Number of bins per histogram

            xscale (str):
                Scaling for x-axes. One of ['linear', 'log'].

            yscale (str):
                Scaling for y-axes. One of ['linear', 'log'].

            method (str, default='plotly')
                Package to use for plotting. 'plotly' or 'matplotlib'.
        """
        if fields is None:
            fields = self.property_fields
        elif isinstance(fields, str):
            fields = [fields]

        nfields = len(fields)

        nrows = max(1, int(np.ceil(nfields/3)))
        if (nrows > 1) or (nfields%3 == 0):
            ncols = 3
        else:
            ncols = nfields%3

        if method == 'plotly':
            fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=fields)
        elif method == 'matplotlib':
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4*ncols, 2*nrows))
            axes = np.atleast_2d(axes)
        else:
            raise RuntimeError('Unsupported plotting method')

        for i, prop in enumerate(fields):
            data = self.get_data(
                'properties', prop,
                query=query,
                ids=ids,
                verbose=verbose,
                ravel=True
            )

            c = i % 3
            r = i // 3


            if nrows > 1:

                if method == 'plotly':
                    fig.add_trace(
                        go.Histogram(x=data, nbinsx=nbins, name=prop),
                        row=r+1, col=c+1,
                    )
                else:
                    _ = axes[r][c].hist(data, bins=nbins)
                    axes[r][c].set_title(prop)
                    axes[r][c].set_xscale(xscale)
                    axes[r][c].set_yscale(yscale)
            else:
                if method == 'plotly':
                    fig.add_trace(
                        go.Histogram(x=data, nbinsx=nbins, name=prop),
                        row=1, col=c+1,
                    )
                else:
                    _ = axes[0][c].hist(data, bins=nbins)
                    axes[r][c].set_title(prop)
                    axes[r][c].set_xscale(xscale)
                    axes[r][c].set_yscale(yscale)

        c += 1
        while c < ncols:
            if method == 'matplotlib':
                axes[r][c].axis('off')
            c += 1

        if method == 'plotly':
            fig.update_layout(
                showlegend=True,
            )
            fig.update_xaxes(type=xscale)
            fig.update_yaxes(type=yscale)
            fig.for_each_annotation(lambda a: a.update(text=""))
        else:
            plt.tight_layout()

        return fig


    def get_statistics(
        self,
        fields,
        query=None,
        ids=None,
        verbose=False,
        ):
        """
        Queries the database and returns the fields specified by `keys` as a
        list or an array of values. Returns the results in memory.

        Example:

        .. code-block:: python

            data = database.get_data(
                collection_name='properties',
                query={'_id': {'$in': <list_of_property_IDs>}},
                fields=['property_name_1.energy', 'property_name_1.forces'],
                cache=True
            )

        Args:

            collection_name (str):
                The name of a collection in the database.

            fields (list or str):
                The fields to return from the documents. Sub-fields can be
                returned by providing names separated by periods ('.')

            query (dict, default=None):
                A Mongo query dictionary. If None, returns the data for all of
                the documents in the collection.

            ids (list):
                The list of IDs to return the data for. If None, returns the
                data for the entire collection. Note that this information can
                also be provided using the :code:`query` argument.

            verbose (bool, default=False):
                If True, prints a progress bar during data extraction

        Returns:
            results (dict)::
                .. code-block:: python

                    {
                        f:  {
                            'average': np.average(data),
                            'std': np.std(data),
                            'min': np.min(data),
                            'max': np.max(data),
                            'average_abs': np.average(np.abs(data))
                        } for f in fields
                    }

        """

        if isinstance(fields, str):
            fields = [fields]

        retdict = {}

        for field in fields:

            data = self.get_data(
                'properties', field, query=query, ids=ids,
                ravel=True, verbose=verbose
            )

            retdict[field] = {
                'average': np.average(data),
                'std': np.std(data),
                'min': np.min(data),
                'max': np.max(data),
                'average_abs': np.average(np.abs(data)),
            }

        if len(fields) == 1:
            return retdict[fields[0]]
        else:
            return retdict


    def filter_on_configurations(self, ds_id, query, verbose=False):
        """
        Searches the configuration sets of a given dataset, and
        returns configuration sets and properties that have been filtered based
        on the given criterion.

            * The returned configuration sets will only include configurations that return True for the filter
            * The returned property IDs will only include properties that point to a configuration that returned True for the filter.

        Args:

            ds_id (str):
                The ID of the dataset to filter

            query (dict):
                A Mongo query that will return the desired objects. Note that
                the key-value pair :code:`{'_id': {'$in': ...}}` will be
                included automatically to filter on only the objects that are
                already linked to the given dataset.

            verbose (bool, default=False):
                If True, prints progress bars

        Returns:

            configuration_sets (list):
                A list of configuration sets that have been pruned to only
                include configurations that satisfy the filter

            property_ids (list):
                A list of property IDs that satisfy the filter
        """

        ds_doc = self.datasets.find_one({'_id': ds_id})

        configuration_sets = []
        property_ids = []

        # Loop over configuration sets
        cursor = self.configuration_sets.find({
            '_id': {'$in': ds_doc['relationships']['configuration_sets']}
            }
        )

        for cs_doc in tqdm(
            cursor,
            desc='Filtering on configuration sets',
            disable=not verbose
            ):

            query['_id'] = {'$in': cs_doc['relationships']['configurations']}

            co_ids = self.get_data('configurations', fields='_id', query=query)

            # Build the filtered configuration sets
            configuration_sets.append(
                ConfigurationSet(
                    configuration_ids=co_ids,
                    description=cs_doc['description'],
                    aggregated_info=self.aggregate_configuration_info(
                        co_ids,
                        verbose=verbose
                    )
                )
            )

        # Now get the corresponding properties
        property_ids = [_['_id'] for _ in self.properties.filter(
            {
            '_id': {'$in': list(itertools.chain.from_iterable(
                cs.configuration_ids for cs in configuration_sets
            ))}
            },
            {'_id': 1}
        )]

        return configuration_sets, property_ids


    def filter_on_properties(
        self, ds_id, filter_fxn=None, query=None, fields=None, verbose=False
        ):
        """
        Searches the properties of a given dataset, and returns configuration
        sets and properties that have been filtered based on the given
        criterion.

            * The returned configuration sets will only include configurations that are pointed to by a property that returned True for the filter
            * The returned property IDs will only include properties that returned True for the filter function.

        Example:

        .. code-block:: python

            configuration_sets, property_ids = database.filter_on_properties(
                ds_id=...,
                filter_fxn=lambda x: np.max(np.abs(x[']))
            )

        Args:

            ds_id (str):
                The ID of the dataset to filter

            filter_fxn (callable, default=None):
                A callable function to use as :code:`filter(filter_fxn, cursor)`
                where :code:`cursor` is a Mongo cursor over all of the
                property documents in the given dataset. If
                :code:`filter_fxn` is None, must specify :code:`query`.

            query (dict, default=None):
                A Mongo query that will return the desired objects. Note that
                the key-value pair :code:`{'_id': {'$in': ...}}` will be
                included automatically to filter on only the objects that are
                already linked to the given dataset.

            fields (str or list, default=None):
                The fields required by :code:`filter_fxn`. Providing the minimum
                number of necessary fields can improve query performance.

            verbose (bool, default=False):
                If True, prints progress bars

        Returns:

            configuration_sets (list):
                A list of configuration sets that have been pruned to only
                include configurations that satisfy the filter

            property_ids (list):
                A list of property IDs that satisfy the filter
        """

        if filter_fxn is None:
            if query is None:
                raise RuntimeError(
                    'filter_fxn and query cannot both be None'
                )
            else:
                filter_fxn = lambda x: True

        ds_doc = self.datasets.find_one({'_id': ds_id})

        configuration_sets = []
        property_ids = []

        # Filter the properties
        retfields = {'_id': 1, 'relationships.configurations': 1}
        if fields is not None:
            if isinstance(fields, str):
                fields = [fields]

            for f in fields:
                retfields[f] = 1

        if query is None:
            query = {}

        query['_id'] = {'$in': ds_doc['relationships']['properties']}

        cursor = self.properties.find(query, retfields)

        all_co_ids = []
        for pr_doc in tqdm(
            cursor,
            desc='Filtering on properties',
            disable=not verbose,
            ):
            if filter_fxn(pr_doc):
                property_ids.append(pr_doc['_id'])
                all_co_ids.append(pr_doc['relationships']['configurations'])

        all_co_ids = list(set(itertools.chain.from_iterable(all_co_ids)))

        # Then filter the configuration sets
        for cs_doc in self.configuration_sets.find({
            '_id': {'$in': ds_doc['relationships']['configuration_sets']}
            }):

            co_ids =list(
                set(cs_doc['relationships']['configurations']).intersection(
                    all_co_ids
                )
            )

            configuration_sets.append(
                ConfigurationSet(
                    configuration_ids=co_ids,
                    description=cs_doc['description'],
                    aggregated_info=self.aggregate_configuration_info(
                        co_ids,
                        verbose=verbose
                    )
                )
            )

        return configuration_sets, property_ids


    # def apply_transformation(
    #     self,
    #     dataset_id,
    #     property_ids,
    #     configuration_ids,
    #     update_map,
    #     ):
    #     """
    #     This function works by looping over the properties that match the
    #     provided query and updating them by applying the updated rule. Fields
    #     from the linked configurations can also be used for the update rule if
    #     by setting configuration_fields.

    #     Example:

    #     .. code-block:: python

    #         # Convert energies to per-atom values
    #         database.apply_transformation(
    #             dataset_id=<dataset_id>,

    #             query={'_id': <property_ids>},
    #             update={
    #                 'property-name.energy':
    #                 lambda fv, doc: fv/doc['configuration']['nsites']
    #             }
    #         )

    #     Args:

    #         dataset_id (str):
    #             The ID of the dataset. Used as a safety measure to only update
    #             entries for the given dataset.

    #         property_ids (list or str):
    #             The IDs of the properties to be updated.

    #         update_map (dict):
    #             A dictionary where the keys are the name of a field to update
    #             (omitting "source-value"), and the values are callable functions
    #             that take the field value and property document as arguments.

    #         configuration_ids (list, default=None):
    #             The IDs of the configurations to use for each property update.
    #             Must be the same length as :code:`pr_ids`. Note that the fields
    #             of configuration :code:`co_ids[i]` will be accessible for the
    #             update command of property :code:`pr_ids[i]` using the
    #             :code:`configuration.<field>` notation. If None, applies the
    #             updates to the properties without using any configuration
    #             fields.

    #     Returns:

    #         update_results (dict):
    #             The results of a Mongo bulkWrite operation
    #     """

    #     dataset = self.get_dataset(dataset_id)['dataset']

    #     pipeline = [
    #         # Filter on the dataset properties
    #         {'$match': {'_id': {'$in': dataset.property_ids}}},
    #         # Filter on the specified properties
    #         {'$match': {'_id': {'$in': property_ids}}},
    #     ]

    #     if configuration_ids:
    #         # Attach the linked configuration documents
    #         pipeline.append(
    #             {'$lookup': {
    #                 'from': 'configurations',
    #                 'localField': 'relationships.configurations',
    #                 'foreignField': '_id',
    #                 'as': 'configuration'
    #             }}
    #         )
    #         pipeline.append(
    #             {'$unwind': '$configuration'}
    #         )
    #     else:
    #         configuration_ids = [None]*len(property_ids)

    #     pr_to_co_map = {p:c for p,c in zip(property_ids, configuration_ids)}

    #     updates = []
    #     for pr_doc in self.properties.aggregate(pipeline):
    #         link_cid = pr_to_co_map[pr_doc['_id']]

    #         # Only perform the operation for the desired configurations
    #         if (link_cid is None) or (pr_doc['configuration']['_id'] == link_cid):
    #             for key, fxn in update_map.items():
    #                 fv = pr_doc
    #                 missing = False
    #                 for k in key.split('.'):
    #                     if k in fv:
    #                         fv = fv[k]
    #                     else:
    #                         # Does not have necessary data
    #                         missing = True

    #                 if missing: continue

    #                 if isinstance(fv, dict):
    #                     # Handle case where "source-value" isn't at end of key
    #                     fv = fv['source-value']
    #                     key += '.source-value'

    #                 data = fxn(fv, pr_doc)

    #                 if isinstance(data, (np.ndarray, list)):
    #                     data = np.atleast_1d(data).tolist()
    #                 elif isinstance(data, (str, bool, int, float)):
    #                     pass
    #                 elif np.issubdtype(data.dtype, np.integer):
    #                     data = int(data)
    #                 elif np.issubdtype(data.dtype, np.float):
    #                     data = float(data)

    #                 updates.append(UpdateOne(
    #                     {'_id': pr_doc['_id']},
    #                     {'$set': {key: data}}
    #                 ))

    #     res = self.properties.bulk_write(updates, ordered=False)
    #     nmatch = res.bulk_api_result['nMatched']
    #     if nmatch:
    #         warnings.warn('Modified {} properties'.format(nmatch))

    #     return res


    def dataset_from_markdown(
        self,
        html_file_path,
        generator=False,
        verbose=False,
    ):
        """
        Loads a Dataset from a markdown file.

        Args:

            html_file_path (str):
                The full path to the markdown file

            generator (bool, default=False):
                If True, uses a generator when inserting data.

            verbose (bool, default=False):
                If True, prints progress bars

        Returns:

            dataset (Dataset):
                The Dataset object after adding it to the Database
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

        images = []
        storage_table = parser.get_data('Storage format')
        header = storage_table[0]
        for row in storage_table:

            # Check if already exists in Database
            if row[header.index('Format')] == 'mongo':
                return self.get_dataset(
                    row[header.index('File')], resync=True, verbose=verbose
                )

            # Else, need to build from scratch
            try:
                elements = [l.strip() for l in row[header.index('Elements')].split(',')]
            except:
                raise BadTableFormatting(
                    "Error when parsing 'Elements' column of 'Storage format' table"
                )

            # Load Configurations
            images.append(load_data(
                file_path=os.path.join(base_path, row[header.index('File')][0][1]),
                file_format=row[header.index('Format')],
                name_field=row[header.index('Name field')],
                elements=elements,
                default_name=parser.data['Name'],
                verbose=verbose
            ))

        images = itertools.chain.from_iterable(images)

        # Add Property definitions and load property_map
        property_map = {}
        for prop in parser.get_data('Properties')[1:]:
            pid         = prop[0][0]
            kim_field   = prop[1]
            ase_field   = prop[2]
            units       = prop[3]

            if units == 'None':
                units = None

            # pid, kim_field, ase_field, units = prop


            if pid in KIM_PROPERTIES:
                pname = definition = pid
            elif isinstance(pid, tuple):
                pname = pid[0]

                edn_path = os.path.abspath(os.path.join(base_path, pid[1]))
                definition = kim_edn.load(edn_path)

            definition['property-id'] = pname
            self.insert_property_definition(definition)

            pid_dict = property_map.setdefault(pname, {})

            if kim_field in pid_dict:
                raise BadTableFormatting(
                    "Duplicate property field found"
                )

            pid_dict[kim_field] = {
                'field': ase_field,
                'units': units
            }

        # Extract property settings
        property_settings = {}
        pso_table = parser.get_data('Property settings')
        header = pso_table[0]
        for row in pso_table[1:]:
            files = []
            if 'Files' in header:
                for ftup in row[header.index('Files')]:
                    # Files will be stored as hyperlink tuples
                    fpath = os.path.abspath(os.path.join(base_path, ftup[1]))
                    with open(fpath, 'r') as f:
                        files.append((
                            ftup[0],
                            '\n'.join([_.strip() for _ in f.readlines()])
                        ))

            property_settings[row[header.index('Property')]] = PropertySettings(
                method=row[header.index('Method')],
                description=row[header.index('Description')],
                labels=[
                    _.strip() for _ in row[header.index('Labels')].split(',')
                ] if 'Labels' in header else [],
                files=files,
            )

        ids = list(self.insert_data(
            images,
            property_map=property_map,
            property_settings=property_settings,
            generator=generator,
            verbose=verbose,
        ))

        all_co_ids, all_pr_ids = list(zip(*ids))

        # Extract configuration sets and trigger CS refresh
        cs_ids = []

        config_sets = parser.get_data('Configuration sets')
        header = config_sets[0]
        for row in config_sets[1:]:
            query = literal_eval(row[header.index('Query')])
            query['_id'] = {'$in': all_co_ids}

            co_ids = self.get_data(
                'configurations',
                fields='_id',
                query=query,
                ravel=True
            ).tolist()

            cs_id = self.insert_configuration_set(
                co_ids,
                description=row[header.index('Description')],
                verbose=True
            )

            cs_ids.append(cs_id)

        # Define the Dataset
        ds_id = self.insert_dataset(
            cs_ids=cs_ids,
            pr_ids=all_pr_ids,
            name='Mo_PRM2019',
            authors=parser.data['Authors'],
            links=parser.data['Links'],
            description=parser.data['Description'],
            verbose=verbose,
        )

        # Extract labels and trigger label refresh for configurations
        labels = parser.get_data('Configuration labels')
        header = labels[0]
        for row in labels[1:]:
            query = literal_eval(row[header.index('Query')])
            query['_id'] = {'$in': all_co_ids}

            self.apply_labels(
                dataset_id=ds_id,
                collection_name='configurations',
                query=query,
                labels=[
                    l.strip() for l in row[header.index('Labels')].split(',')
                ],
                verbose=True
            )

        return self.get_dataset(ds_id, resync=True, verbose=verbose)


    def dataset_to_markdown(
        self,
        ds_id,
        base_folder,
        html_file_name,
        data_file_name,
        data_format,
        name_field=ATOMS_NAME_FIELD,
        histogram_fields=None,
        yscale='linear',
        ):
        """
        Saves a Dataset and writes a properly formatted markdown file. In the
        case of a Dataset that has child Dataset objects, each child Dataset
        is written to a separate sub-folder.

        Args:

            ds_id (str):
                The ID of the dataset.

            base_folder (str):
                Top-level folder in which to save the markdown and data files

            html_file_name (str):
                Name of file to save markdown to

            data_file_name (str):
                Name of file to save configuration and properties to

            data_format (str, default='mongo'):
                Format to use for data file. If 'mongo', does not save the
                configurations to a new file, and instead adds the ID of the
                Dataset in the Mongo Database.

            name_field (str):
                The name of the field that should be used to generate
                configuration names

            histogram_fields (list, default=None):
                The property fields to include in the histogram plot. If None,
                plots all fields.

            yscale (str, default='linear'):
                Scaling to use for histogram plotting
        """

        template = \
"""
# Summary
|Chemical systems|Element ratios|# of properties|# of configurations|# of atoms|
|---|---|---|---|---|
|{}|{}|{}|{}|{}|

# Name

{}

# Authors

{}

# Links

{}

# Description

{}

# Storage format

|Elements|File|Format|Name field|
|---|---|---|---|
| {} | {} | {} | {} |

# Properties

|Property|KIM field|ASE field|Units
|---|---|---|---|
{}

# Property settings

|ID|Method|Description|Labels|Files|
|---|---|---|---|---|
{}

# Configuration sets

|ID|Description|# of structures| # of atoms|
|---|---|---|---|
{}

# Configuration labels

|Labels|Counts|
|---|---|
{}

# Figures
![The results of plot_histograms](histograms.png)
"""

        if not os.path.isdir(base_folder):
            os.mkdir(base_folder)

        html_file_name = os.path.join(base_folder, html_file_name)

        dataset = self.get_dataset(ds_id)['dataset']

        definition_files = {}
        for pname in dataset.aggregated_info['property_types']:
            definition = self.get_property_definition(pname)

            def_fpath = os.path.join(base_folder, f'{pname}.edn')

            json.dump(definition, open(def_fpath, 'w'))

            definition_files[pname] = def_fpath

        property_map = {}
        for pr_doc in self.properties.find(
            {'_id': {'$in': dataset.property_ids}}
            ):
            if pr_doc['type'] not in property_map:
                property_map[pr_doc['type']] = {
                    f: {
                        'field': f,
                        'units': v['source-unit'] if 'source-unit' in v else None
                    }
                    for f,v in pr_doc[pr_doc['type']].items()
                }

        agg_info = dataset.aggregated_info

        # TODO: property settings should just be attached to the atoms?

        # TODO: how will you handle property settings? They are now fields on a
        # CO, so should you store them in the XYZ file? Should
        # get_configurations also have an attach_settings option??
        # This could duplicate a lot of data. Could just export PSOs as separate files

        property_settings = {}
        for pso_doc in self.property_settings.find(
            {'relationships.properties': {'$in': dataset.property_ids}}
            ):
            property_settings[pso_doc['_id']] = self.get_property_settings(
                pso_doc['_id']
            )

        configuration_sets = {
            csid: self.get_configuration_set(csid)['configuration_set']
            for csid in dataset.configuration_set_ids
        }

        # Write the markdown file
        with open(html_file_name, 'w') as html:

            formatting_arguments = []

            # Summary
            formatting_arguments.append(', '.join(agg_info['chemical_systems']))

            tmp = []
            for e, er in agg_info['total_elements_ratios'].items():
                tmp.append('{} ({:.1f}%)'.format(e, er*100))

            formatting_arguments.append(', '.join(tmp))

            formatting_arguments.append(sum(agg_info['property_types_counts']))
            formatting_arguments.append(agg_info['nconfigurations'])
            formatting_arguments.append(agg_info['nsites'])

            # Name
            formatting_arguments.append(dataset.name)

            # Authors
            formatting_arguments.append('\n\n'.join(dataset.authors))

            # Links
            formatting_arguments.append('\n\n'.join(dataset.links))

            # Description
            formatting_arguments.append(dataset.description)

            # Storage format
            formatting_arguments.append(', '.join(agg_info['elements']))

            if data_format == 'mongo':
                formatting_arguments.append(ds_id)
            else:
                formatting_arguments.append(
                    '[{}]({})'.format(data_file_name, data_file_name)
                )

            formatting_arguments.append(data_format)

            formatting_arguments.append(name_field)

            tmp = []
            for pid, fdict in property_map.items():
                for f,v in fdict.items():
                    tmp.append(
                        '| {} | {} | {} | {}'.format(
                            '[{}]({})'.format(pid, definition_files[pid]),
                            f,
                            v['field'],
                            v['units']
                        )
                    )

            formatting_arguments.append('\n'.join(tmp))

            tmp = []
            for pso_id, pso in property_settings.items():
                tmp.append('| {} | {} | {} | {} | {} |'.format(
                    pso_id,
                    pso.method,
                    pso.description,
                    ', '.join(pso.labels),
                    ', '.join('[{}]({})'.format(f, f) for f in pso.files)
                ))

            formatting_arguments.append('\n'.join(tmp))

            tmp = []
            for cs_id, cs in configuration_sets.items():
                tmp.append('| {} | {} | {} | {} |'.format(
                    cs_id,
                    cs.description,
                    cs.aggregated_info['nconfigurations'],
                    cs.aggregated_info['nsites']
                ))

            formatting_arguments.append('\n'.join(tmp))

            tmp = []
            for l, lc in zip(
                dataset.aggregated_info['configuration_labels'], dataset.aggregated_info['configuration_labels_counts']
                ):

                tmp.append('| {} | {} |'.format(l, lc))

            formatting_arguments.append('\n'.join(tmp))

            html.write(template.format(*formatting_arguments))


        # Save figures
        if histogram_fields is None:
            histogram_fields = dataset.aggregated_info['property_fields']

        if len(histogram_fields) > 0:
            fig = self.plot_histograms(
                histogram_fields,
                ids=dataset.property_ids,
                yscale=yscale,
                method='matplotlib'
            )

            plt.savefig(os.path.join(base_folder, 'histograms.png'))
            plt.close()

        # Copy any PSO files
        all_file_names = []
        for pso_id, pso in property_settings.items():
            for fi, f in enumerate(pso.files):
                new_name = os.path.join(
                    base_folder,
                    pso_id + '_' + os.path.split(f)[-1]
                )
                shutil.copyfile(f, new_name)

                all_file_names.append(new_name)
                pso.files[fi] = new_name

        if data_format == 'xyz':
            data_format = 'extxyz'

        if data_format != 'mongo':
            data_file_name = os.path.join(base_folder, data_file_name)

            images = self.get_configurations(
                ids=list(set(itertools.chain.from_iterable(
                    cs.configuration_ids for cs in configuration_sets.values()
                ))),
                attach_properties=True,
                generator=True,
            )

            ase_write(
                data_file_name,
                images=images,
                format=data_format,
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
    generator=True,
    verbose=False,
    **kwargs,
    ):
    """
    Loads a list of Configuration objects.

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

        generator (bool, default=True):
            If True, returns a generator of Configurations. If False, returns a
            list.

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

        results = converter.load(
            file_path,
            name_field=name_field,
            elements=elements,
            default_name=default_name,
            labels_field=labels_field,
            glob_string=glob_string,
            verbose=verbose,
            **kwargs,
        )

    elif file_format in ['xyz', 'extxyz', 'cfg']:
        if file_format in ['xyz', 'extxyz']:
            converter = EXYZConverter()
        elif file_format == 'cfg':
            converter = CFGConverter()

        results = converter.load(
            file_path,
            name_field=name_field,
            elements=elements,
            default_name=default_name,
            labels_field=labels_field,
            verbose=verbose,
        )
    else:
        raise RuntimeError(
            "Invalid `file_format`. Must be one of "\
                "['xyz', 'extxyz', 'cfg', 'folder']"
        )

    return results if generator else list(results)

class ConcatenationException(Exception):
    pass

class InvalidGroupError(Exception):
    pass

class MissingEntryError(Exception):
    pass

class DuplicateDefinitionError(Exception):
    pass
