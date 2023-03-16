import json
import datetime
import warnings
import itertools
import string
from math import ceil

import numpy as np
from tqdm import tqdm
import multiprocessing
from copy import deepcopy
from hashlib import sha512
from functools import partial
from pymongo import MongoClient, UpdateOne
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from ase.io import write as ase_write
from ase import Atoms
from unidecode import unidecode
import periodictable
import time
from kim_property.definition import check_property_definition
from kim_property.definition import PROPERTY_ID as VALID_KIM_ID
from django.utils.crypto import get_random_string
from cfkit import (
    ID_FORMAT_STRING,
    _CONFIGS_COLLECTION, _PROPS_COLLECTION, _METADATA_COLLECTION, _DATAOBJECT_COLLECTION,
    _CONFIGSETS_COLLECTION, _PROPDEFS_COLLECTION, _DATASETS_COLLECTION,
    ATOMS_NAME_FIELD, MAX_STRING_LENGTH,
    SHORT_ID_STRING_NAME, EXTENDED_ID_STRING_NAME
)
from cfkit.configuration import BaseConfiguration, AtomicConfiguration
from cfkit.property import Property
from cfkit.configuration_set import ConfigurationSet
from cfkit.converters import CFGConverter, EXYZConverter, FolderConverter
from cfkit.dataset import Dataset
from cfkit.metadata import Metadata
from cfkit.data_object import DataObject


class MongoDatabase(MongoClient):
    """
    A MongoDatabase stores all of the data in Mongo documents, and
    provides additinal functionality like filtering and optimized queries.

    The Mongo database has the following structure

    .. code-block:: text

        /configurations
            _id
            short-id 
            atomic_numbers
            positions
            cell
            pbc
            names
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
                property_instances
                configuration_sets

        /property_definitions
            _id
            short-id
            definition

        /properties
            _id
            short-id
            type
            property_name
                each field in the property definition
            methods
            last_modified
            relationships
                metadata
                configurations

        /metadata
            hash
            dict

        /configuration_sets
            _id
            short-id
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
            short-id
            extended-id
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
            relationships
                property_instances
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

        metadata (Collection):
            A Mongo collection of metadata documents

        configuration_sets (Collection):
            A Mongo collection of configuration set documents

        datasets (Collection):
            A Mongo collection of dataset documents
    """

    # TODO: Should database be instantiated with nprocs, or should it be passed in as an
    #       argument to methods in which this would be relevant
    def __init__(
            self, database_name, configuration_type=AtomicConfiguration, nprocs=1, uri=None,
            drop_database=False, user=None, pwrd=None, port=27017,
            *args, **kwargs
    ):
        """
        Args:

            database_name (str):
                The name of the database

            configuration_type (Configuration, default=BaseConfiguration):
                The configuration type that will be stored in the database.

            nprocs (int):
                The size of the processor pool

            uri (str):
                The full Mongo URI

            drop_database (bool, default=False):
                If True, deletes the existing Mongo database.

            user (str, default=None):
                Mongo server username

            pwrd (str, default=None):
                Mongo server password

            port (int, default=27017):
                Mongo server port number

            *args, **kwargs (list, dict):
                All additional arguments will be passed directly to the
                MongoClient constructor.


        """
        self.configuration_type = configuration_type
        self.uri = uri
        self.login_args = args
        self.login_kwargs = kwargs

        self.user = user
        self.pwrd = pwrd
        self.port = port

        if self.uri is not None:
            super().__init__(self.uri, *args, **kwargs)
        else:
            if user is None:
                super().__init__('localhost', self.port, *args, **kwargs)
            else:
                super().__init__(
                    'mongodb://{}:{}@localhost:{}/'.format(
                        self.user, self.pwrd, self.port, *args, **kwargs
                    )
                )

        self.database_name = database_name

        if drop_database:
            self.drop_database(database_name)

        self.configurations = self[database_name][_CONFIGS_COLLECTION]
        self.property_instances = self[database_name][_PROPS_COLLECTION]
        self.property_definitions = self[database_name][_PROPDEFS_COLLECTION]
        self.data_objects = self[database_name][_DATAOBJECT_COLLECTION]
        self.metadata = self[database_name][_METADATA_COLLECTION]
        self.configuration_sets = self[database_name][_CONFIGSETS_COLLECTION]
        self.datasets = self[database_name][_DATASETS_COLLECTION]

        self.property_definitions.create_index(
            keys='definition.property-name', name='definition.property-name',
            unique=True
        )

        self.configuration_sets.create_index(
            keys=SHORT_ID_STRING_NAME, name=SHORT_ID_STRING_NAME, unique=True
        )
        self.datasets.create_index(
            keys=SHORT_ID_STRING_NAME, name=SHORT_ID_STRING_NAME, unique=True
        )

        self.configurations.create_index(
            keys='hash', name='hash', unique=True
        )
        self.property_instances.create_index(
            keys='hash', name='hash', unique=True
        )
        self.metadata.create_index(
            keys='hash', name='hash', unique=True
        )
        self.data_objects.create_index(
            keys='hash', name='hash', unique=True
        )
        self.configuration_sets.create_index(
            keys='hash', name='hash', unique=True
        )
        self.datasets.create_index(
            keys='hash', name='hash', unique=True
        )
        self.configurations.create_index(
            keys=SHORT_ID_STRING_NAME, name=SHORT_ID_STRING_NAME, unique=True
        )
        self.property_instances.create_index(
            keys=SHORT_ID_STRING_NAME, name=SHORT_ID_STRING_NAME, unique=True
        )
        self.metadata.create_index(
            keys=SHORT_ID_STRING_NAME, name=SHORT_ID_STRING_NAME, unique=True
        )
        self.data_objects.create_index(
            keys=SHORT_ID_STRING_NAME, name=SHORT_ID_STRING_NAME, unique=True
        )
        self.property_instances.create_index(
            keys='relationships.metadata', name='pi_relationships.metadata'
        )
        self.configuration_sets.create_index(
            keys='relationships.datasets', name='cs_relationships.datasets'
        )
        self.configurations.create_index(
            keys='relationships.metadata', name='co_relationships.metadata'
        )
        self.configurations.create_index(
            keys='relationships.data_objects', name='co_relationships.data_objects'
        )
        self.property_instances.create_index(
            keys='relationships.data_objects', name='pi_relationships.data_objects'
        )
        self.data_objects.create_index(
            keys='relationships.datasets', name='do_relationships.datasets'
        )
        self.configurations.create_index(
            keys='relationships.configuration_sets', name='co_relationships.configuration_sets'
        )
        self.nprocs = nprocs


    def insert_data(
            self,
            configurations,
            property_map=None,
            co_md_map=None,
            transform=None,
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
                                    'method': 'VASP',
                                    'description': 'A static VASP calculation',
                                    'files': None,
                                    'labels': ['Monkhorst-Pack'],

                                    'xc-functional': {'field': 'xcf', 'units': None}
                                }
                            }
                        }


                If None, only loads the configuration information (atomic
                numbers, positions, lattice vectors, and periodic boundary
                conditions).

                The '_settings' key is a special key that can be used to specify
                the contents of a PropertySettings object that will be
                constructed and linked to each associated property instance.


            co_md_map (dict):
                A dictionary that is used to specify how to load metadata
                defined for a configuration.

            transform (callable, default=None):
                If provided, `transform` will be called on each configuration in
                :code:`configurations` as :code:`transform(configuration)`.
                Note that this happens before anything else is done. `transform`
                should modify the Configuration in-place.


            verbose (bool, default=False):
                If True, prints a progress bar

        Returns:

            ids (list):
                A list of (config_id, property_id) tuples of the inserted data.
                If no properties were inserted, then property_id will be None.

        """

        if self.uri is not None:
            mongo_login = self.uri
        else:
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
            'last_modified', 'definition', '_id', SHORT_ID_STRING_NAME, '_settings',
            'property-name', EXTENDED_ID_STRING_NAME, '_metadata'
        }

        # Sanity checks for property map
        for pname, pdict_list in property_map.items():
            pd_doc = self.property_definitions.find_one({
                'definition.property-name': pname
            })

            if pd_doc:
                # property_field_name, {'ase_field': ..., 'units': ...}
                for pdict in pdict_list:
                    for k, pd in pdict.items():
                        if k in ignore_keys:
                            continue
                        if k not in pd_doc['definition']:
                            warnings.warn(
                                'Provided field "{}" in property_map does not match ' \
                                'property definition'.format(k)
                            )
                        if 'value' in pd:
                            if 'field' in pd and pd['field'] is not None:
                                raise RuntimeError(
                                    "Error with key '{}'. property_map must specify exactly ONE of 'field' or 'value'".format(
                                        k)
                                )
                        else:
                            if ('field' not in pd) or (pd['field'] is None):
                                raise RuntimeError(
                                    "Error with key '{}'. property_map must specify exactly ONE of 'field' or 'value'".format(
                                        k)
                                )

                        if 'units' not in pd:
                            raise RuntimeError(
                                "Must specify all 'units' sections in " \
                                "property_map. Set value to None if no units."
                            )
            else:
                warnings.warn(
                    'Property name "{}" in property_map does not have an ' \
                    'existing definition in the database.'.format(pname)
                )

        if 1:
            configurations = list(configurations)

            n = len(configurations)
            k = self.nprocs

            split_configs = [
                configurations[i * (n // k) + min(i, n % k):(i + 1) * (n // k) + min(i + 1, n % k)]
                for i in range(k)
            ]

            pfunc = partial(
                self._insert_data,
                mongo_login=mongo_login,
                database_name=self.database_name,
                property_map=property_map,
                co_md_map=co_md_map,
                transform=transform,
                verbose=verbose
            )

            pool = multiprocessing.Pool(self.nprocs)

            return list(itertools.chain.from_iterable(
                pool.map(pfunc, split_configs)
            ))

    @staticmethod
    def _insert_data(
            configurations, database_name, mongo_login, co_md_map=None,
            property_map=None, transform=None,
            verbose=False
    ):
        if isinstance(mongo_login, int):
            client = MongoClient('localhost', mongo_login)
        else:
            client = MongoClient(mongo_login)
        coll_configurations = client[database_name][_CONFIGS_COLLECTION]
        coll_properties = client[database_name][_PROPS_COLLECTION]
        coll_data_objects = client[database_name][_DATAOBJECT_COLLECTION]
        coll_property_definitions = client[database_name][_PROPDEFS_COLLECTION]
        coll_metadata = client[database_name][_METADATA_COLLECTION]

        if isinstance(configurations, BaseConfiguration):
            configurations = [configurations]

        if property_map is None:
            property_map = {}

        property_definitions = {}
        for pname in property_map:
            doc = coll_property_definitions.find_one({
                'definition.property-name': pname
            })

            if doc is None:
                raise RuntimeError(
                    "Property definition '{}' does not exist. " \
                    "Use insert_property_definition() first".format(pname)
                )
            else:
                property_definitions[pname] = doc['definition']

        ignore_keys = {
            'property-id', 'property-title', 'property-description',
            'last_modified', 'definition', '_id', SHORT_ID_STRING_NAME, 'settings',
            'property-name',
        }

        expected_keys = {
            pname: [set(
                # property_map[pname][f]['field']
                pmap[f]['field']
                for f in property_definitions[pname].keys() - ignore_keys
                if property_definitions[pname][f]['required']
                and 'field' in pmap[f]
            ) for pmap in property_map[pname]]
            for pname in property_map
        }

        insertions = []
        ca_ids = set()
        config_docs = []
        property_docs = []
        calc_docs = []
        meta_docs = []
        meta_update_dict = {}
        # Add all of the configurations into the Mongo server
        ai = 1
        for atoms in tqdm(
                configurations,
                desc='Preparing to add configurations to Database',
                disable=not verbose,
        ):
            property_docs_do = []
            calc_lists = {}
            calc_lists['PI'] = []
            calc_lists['PI_type'] = []
            if transform:
                transform(atoms)

            c_update_doc, c_hash = _build_c_update_doc(atoms)
            c_update_doc['$addToSet']['relationships.data_objects']={}
            calc_lists['CO'] = c_hash
            calc_lists['CO_hill'] = atoms.configuration_summary()['chemical_formula_hill']
            if co_md_map:
                co_md = Metadata.from_map(d=co_md_map, source=atoms)
                co_md_set_on_insert = _build_md_insert_doc(co_md)
                co_md_update_doc = {  # update document
                    '$setOnInsert': co_md_set_on_insert,
                    '$set': {
                        'last_modified': datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
                    },
                }

                meta_docs.append(UpdateOne(
                    {'hash': str(co_md._hash)},
                    co_md_update_doc,
                    upsert=True,
                    hint='hash',
                ))
                c_update_doc['$addToSet']['relationships.metadata'] = {'$each': ['MD_%s' % str(co_md._hash)]}
            available_keys = set().union(atoms.info.keys(), atoms.arrays.keys())
            p_hash = None

            new_p_hashes = []
            for pname, pmap_list in property_map.items():
                for pmap_i, pmap in enumerate(pmap_list):
                    pmap_copy = dict(pmap)
                    if '_metadata' in pmap_copy:
                        del pmap_copy['_metadata']

                    # Pre-check to avoid having to delete partially-added properties
                    missing_keys = expected_keys[pname][pmap_i] - available_keys
                    if missing_keys:
                        # warnings.warn(
                        #     "Configuration is missing keys {} for Property"\
                        #     "Instance construction. Available keys: {}. "\
                        #     "Skipping".format(
                        #         missing_keys, available_keys
                        #     )
                        # )
                        continue
                    # checks if property is present in atoms->if not, skip over it

                    available = 0
                    for k in pmap_copy.keys():
                        if 'value' in pmap_copy[k]:
                            available += 1
                        elif pmap_copy[k]['field'] in available_keys:
                            available += 1
                    if not available:
                        continue

                    metadata_hashes = []
                    # Attach property metadata, if any were given
                    if '_metadata' in pmap:
                        pi_md = Metadata.from_map(d=pmap['_metadata'], source=atoms)

                        '''
                        gathered_fields = {}
                        for pi_md_field in pi_md_map.keys():
                            if 'value' in pi_md_map[pi_md_field]:
                                v = pi_md_map[pi_md_field]['value']
                            else:
                                field_key = pi_md_map[pi_md_field]['field']

                                if field_key in atoms.info:
                                    v = atoms.info[field_key]
                                elif field_key in atoms.arrays:
                                    v = atoms.arrays[field_key]
                                else:
                                # No keys are required; ignored if missing
                                    continue
                            if "units" in pi_md_map[pi_md_field]:
                                gathered_fields[pi_md_field] = {
                                    'source-value': v,
                                    'source-unit':  pi_md_map[pi_md_field]['units'],
                                }
                            else:
                                gathered_fields[pi_md_field] = {
                                    'source-value': v
                                }

                        pi_md = Metadata(linked_type='PI',
                                         metadata=gathered_fields
                                         )
'''
                        '''
                        
                        pi_md_set_on_insert = {
                            'hash': str(pi_md._hash),
                        }

                        for gf, gf_dict in pi_md.metadata.items():
                            if isinstance(gf_dict['source-value'], (int, float, str)):
                                # Add directly
                                pi_md_set_on_insert[gf] = {
                                    'source-value': gf_dict['source-value']
                                }
                            else:
                                # Then it's array-like and should be converted to a list
                                pi_md_set_on_insert[gf] = {
                                    'source-value': np.atleast_1d(
                                        gf_dict['source-value']
                                    ).tolist()
                                }

                            if 'source-unit' in gf_dict:
                                pi_md_set_on_insert[gf]['source-unit'] = gf_dict['source-unit']
'''
                        pi_md_set_on_insert = _build_md_insert_doc(pi_md)
                        prop = Property.from_definition(
                            definition=property_definitions[pname],
                            configuration=atoms,
                            property_map=pmap_copy
                        )
                        p_hash = str(hash(prop))

                        new_p_hashes.append(p_hash)

                        pi_md_update_doc = {  # update document
                            '$setOnInsert': pi_md_set_on_insert,
                            '$set': {
                                'last_modified': datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
                            },
                        }
                        #meta_update_dict[str(pi_md._hash)] = pi_md_update_doc

                        meta_docs.append(UpdateOne(
                             {'hash': str(pi_md._hash)},
                             pi_md_update_doc,
                             upsert=True,
                             hint='hash',
                         ))

                        metadata_hashes.append(str(pi_md._hash))

                    else:
                        prop = Property.from_definition(
                            definition=property_definitions[pname],
                            configuration=atoms,
                            property_map=pmap_copy
                        )
                        p_hash = str(hash(prop))

                        new_p_hashes.append(p_hash)
                    calc_lists['PI'].append(p_hash)
                    calc_lists['PI_type'].append(pname)
                    # Prepare the property instance EDN document
                    setOnInsert = {}
                    # for k in property_map[pname]:
                    for k in pmap:
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
                        # TODO: Look at: can probably safely move out one level
                        p_update_doc = {
                            '$addToSet': {
                                # PR -> PSO pointer
                                'relationships.metadata': {
                                    '$each': ['MD_' + i for i in metadata_hashes]
                                },
                                'relationships.data_objects': {},
                            },
                            '$setOnInsert': {
                                'hash': p_hash,
                                SHORT_ID_STRING_NAME: 'PI_' + p_hash,
                                'type': pname,
                                pname: setOnInsert
                            },
                            '$set': {
                                'last_modified': datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
                            }
                        }

                    property_docs_do.append(p_update_doc)

                    # c_update_doc['$addToSet']['relationships.property_instances']['$each'].append(
                    #    'PI_'+p_hash
                    # ) #can probably safely remove since linked to DO

                    # insertions.append((c_hash, p_hash))

#            config_docs.append(
#                UpdateOne(
#                    {'hash': c_hash},
#                    c_update_doc,
#                    upsert=True,
#                    hint='hash',
#                )
#            )

            calc = DataObject(calc_lists['CO'], calc_lists['PI'])
            ca_hash = str(calc._hash)
            ca_ids.add(ca_hash)
            ca_insert_doc = _build_ca_insert_doc(calc)
            ca_insert_doc['chemical_formula_hill'] = calc_lists['CO_hill']
            ca_update_doc = {  # update document
                '$setOnInsert': ca_insert_doc,
                '$addToSet': {'property_types': {'$each': calc_lists['PI_type']}},
                '$inc': {
                    'ncounts': 1
                },

                '$set': {
                    'last_modified': datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
                },
            }
            calc_docs.append(
                UpdateOne(
                    {'hash': ca_hash},
                    ca_update_doc,
                    upsert=True,
                    hint='hash',
                )
            )
            for pi_doc in property_docs_do:
                pi_doc['$addToSet'] ['relationships.data_objects']={'$each': ["DO_%s" %ca_hash]}
                property_docs.append(UpdateOne(
                        {'hash': pi_doc['$setOnInsert']['hash']},
                        pi_doc,
                        upsert=True,
                        hint='hash',
                    ))
            c_update_doc['$addToSet']['relationships.data_objects'] = {'$each': ["DO_%s" %ca_hash]}
            config_docs.append(
                UpdateOne(
                    {'hash': c_hash},
                    c_update_doc,
                    upsert=True,
                    hint='hash',
                )
            )

            insertions.append((c_hash, ca_hash))
            # TODO: Fix this as it duplicates things when no property is present
            # if not p_hash:
            # Only yield if something wasn't yielded earlier
            #    insertions.append((c_hash, ca_hash))

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
        if calc_docs:
            res = coll_data_objects.bulk_write(calc_docs, ordered=False)
            nmatch = res.bulk_api_result['nMatched']
            if nmatch:
                warnings.warn(
                    '{} duplicate data objects detected'.format(nmatch)
                )

        if meta_docs:
            res = coll_metadata.bulk_write(
                meta_docs,
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
                    '{} duplicate metadata objects detected'.format(nmatch)
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
                for more details. If a string is provided, it must be the full
                path to an existing property definition.

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
            # definition = KIM_PROPERTIES[definition]
            with open(definition, 'r') as f:
                definition = json.load(f)

        if self.property_definitions.count_documents(
                {'definition.property-name': definition['property-name']}
        ):
            warnings.warn(
                "Property definition with name '{}' already exists. " \
                "Using existing definition.".format(
                    definition['property-name']
                )
            )

        dummy_dict = deepcopy(definition)

        if 'property-id' not in dummy_dict:
            dummy_dict['property-id'] = 'tag:@,0000-00-00:property/'
            dummy_dict['property-id'] += definition['property-name']
        else:

            # Spoof if necessary
            if VALID_KIM_ID.match(dummy_dict['property-id']) is None:
                # Invalid ID. Try spoofing it
                dummy_dict['property-id'] = 'tag:@,0000-00-00:property/'
                dummy_dict['property-id'] += definition['property-id']
                warnings.warn(
                    "Invalid KIM property-id; " \
                    "Temporarily renaming to {}. " \
                    "See https://openkim.org/doc/schema/properties-framework/ " \
                    "for more details.".format(dummy_dict['property-id'])
                )

        # Hack to avoid the fact that "property-name" has to be a dictionary
        # in order for OpenKIM's check_property_definition to work
        tmp = dummy_dict['property-name']
        del dummy_dict['property-name']

        check_property_definition(dummy_dict)

        dummy_dict['property-name'] = tmp

        self.property_definitions.update_one(
            {'definition.property-name': definition['property-name']},
            {
                '$setOnInsert': {
                    'definition': dummy_dict
                }
            },
            upsert=True,
            hint='definition.property-name',
        )

    def get_property_definition(self, name):
        """Returns a property definition using its 'definition.property-name' key"""
        return self.property_definitions.find_one({'definition.property-name': name})

    def insert_property_settings(self, ps_object):
        """
        Inserts a new property settings object into the database by creating
        and populating the necessary groups in :code:`/root/property_settings`.

        Args:

            ps_object (PropertySettings)
                The :class:`~colabfit.tools.property_settings.PropertySettings`
                object to insert into the database.


        Returns:

            ps_hash (str):
                The hash of the inserted property settings object.
        """

        ps_hash = str(ps_object._hash)
        self.property_settings.update_one(
            {'hash': ps_hash},
            {
                '$addToSet': {
                    'labels': {'$each': list(ps_object.labels)}
                },
                '$setOnInsert': {
                    'hash': str(ps_object._hash),
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
            upsert=True,
            hint='hash',
        )

        return ps_hash

    def get_property_settings(self, pso_hash):
        pso_doc = self.property_settings.find_one({'hash': pso_hash})
        return PropertySettings(
            method=pso_doc['method'],
            description=pso_doc['description'],
            labels=set(pso_doc['labels']),
            files=[
                (d['file_name'], d['file_contents'])
                for d in pso_doc['files']
            ]
        )

    def query_in_batches(self,
                         collection_name,
                         query_key,
                         query_list,
                         other_query=None,
                         return_key=None,
                         batch_size=100000,
                         **kwargs):

        """
            Queries the database in batches and returns results. This should be used for large queries when building CS
            and DS. Queries are called using '$in' functionality

            Args:

                collection_name (str):
                    The name of a collection in the database.

                query_key (str):
                    Key to use in an '$in' query

                query_list (list):
                    List of values to search over using '$in'

                other_query (dict, default=None):
                    Any other query that should also be performed along with '$in' query

                return_key (str, default=None):
                    If not None, values corresponding to return_key are yielded, otherwise everything is yielded.

                batch_size (int, default=100000):
                    Number of values that the query searches over using $in functionality

               Other arguments in a typical PyMongo 'find'

            Returns:

                data (dict):
                    key = k for k in keys. val = in-memory data
            """

        nbatches = ceil(len(query_list)/batch_size)
        collection = self[self.database_name][collection_name]
        for i in range(nbatches):
            if i+1 < nbatches:
                if other_query is not None:
                    cursor = collection.find(
                        {query_key: {'$in': query_list[i*batch_size:(i+1)*batch_size]}}.update(other_query), **kwargs)
                else:
                    cursor = collection.find(
                        {query_key: {'$in': query_list[i * batch_size:(i + 1) * batch_size]}}, **kwargs)
            else:
                if other_query is not None:
                    cursor = collection.find(
                        {query_key: {'$in': query_list[i * batch_size:]}}.update(other_query), **kwargs)
                else:
                    cursor = collection.find(
                        {query_key: {'$in': query_list[i * batch_size:]}}, **kwargs)
            for j in cursor:
                if return_key is not None:
                    yield j[return_key]
                else:
                    yield j

    # @staticmethod
    def get_data(
            self, collection_name,
            fields,
            query=None,
            hashes=None,
            keep_hashes=False,
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
                query={SHORT_ID_STRING_NAME: {'$in': <list_of_property_IDs>}},
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

            hashes (list):
                The list of hashes to return the data for. If None, returns the
                data for the entire collection. Note that this information can
                also be provided using the :code:`query` argument.

            keep_hashes (bool, default=False):
                If True, includes the "hash" field as one of the returned values.

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

        if hashes is not None:
            if isinstance(hashes, str):
                hashes = [hashes]
            elif isinstance(hashes, np.ndarray):
                hashes = hashes.tolist()

            query['hash'] = {'$in': hashes}

        if isinstance(fields, str):
            fields = [fields]

        retfields = {k: 1 for k in fields}

        if keep_hashes:
            retfields['hash'] = 1

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

        for k, v in data.items():
            # data[k] = np.array(data[k])
            # TODO: Standardize=> Currently, output is array if numpy operations are used, otherwise it's list
            if concatenate or ravel:
                try:
                    data[k] = np.concatenate(v)
                except:
                    data[k] = np.array(v)

            if vstack:
                data[k] = np.vstack(v)

        if ravel:
            for k, v in data.items():
                data[k] = v.ravel()

        if len(retfields) == 1:
            return data[list(retfields.keys())[0]]
        else:
            return data

    def get_configuration(self, i, property_hashes=None, attach_properties=False):
        """
        Returns a single configuration by calling :meth:`get_configurations`
        """
        return self.get_configurations(
            [i], property_hashes=property_hashes, attach_properties=attach_properties
        )[0]

    def get_configurations(
            self, configuration_hashes,
            property_hashes=None,
            attach_properties=False,
            attach_settings=False,
            generator=False,
            verbose=False
    ):
        """
        A generator that returns in-memory Configuration objects one at a time
        by loading the atomic numbers, positions, cells, and PBCs.

        Args:

            configuration_hashes (list or 'all'):
                A list of string hashes specifying which Configurations to return.
                If 'all', returns all of the configurations in the database.

            property_hashes (list, default=None):
                A list of Property hashes. Used for limiting searches when
                :code:`attach_properties==True`.  If None,
                :code:`attach_properties` will attach all linked Properties.
                Note that this only attaches one property per Configuration, so
                if multiple properties point to the same Configuration, that
                Configuration will be returned multiple times.

            attach_properties (bool, default=False):
                If True, attaches all the data of any linked properties from
                :code:`property_hashes`. The property data will either be added to
                the :code:`arrays` dictionary on a Configuration (if it can be
                converted to a matrix where the first dimension is the same
                as the number of atoms in the Configuration) or the :code:`info`
                dictionary (if it wasn't added to :code:`arrays`). Property
                fields in a list to accomodate the possibility of multiple
                properties of the same type pointing to the same configuration.
                WARNING: don't use this option if multiple properties of the
                same type point to the same Configuration, but the properties
                don't have values for all of their fields.

            attach_settings (bool, default=False):
                NOT supported yet. If True, attaches all of the fields of the property settings
                that are linked to the attached property instances. If
                :code:`attach_settings=True`, must also have
                :code:`attach_properties=True`.

            generator (bool, default=False):
                NOT supported yet. If True, this function returns a generator of
                the configurations. This is useful if the configurations can't
                all fit in memory at the same time.

            verbose (bool):
                If True, prints progress bar

        Returns:

            configurations (iterable):
                A list or generator of the re-constructed configurations
        """

        if attach_settings:
            raise NotImplementedError

        if configuration_hashes == 'all':
            query = {'hash': {'$exists': True}}
        else:
            if isinstance(configuration_hashes, str):
                configuration_hashes = [configuration_hashes]

            query = {'hash': {'$in': configuration_hashes}}

        if generator:
            raise NotImplementedError
            # return self._get_configurations(
            #     query=query,
            #     property_ids=property_ids,
            #     attach_properties=attach_properties,
            #     verbose=verbose
            # )
        else:
            return list(self._get_configurations(
                query=query,
                property_hashes=property_hashes,
                attach_properties=attach_properties,
                attach_settings=attach_settings,
                verbose=verbose
            ))

    def _get_configurations(
            self,
            query,
            property_hashes,
            attach_properties,
            attach_settings,
            verbose=False
    ):

        if not attach_properties:
            for co_doc in tqdm(
                    self.configurations.find(
                        query,
                        {
                            *self.configuration_type.unique_identifier_kw,
                            'names',
                            'hash',
                        }
                    ),
                    desc='Getting configurations',
                    disable=not verbose
            ):
                c = self.configuration_type(**{k: v for k, v in co_doc.items()
                                               if k in self.configuration_type.unique_identifier_kw})

                c.info['hash'] = co_doc['hash']
                c.info[ATOMS_NAME_FIELD] = co_doc['names']

                yield c
        else:
            config_dict = {}
            for co_doc in tqdm(
                    self.configurations.find(query),
                    desc='Getting configurations',
                    disable=not verbose
            ):
                c = self.configuration_type(**{k: v for k, v in co_doc.items()
                                               if k in self.configuration_type.unique_identifier_kw})

                c.info['hash'] = co_doc['hash']
                c.info[ATOMS_NAME_FIELD] = co_doc['names']

                config_dict[co_doc['hash']] = c

            all_attached_prs = set([_['hash'] for _ in self.property_instances.find(
                {'relationships.configurations': query['hash']},
                {'hash'}
            )])

            if property_hashes is not None:
                property_hashes = list(all_attached_prs.union(set(property_hashes)))
            else:
                property_hashes = list(all_attached_prs)

            for pr_doc in tqdm(
                    self.property_instances.find({'hash': {'$in': property_hashes}}),
                    desc='Attaching properties',
                    disable=not verbose
            ):

                for co_id in pr_doc['relationships']['configurations']:
                    if co_id not in config_dict: continue

                    c = config_dict[co_id]

                    n = len(c)

                    for field_name, field in pr_doc[pr_doc['type']].items():
                        v = field['source-value']

                        dct = c.info
                        if isinstance(v, list):
                            if len(v) == n:
                                dct = c.arrays

                        field_name = f'{pr_doc["type"]}.{field_name}'

                        if field_name in dct:
                            # Then this is a duplicate property
                            dct[field_name].append(v)
                        else:
                            # Then this is the first time
                            # the property of this type is being added
                            dct[field_name] = [v]

            for v in config_dict.values():
                yield v

            # pipeline = [
            #     {'$match': query},
            #     {'$lookup': {
            #         'from': 'properties',
            #         'localField': 'relationships.properties',
            #         'foreignField': '_id',
            #         'as': 'linked_properties'
            #     }},
            # ]

            # if property_ids is not None:
            #     pipeline.append(
            #         {'$match': {'linked_properties._id': {'$in': property_ids}}}
            #     )

            # if attach_settings:
            #     pipeline.append(
            #         {'$lookup': {
            #             'from': 'property_settings',
            #             'localField': 'linked_properties.relationships.property_settings',
            #             'foreignField': '_id',
            #             'as': 'linked_property_settings'
            #         }}
            #     )

            # for co_doc in tqdm(
            #         self.configurations.aggregate(pipeline),
            #         desc='Getting configurations',
            #         disable=not verbose
            #     ):

            #     c = Configuration(
            #         symbols=co_doc['atomic_numbers'],
            #         positions=co_doc['positions'],
            #         cell=co_doc['cell'],
            #         pbc=co_doc['pbc'],
            #     )

            #     c.info['_id'] = co_doc['_id']
            #     c.info[ATOMS_NAME_FIELD] = co_doc['names']
            #     c.info[ATOMS_LABELS_FIELD] = co_doc['labels']

            #     n = len(c)

            #     for pr_doc in co_doc['linked_properties']:
            #         for field_name, field in pr_doc[pr_doc['type']].items():
            #             v = field['source-value']

            #             dct = c.info
            #             if isinstance(v, list):
            #                 if len(v) == n:
            #                     dct = c.arrays

            #             field_name = f'{pr_doc["type"]}.{field_name}'

            #             if field_name in dct:
            #                 # Then this is a duplicate property
            #                 dct[field_name].append(v)
            #             else:
            #                 # Then this is the first time
            #                 # the property of this type is being added
            #                 dct[field_name] = [v]

            #     if attach_settings:
            #         for ps_doc in co_doc['linked_property_settings']:
            #             for k,v in ps_doc.items():
            #                 if k in [
            #                     '_id', '_description', '_labels', '_method'
            #                     ]:
            #                     c.info['_settings.'+k] = v
            #                 elif k in ['_files', 'last_modified', 'relationships']:
            #                     pass
            #                 else:
            #                     v = field['source-value']

            #                     dct = c.info
            #                     if isinstance(v, list):
            #                         if len(v) == n:
            #                             dct = c.arrays

            #                     dct[k] = v

            #     yield c

    def concatenate_configurations(self):
        """
        Concatenates the atomic_numbers, positions, cells, and pbcs groups in
        /configurations.
        """
        self.database.concatenate_configurations()

    def insert_configuration_set(self, hashes, name, description='', ordered=False, overloaded_cs_id=None,
                                 verbose=False):
        """
        Inserts the configuration set of IDs to the database.

        Args:

            hashes (list or str):
                The hashes of the configurations to include in the configuartion
                set.
            name (str):
                Name of CS---used in forming extended-id
            ordered (bool):
                Flag specifying if COs in CS should be considered ordered.
            overloaded_cs_id (str):
                Used to overload naming convention when updating versions
            description (str, optional):
                A human-readable description of the configuration set.
        """

        if isinstance(hashes, str):
            hashes = [hashes]

        hashes = list(set(hashes))

        # TODO: Look at below
        cs_hash = sha512()
        cs_hash.update(description.encode('utf-8'))
        for i in sorted(hashes):
            cs_hash.update(str(i).encode('utf-8'))

        cs_hash = int(cs_hash.hexdigest(), 16)

        # Check for duplicates
        try:
            return self.configuration_sets.find_one({'hash': str(cs_hash)})[SHORT_ID_STRING_NAME]
        except:
            pass

        if overloaded_cs_id is None:
            cs_id = ID_FORMAT_STRING.format('CS', generate_string(), 0)
        else:
            cs_id = overloaded_cs_id
        # Make sure all of the configurations exist
        #if self.configurations.count_documents({'hash': {'$in': hashes}}) != len(hashes):
        #    raise MissingEntryError(
        #        "Not all of the COs provided to insert_configuration_set exist" \
        #        " in the database."
        #    )

        aggregated_info = self.configuration_type.aggregate_configuration_summaries(
            self,
            hashes,
        )

        self.configuration_sets.update_one(
            {'hash': str(cs_hash)},
            {
                '$setOnInsert': {
                    SHORT_ID_STRING_NAME: cs_id,
                    'name': name,
                    EXTENDED_ID_STRING_NAME: f'{name}__{cs_id}',
                    'description': description,
                    'hash': str(cs_hash),
                    'ordered': ordered
                },
                '$set': {
                    'aggregated_info': aggregated_info,
                    'last_modified': datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
                },
            },
            upsert=True,
            hint='hash',
        )

        # Add the backwards relationships CO->CS
        config_docs = []
        for c_hash in hashes:
            config_docs.append(UpdateOne(
                {'hash': c_hash},
                {
                    '$addToSet': {
                        'relationships.configuration_sets': cs_id
                    }
                },
                hint='hash',
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

        cs_doc = self.configuration_sets.find_one({SHORT_ID_STRING_NAME: cs_id})

        return {
            'last_modified': cs_doc['last_modified'],
            'configuration_set': ConfigurationSet(
                configuration_ids=cs_doc['relationships']['configurations'],
                name=cs_doc['name'],
                description=cs_doc['description'],
                aggregated_info=cs_doc['aggregated_info']
            )
        }

    # TODO look at this function to change updating to involve hash
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

        cs_doc = self.configuration_sets.find_one({SHORT_ID_STRING_NAME: cs_id})

        aggregated_info = self.configuration_type.aggregate_configuration_summaries(
            self,
            cs_doc['relationships']['configurations'],
            verbose=verbose,
        )

        self.configuration_sets.update_one(
            {SHORT_ID_STRING_NAME: cs_id},
            {'$set': {'aggregated_info': aggregated_info}},
            upsert=True,
            hint=SHORT_ID_STRING_NAME,
        )

    # TODO: need to make sure can't make duplicate CS just with different versions
    # TODO: Could do this by creating ConfigurationSets for all versioned CS and use a defined equality with hashing
    def update_configuration_set(self, cs_id, add_ids=None, remove_ids=None):

        if add_ids is None and remove_ids is None:
            raise RuntimeError('Please input configuration IDs to add or remove from the configuration set.')

        # increment version number
        current_hash, current_version = cs_id.split('_')[1:]
        cs_doc = self.configuration_sets.find_one({SHORT_ID_STRING_NAME: {'$eq': cs_id}})
        family_ids = cs_doc[SHORT_ID_STRING_NAME]
        # Remove first -1
        version = int(family_ids.split('_')[-1]) + 1
        new_cs_id = 'CS_' + current_hash + '_' + str(version)
        # Get configuration ids from current version and append and/or remove
        ids = cs_doc['relationships']['configurations']
        ids = [i.split('_')[-1] for i in ids]
        init_len = len(ids)

        if add_ids is not None:
            if isinstance(add_ids, str):
                add_ids = [add_ids]
            ids.extend(add_ids)
            ids = list(set(ids))
            if len(ids) == init_len:
                raise RuntimeError('All configurations to be added are already present in CS.')
            init_len = len(ids)

        if remove_ids is not None:
            if isinstance(remove_ids, str):
                remove_ids = [remove_ids]
            remove_ids = list(set(remove_ids))
            for r in remove_ids:
                try:
                    ids.remove(r)
                except:
                    raise UserWarning(f'A configuration with the ID {r} was not'
                                      f'in the original CS, so it could not be removed.')
            if len(ids) == init_len:
                raise RuntimeError('All configurations to be removed are not present in CS.')

        # insert new version of CS
        self.insert_configuration_set(ids, name=cs_doc['name'], description=cs_doc['description'],
                                      overloaded_cs_id=new_cs_id)
        return new_cs_id

    # TODO: May need to recompute hash-But when is resyncing necessary?
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

        ds_doc = self.datasets.find_one({SHORT_ID_STRING_NAME: ds_id})

        cs_ids = ds_doc['relationships']['configuration_sets']
        pr_ids = ds_doc['relationships']['property_instances']

        for csid in cs_ids:
            self.resync_configuration_set(csid, verbose=verbose)

        aggregated_info = {}

        # Aggregate over DOs
        #for k, v in self.aggregate_configuration_set_info(cs_ids).items():
        #    aggregated_info[k] = v

        for k, v in self.aggregate_data_object_info(pr_ids, verbose=verbose).items():
            if k in {
                'types', 'types_counts',
                'fields', 'fields_counts'
            }:
                k = 'property_' + k

            aggregated_info[k] = v

        self.datasets.update_one(
            {SHORT_ID_STRING_NAME: ds_id},
            {'$set': {'aggregated_info': aggregated_info}},
            upsert=True,
            hint=SHORT_ID_STRING_NAME,
        )

    def aggregate_property_info(self, pr_hashes, verbose=False):
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

        if isinstance(pr_hashes, str):
            pr_hashes = [pr_hashes]

        aggregated_info = {
            'types': [],
            'types_counts': [],
            'fields': [],
            'fields_counts': [],
            # 'methods': [],
            # 'methods_counts': [],
        }

        ignore_keys = {
            'property-id', 'property-title', 'property-description', '_id',
            SHORT_ID_STRING_NAME, 'property-name', 'hash'
        }

        for doc in tqdm(
                self.property_instances.find({'hash': {'$in': pr_hashes}}),
                desc='Aggregating property info',
                disable=not verbose,
                total=len(pr_hashes)
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

            # for l in doc['methods']:
            #    if l not in aggregated_info['methods']:
            #        aggregated_info['methods'].append(l)
            #        aggregated_info['methods_counts'].append(1)
            #    else:
            #        idx = aggregated_info['methods'].index(l)
            #        aggregated_info['methods_counts'][idx] += 1

        return aggregated_info

    def aggregate_data_object_info(self, pr_hashes, verbose=False):
        """
        Aggregates the following information from a list of data_objects:

            * types
            * types_counts

        Args:

            pr_ids (list or str):
                The IDs of the configurations to aggregate information from

            verbose (bool, default=False):
                If True, prints a progress bar

        Returns:

            aggregated_info (dict):
                All of the aggregated info
        """

        if isinstance(pr_hashes, str):
            pr_hashes = [pr_hashes]

        aggregated_info = {
            'property_types': [],
            'property_types_counts': [],
            # 'fields': [],
            # 'fields_counts': [],
            # 'methods': [],
            # 'methods_counts': [],
        }

        ignore_keys = {
            'property-id', 'property-title', 'property-description', '_id',
            SHORT_ID_STRING_NAME, 'property-name', 'hash'
        }

        for doc in tqdm(
                #self.data_objects.find({'hash': {'$in': pr_hashes}}),
                self.query_in_batches(query_key='hash',query_list=pr_hashes,collection_name='data_objects'),
                desc='Aggregating data_object info',
                disable=not verbose,
                total=len(pr_hashes)
        ):
            for i in range(len(doc['property_types'])):
                if doc['property_types'][i] not in aggregated_info['property_types']:
                    aggregated_info['property_types'].append(doc['property_types'][i])
                    aggregated_info['property_types_counts'].append(1)
                else:
                    idx = aggregated_info['property_types'].index(doc['property_types'][i])
                    aggregated_info['property_types_counts'][idx] += 1
        do_ids = ['DO_%s' %i for i in pr_hashes]
        co_ids = list(self.query_in_batches(collection_name='configurations', query_key='relationships.data_objects',
                                       query_list=do_ids, return_key='hash', projection= {'hash':1}))
        ag_2 = self.configuration_type.aggregate_configuration_summaries(self, co_ids,
                                                                         verbose=verbose)
        aggregated_info.update(ag_2)

        return aggregated_info

    '''
            for l in doc[doc['property_types']]:
                if l in ignore_keys: continue

                l = '.'.join([doc['property_types'], l])

                if l not in aggregated_info['fields']:
                    aggregated_info['fields'].append(l)
                    aggregated_info['fields_counts'].append(1)
                else:
                    idx = aggregated_info['fields'].index(l)
                    aggregated_info['fields_counts'][idx] += 1
    '''

    # for l in doc['methods']:
    #    if l not in aggregated_info['methods']:
    #        aggregated_info['methods'].append(l)
    #        aggregated_info['methods_counts'].append(1)
    #    else:
    #        idx = aggregated_info['methods'].index(l)
    #        aggregated_info['methods_counts'][idx] += 1

    # TODO: Make Configuration "type" agnostic (only need to change docstring)
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
            self.configuration_sets.find({SHORT_ID_STRING_NAME: {'$in': cs_ids}})
        )))

        return self.configuration_type.aggregate_configuration_summaries(self, [i.replace('CO_', '') for i in co_ids],
                                                                         verbose=verbose)

    def insert_dataset(
            self, pr_hashes, name,
            authors=None,
            cs_ids=None,
            links=None,
            description='',
            data_license='CC0',
            resync=False,
            verbose=False,
            generator=None,
            overloaded_ds_id=None,
    ):
        """
        Inserts a dataset into the database.

        Args:

            pr_hashes (list or str):
                The hashes of the properties to link to the dataset

            name (str):
                The name of the dataset

            authors (list or str or None):
                The names of the authors of the dataset. If None, then no
                authors are added.


            cs_ids (list or str, default=None):
                The IDs of the configuration sets to link to the dataset.

            links (list or str or None):
                External links (e.g., journal articles, Git repositories, ...)
                to be associated with the dataset. If None, then no links are
                added.

            description (str or None):
                A human-readable description of the dataset. If None, then not
                description is added.

            data_license (str):
                License associated with the Dataset's data

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

        if isinstance(pr_hashes, str):
            pr_hashes = [pr_hashes]

        # Remove possible duplicates
        if cs_ids is not None:
            cs_ids = list(set(cs_ids))
        pr_hashes = list(set(pr_hashes))


        if isinstance(authors, str):
            authors = [authors]

        for auth in authors:
            if not ''.join(auth.split(' ')[-1].replace('-', '')).isalpha():
                raise RuntimeError(
                    "Bad author name '{}'. Author names can only contain [a-z][A-Z]".format(auth)
                )

        if isinstance(links, str):
            links = [links]

        ds_hash = sha512()
        if cs_ids is not None:
            for ci in sorted(cs_ids):
                ds_hash.update(str(ci).encode('utf-8'))
        for pi in sorted(pr_hashes):
            ds_hash.update(str(pi).encode('utf-8'))

        ds_hash = int(ds_hash.hexdigest(), 16)
        # Check for duplicates
        try:
            return self.datasets.find_one({'hash': str(ds_hash)})[SHORT_ID_STRING_NAME]
        except:
            pass

        if overloaded_ds_id is None:
            ds_id = ID_FORMAT_STRING.format('DS', generate_string(), 0)
        else:
            ds_id = overloaded_ds_id

        aggregated_info = {}

        # Aggregate over DOs instead
        #for k, v in self.aggregate_configuration_set_info(
        #        cs_ids, verbose=verbose).items():
        #    aggregated_info[k] = v

        for k, v in self.aggregate_data_object_info(
                pr_hashes, verbose=verbose).items():
            aggregated_info[k] = v

        id_prefix = '_'.join([
            name,
            ''.join([
                unidecode(auth.split()[-1]) for auth in authors
            ]),
        ])

        if len(id_prefix) > (MAX_STRING_LENGTH - len(ds_id) - 2):
            id_prefix = id_prefix[:MAX_STRING_LENGTH - len(ds_id) - 2]
            warnings.warn(f"ID prefix is too long. Clipping to {id_prefix}")
        extended_id = f'{id_prefix}__{ds_id}'

        # TODO: get_dataset should be able to use extended-id; authors can't symbols

        self.datasets.update_one(
            {'hash': str(ds_hash)},
            {
                '$setOnInsert': {
                    SHORT_ID_STRING_NAME: ds_id,
                    EXTENDED_ID_STRING_NAME: extended_id,
                    'name': name,
                    'authors': authors,
                    'links': links,
                    'description': description,
                    'hash': str(ds_hash),
                    'license': data_license,
                },
                '$set': {
                    'aggregated_info': aggregated_info,
                    'last_modified': datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
                },
            },
            upsert=True,
            hint='hash',
        )

        # Add the backwards relationships CS->DS
        if cs_ids is not None:
            config_set_docs = []
            for csid in cs_ids:
                config_set_docs.append(UpdateOne(
                    {SHORT_ID_STRING_NAME: csid},
                    {'$addToSet': {'relationships.datasets': ds_id}},
                    hint=SHORT_ID_STRING_NAME,
                ))

            self.configuration_sets.bulk_write(config_set_docs)

        # Add the backwards relationships PR->DS and origin using aggregation pipeline for logic
        property_docs = []
        for pid in tqdm(pr_hashes, desc='Updating DO->DS relationships'):
            property_docs.append(UpdateOne(
                {'hash': pid},
                [{'$set': {'relationships.origin': {'$ifNull': ['$relationships.origin', ds_id]}
                    , 'relationships.datasets': {
                        '$setUnion': [{'$ifNull': ['$relationships.datasets', []]}, [ds_id]]}}}],
                hint='hash',
            ))
        self.data_objects.bulk_write(property_docs)

        return ds_id

    def get_dataset(self, ds_id, resync=False, verbose=False):
        """
        Returns the dataset with the given ID.

        Args:

            ds_ids (str):
                Either the 'short-id' or 'extended-id' of a dataset

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

        if len(ds_id) > 19:
            # Then this must be an extended ID
            ds_id = ds_id.split('__')[-1]

        if resync:
            self.resync_dataset(ds_id, verbose=verbose)

        ds_doc = self.datasets.find_one({SHORT_ID_STRING_NAME: ds_id})

        return {
            SHORT_ID_STRING_NAME: ds_id,
            'last_modified': ds_doc['last_modified'],
            'dataset': Dataset(
                configuration_set_ids=ds_doc['relationships']['configuration_sets'],
                property_ids=ds_doc['relationships']['data_objects'],
                name=ds_doc['name'],
                authors=ds_doc['authors'],
                links=ds_doc['links'],
                description=ds_doc['description'],
                data_license=ds_doc['license'],
                aggregated_info=ds_doc['aggregated_info']
            )
        }

    # TODO: Handle properties somewhere->should we allow for only properties to be update?
    # TODO: Allow for metadata updating
    def update_dataset(self, ds_id, add_cs_ids=None, remove_cs_ids=None, add_do_ids=None, remove_do_ids=None):

        if add_cs_ids is None and remove_cs_ids is None:
            raise RuntimeError('Please input configuration set IDs/properties to add or remove from the dataset.')

        # increment version number
        current_hash, current_version = ds_id.split('_')[1:]
        ds_doc = self.datasets.find_one({SHORT_ID_STRING_NAME: {'$eq': ds_id}})
        family_ids = ds_doc[SHORT_ID_STRING_NAME]
        version = int(family_ids.split('_')[-1]) + 1
        new_ds_id = 'DS_' + current_hash + '_' + str(version)

        # Get configuration set ids from current version and append and/or remove
        cs_ids = ds_doc['relationships']['configuration_sets']
        do_ids = ds_doc['relationships']['data_objects']
        do_ids = [i.split('_')[-1] for i in do_ids]
        init_len = len(cs_ids)

        if add_cs_ids is not None:
            if isinstance(add_cs_ids, str):
                add_cs_ids = [add_cs_ids]
            cs_ids.extend(add_cs_ids)
            cs_ids = list(set(cs_ids))
            if len(cs_ids) == init_len:
                raise RuntimeError('All configuration sets to be added are already present in DS.')
            init_len = len(cs_ids)

            # Remove old version of CS if new version is in added
            for id in add_cs_ids:
                current_hash, version = id.split('_')[1:]
                if int(version) > 0:
                    try:
                        old_version = self.configuration_sets.find_one(
                            {SHORT_ID_STRING_NAME: {'$regex': f'CS_{current_hash}_.*'}}
                        )
                        cs_ids.remove(old_version)
                    except:
                        pass

        if remove_cs_ids is not None:
            if isinstance(remove_cs_ids, str):
                remove_cs_ids = [remove_cs_ids]
            remove_cs_ids = list(set(remove_cs_ids))
            for r in remove_cs_ids:
                try:
                    cs_ids.remove(r)
                except:
                    raise UserWarning(f'A configuration set with the ID {r} was not'
                                      f'in the original DS, so it could not be removed.')
            if len(cs_ids) == init_len:
                raise RuntimeError('All configuration sets to be removed are not present in DS.')

        init_len_do = len(do_ids)

        if add_do_ids is not None:
            if isinstance(do_ids, str):
                add_do_ids = [add_do_ids]
            do_ids.extend(add_do_ids)
            do_ids = list(set(do_ids))
            if len(do_ids) == init_len_do:
                raise RuntimeError('All data objects to be added are already present in DS.')
        init_len_do = len(do_ids)

        if remove_do_ids is not None:
            if isinstance(remove_do_ids, str):
                remove_do_ids = [remove_do_ids]
            remove_do_ids = list(set(remove_do_ids))
            for r in remove_do_ids:
                try:
                    do_ids.remove(r)
                except:
                    raise UserWarning(f'A data object with the ID {r} was not'
                                      f'in the original DS, so it could not be removed.')
            if len(do_ids) == init_len_do:
                raise RuntimeError('All data objects to be removed are not present in DS.')
        # insert new version of DS

        self.insert_dataset(cs_ids, do_ids, name=ds_doc['name'], authors=ds_doc['authors'],
                            links=ds_doc['links'], description=ds_doc['description'], overloaded_ds_id=new_ds_id)
        return (new_ds_id)

    def aggregate_dataset_info(self, ds_ids):
        """
        Aggregates information from a list of datasets.

        NOTE: this will face all of the same challenges as
        aggregate_configuration_set_info()

            * you need to find the overlap of COs and PRs.
        """
        pass

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

        Returns:
            Returns the figure object.
        """
        if fields is None:
            fields = self.property_fields
        elif isinstance(fields, str):
            fields = [fields]

        nfields = len(fields)

        nrows = max(1, int(np.ceil(nfields / 3)))
        if (nrows > 1) or (nfields % 3 == 0):
            ncols = 3
        else:
            ncols = nfields % 3

        if method == 'plotly':
            fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=fields)
        elif method == 'matplotlib':
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 2 * nrows))
            axes = np.atleast_2d(axes)
        else:
            raise RuntimeError('Unsupported plotting method')

        for i, prop in enumerate(fields):
            data = self.get_data(
                _PROPS_COLLECTION,
                prop,
                query=query,
                hashes=ids,
                verbose=verbose,
                ravel=True
            )

            c = i % 3
            r = i // 3

            if nrows > 1:

                if method == 'plotly':
                    fig.add_trace(
                        go.Histogram(x=data, nbinsx=nbins, name=prop),
                        row=r + 1, col=c + 1,
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
                        row=1, col=c + 1,
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
                query={SHORT_ID_STRING_NAME: {'$in': <list_of_property_IDs>}},
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
                'properties', field, query=query, hashes=ids,
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
                the key-value pair :code:`{SHORT_ID_STRING_NAME: {'$in': ...}}` will be
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

        ds_doc = self.datasets.find_one({SHORT_ID_STRING_NAME: ds_id})

        configuration_sets = []
        property_ids = []

        # Loop over configuration sets
        cursor = self.configuration_sets.find({
            SHORT_ID_STRING_NAME: {'$in': ds_doc['relationships']['configuration_sets']}
        }
        )

        for cs_doc in tqdm(
                cursor,
                desc='Filtering on configuration sets',
                disable=not verbose
        ):
            query[SHORT_ID_STRING_NAME] = {'$in': cs_doc['relationships']['configurations']}

            co_hashes = self.get_data('configurations', fields='hash', query=query)

            # Build the filtered configuration sets
            configuration_sets.append(
                ConfigurationSet(
                    configuration_ids=co_hashes,
                    description=cs_doc['description'],
                    aggregated_info=self.configuration_type.aggregate_configuration_summaries(
                        self,
                        co_hashes,
                        verbose=verbose,
                    )
                )
            )

        # Now get the corresponding properties
        # TODO: Eric->Check correctness here
        property_hashes = [_['hash'] for _ in self.property_instances.filter(
            {
                'hash': {'$in': list(itertools.chain.from_iterable(
                    cs.configuration_ids for cs in configuration_sets
                ))}
            },
            {'hash': 1}
        )]

        return configuration_sets, property_hashes

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

        ds_doc = self.datasets.find_one({SHORT_ID_STRING_NAME: ds_id})

        configuration_sets = []
        property_hashes = []

        # Filter the properties
        retfields = {'hash': 1, 'relationships.configurations': 1}
        if fields is not None:
            if isinstance(fields, str):
                fields = [fields]

            for f in fields:
                retfields[f] = 1

        if query is None:
            query = {}

        query['hash'] = {'$in': ds_doc['relationships']['property_instances']}

        cursor = self.property_instances.find(query, retfields)

        all_co_hashes = []
        for pr_doc in tqdm(
                cursor,
                desc='Filtering on properties',
                disable=not verbose,
        ):
            if filter_fxn(pr_doc):
                property_hashes.append(pr_doc['hash'])
                all_co_hashes.append(pr_doc['relationships']['configurations'])

        all_co_hashes = list(set(itertools.chain.from_iterable(all_co_hashes)))

        # Then filter the configuration sets
        for cs_doc in self.configuration_sets.find({
            SHORT_ID_STRING_NAME: {'$in': ds_doc['relationships']['configuration_sets']}
        }):
            co_hashes = list(
                set(cs_doc['relationships']['configurations']).intersection(
                    all_co_hashes
                )
            )

            configuration_sets.append(
                ConfigurationSet(
                    configuration_ids=co_hashes,
                    description=cs_doc['description'],
                    aggregated_info=self.configuration_type.aggregate_configuration_summaries(
                        self,
                        co_hashes,
                        verbose=verbose
                    )
                )
            )

        return configuration_sets, property_hashes

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

    '''
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
            query['hash'] = {'$in': all_co_ids}

            co_ids = self.get_data(
                'configurations',
                fields='hash',
                query=query,
                ravel=True
            ).tolist()
            # TODO: Eric ->add name below
            cs_id = self.insert_configuration_set(
                co_ids,
                name='From-Markdown',
                description=row[header.index('Description')],
                verbose=True
            )

            cs_ids.append(cs_id)

        # Define the Dataset
        ds_id = self.insert_dataset(
            cs_ids=cs_ids,
            pr_hashes=all_pr_ids,
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
            query['hash'] = {'$in': all_co_ids}

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

    '''

    '''
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
        for pr_doc in self.property_instances.find(
            {'hash': {'$in': dataset.property_ids}}
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

        # TODO: property settings should populate the MD file table

        # property_settings = {}
        # for pso_doc in self.property_settings.find(
        #     {'relationships.properties': {'$in': dataset.property_ids}}
        #     ):
        #     property_settings[pso_doc['_id']] = self.get_property_settings(
        #         pso_doc['_id']
        #     )

        property_settings = list(
            self.property_settings.find(
                {'relationships.property_instances': {'$in': dataset.property_ids}}
            )
        )

        # Build Property Settings table

        ps_table_lines = {}
        for ps_doc in property_settings:
            ps_tup = ('settings', )
            ps_table.append('| {} | {} | {} | {} | {} |'.format(
                pso_id,
                pso.method,
                pso.description,
                ', '.join(pso.labels),
                ', '.join('[{}]({})'.format(f, f) for f in pso.files)
            ))


        for pr_doc in self.property_instances.aggregate([
                {'$match': {'$in': dataset.property_ids}},
                {'$lookup': {
                    'from': 'property_settings',
                    'localField': 'relationships.property_settings',
                    'foreignField': '_id',
                    'as': 'linked_settings'
                }},
            ]):

            for ps_doc in pr_doc['linked_settings']:
                pass

        # TODO: get the types of the properties linked to the PSs

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
                configuration_hashes=list(set(itertools.chain.from_iterable(
                    cs.configuration_ids for cs in configuration_sets.values()
                ))),
                attach_settings=True,
                attach_properties=True,
                generator=True,
            )

            ase_write(
                data_file_name,
                images=images,
                format=data_format,
            )
    '''

    '''
    def export_dataset(self, ds_id, output_folder, fmt, mode, verbose=False):
        """
        Exports the dataset whose :code:`SHORT_ID_STRING_NAME` matches :code:`ds_id` to
        the given format.

        Args:

            ds_id (str):
                An ID matching the form DS_XXXXXXXXXXXX_XXX

            output_folder (str):
                The path to a folder in which to save the dataset. Database
                contents will be save under
                :code:`<output_folder>/database.<fmt>`, and all other files
                (property settings files, property definitions, etc.) will be
                saved as :code:`<output_folder>/<file_name>`.

            fmt (str):
                The format to which to export the data. Supported formats:
                ['hdf5'].

            mode (str):
                'r', 'w', or 'a'

            verbose (bool, default=True):
                If True, prints progress bar
        """

        # Check if folders exist
        path = os.path.join(output_folder)
        if not os.path.isdir(path):
            os.mkdir(path)

        path = os.path.join(output_folder, 'property_definitions')
        if not os.path.isdir(path):
            os.mkdir(path)

        path = os.path.join(output_folder, 'property_settings_files')
        if not os.path.isdir(path):
            os.mkdir(path)

        supported_formats = ['hdf5']
        if fmt not in supported_formats:
            raise RuntimeError(
                f"The only supported formats are {supported_formats}"
            )

        ds_doc = self.datasets.find_one({SHORT_ID_STRING_NAME: ds_id})

        configuration_ids = []

        for cs_id in ds_doc['relationships']['configuration_sets']:
            configuration_ids += self.get_configuration_set(
                cs_id
            )['configuration_set'].configuration_ids

        property_ids = ds_doc['relationships']['property_instances']

        # Write the property definitions to files
        prop_definitions = {}
        for pd_name in ds_doc['aggregated_info']['property_types']:
            pd_doc = self.get_property_definition(pd_name)['definition']

            pd_path = os.path.join(
                output_folder, 'property_definitions', f'{pd_name}.json'
            )

            prop_definitions[pd_name] = pd_doc

            with open(pd_path, 'w') as pd_file:
                json.dump(pd_doc, pd_file, indent=4)

        if fmt == 'hdf5':

            hdf5_path = os.path.join(output_folder, f'{ds_id}.hdf5')
            with h5py.File(hdf5_path, mode) as outfile:
                # Build all groups

                pi_coll_group = outfile.create_group(_PROPS_COLLECTION)
                ps_coll_group = outfile.create_group(_PROPSETTINGS_COLLECTION)
                co_coll_group = outfile.create_group(_CONFIGS_COLLECTION)
                cs_coll_group = outfile.create_group(_CONFIGSETS_COLLECTION)

                # Write dataset info
                outfile.attrs['description'] = ds_doc['description']

                outfile.attrs.create(
                    'authors',
                    np.array(ds_doc['authors'], dtype=STRING_DTYPE_SPECIFIER)
                )

                outfile.attrs.create(
                    'links',
                    np.array(ds_doc['links'], dtype=STRING_DTYPE_SPECIFIER)
                )

                # TODO: decide if you want to export aggregated info too
                # info_group = outfile.create_group('aggregated_info')

                # for k,v in ds_doc['aggregated_info'].items():
                #     info_group.create_dataset(k, data=v)

                # Write the configurations
                for co_doc in self.configurations.find(
                        {'hash': {'$in': configuration_ids}}
                    ):

                    co_group = co_coll_group.create_group(co_doc['hash'])

                    co_group.create_dataset(
                        'names',
                        data=np.array(co_doc['names'], dtype=STRING_DTYPE_SPECIFIER)
                    )

                    co_group.create_dataset(
                        'labels',
                        data=np.array(co_doc['labels'], dtype=STRING_DTYPE_SPECIFIER)
                    )

                    co_group.create_dataset(
                        'relationships.property_instances',
                        data=np.array(co_doc['relationships']['property_instances'], dtype=STRING_DTYPE_SPECIFIER)
                    )

                    co_group.create_dataset(
                        'relationships.configuration_sets',
                        data=np.array(co_doc['relationships']['configuration_sets'], dtype=STRING_DTYPE_SPECIFIER)
                    )

                    for key in self.configuration_type.unique_identifier_kw:
                        co_group.create_dataset(
                            key,
                            dtype=self.configuration_type.unique_identifier_kw_types[key],
                            data=co_doc[key]
                        )

                # Write property instances
                ps_ids = []
                for pi_doc in self.property_instances.find(
                        {'hash': {'$in': property_ids}}
                    ):
                    pi_group = pi_coll_group.create_group(pi_doc['hash'])

                    pi_group.create_dataset(
                        'type',
                        data=np.array(pi_doc['type'],
                        dtype=STRING_DTYPE_SPECIFIER),
                    )

                    pi_group.create_dataset(
                        'methods',
                        data=np.array(pi_doc['methods'],
                        dtype=STRING_DTYPE_SPECIFIER),
                    )

                    pi_group.create_dataset(
                        'labels',
                        data=np.array(pi_doc['labels'],
                        dtype=STRING_DTYPE_SPECIFIER),
                    )

                    pi_group.create_dataset(
                        'relationships.configurations',
                        data=np.array(pi_doc['relationships']['configurations'],
                        dtype=STRING_DTYPE_SPECIFIER),
                    )

                    pi_group.create_dataset(
                        'relationships.property_settings',
                        data=np.array(pi_doc['relationships']['property_settings'],
                        dtype=STRING_DTYPE_SPECIFIER),
                    )

                    ps_ids += pi_doc['relationships']['property_settings']

                    data_group = pi_group.create_group(pi_doc['type'])

                    for key, value in pi_doc[pi_doc['type']].items():
                        dtype = prop_definitions[pi_doc['type']][key]['type']
                        if dtype == 'string':
                            dtype = STRING_DTYPE_SPECIFIER

                        data_group.create_dataset(
                            key,
                            dtype=dtype,
                            data=value['source-value'],
                        )

                # Write property settings
                ps_ids = list(set(ps_ids))
                for ps_doc in self.property_settings.find(
                    {'hash': {'$in': ps_ids}}
                    ):

                    ps_group = ps_coll_group.create_group(ps_doc['hash'])

                    ps_group.attrs['description'] = ps_doc['description']
                    ps_group.attrs['method'] = ps_doc['method']
                    ps_group.attrs.create(
                        'labels',
                        np.array(ps_doc['labels'], dtype=STRING_DTYPE_SPECIFIER)
                    )

                    for fname, fcontents in ps_doc['files']:
                        with open(
                                os.path.join(
                                    output_folder, 'property_settings_files', fname
                                ),
                                'w'
                            ) as fpointer:

                            fpointer.write(fcontents)

                # Write configuration sets
                for cs_doc in self.configuration_sets.find(
                        {SHORT_ID_STRING_NAME: {
                                '$in':
                                ds_doc['relationships']['configuration_sets']
                            }
                        },
                    ):

                    cs_group = cs_coll_group.create_group(cs_doc[SHORT_ID_STRING_NAME])

                    cs_group.attrs['description'] = cs_doc['description']
                    cs_group.create_dataset(
                        'relationships.configurations',
                        data=np.array(cs_doc['relationships']['configurations'],
                        dtype=STRING_DTYPE_SPECIFIER),
                    )
    '''

    def export_ds_to_xyz(self, ds_id, nprocs=1):
        ds_doc = self.datasets.find_one({SHORT_ID_STRING_NAME: {'$eq': ds_id}})
        cas = ds_doc['relationships']['data_objects']
        cos = list(
            self.configurations.find({'relationships.data_objects': {'$in': cas}}).sort('relationships.data_objects',
                                                                                        1))
        pis = list(self.property_instances.find({'relationships.data_objects': {'$in': cas}}).sort(
            'relationships.data_objects', 1))
        p = multiprocessing.Pool(nprocs)
        results = []
        # results.append(result for result in tqdm(p.imap_unordered(partial(build_do,client_name=self.database_name),cas)))
        for result in tqdm(p.imap_unordered(partial(build_do, client_name=self.database_name), cas), total=len(cas)):
            results.append(result)
        p.close()
        p.join()
        ase_write('%s.xyz' % ds_doc['extended-id'], results, tolerant=True)


def build_do(ca, client_name):
    client = MongoDatabase(client_name)
    co = client.configurations.find_one({'relationships.data_objects': {'$in': [ca]}})
    pis = list(client.property_instances.find({'relationships.data_objects': {'$in': [ca]}}))
    a = Atoms(numbers=co['atomic_numbers'], positions=co['positions'], cell=co['cell'], pbc=co['pbc'])
    a.info['do-colabfit-id'] = co['relationships']['data_objects']
    for i, pi in enumerate(pis):
        # print ('pi_type',pi['type'])
        for k, v in pi.items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    if isinstance(v2, dict):
                        if 'source-value' in pi[k][k2]:
                            # hardcode whether property goes in info or arrays for now
                            if k2 == 'forces':
                                a.arrays['forces'] = np.array(pi[k][k2]['source-value'])
                            else:
                                if k2 == 'stress':  # Needed so that ASE formats properly
                                    a.info['stress'] = np.array(pi[k][k2]['source-value'])
                                else:
                                    a.info['%s.%s' % (k, k2)] = pi[k][k2]['source-value']
    return a


# TODO: Change labels_field to metadata_fields
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
            A list of strings of allowed element types. If None, all element
            types are allowed.

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
            generate a list of files to be passed to `self.reader`.

        generator (bool, default=True):
            If True, returns a generator of Configurations. If False, returns a
            list.

        verbose (bool):
            If True, prints progress bar.

    All other keyword arguments will be passed with
    `converter.load(..., **kwargs)`
    """

    if elements is None:
        elements = [e.symbol for e in periodictable.elements]
        elements.remove('n')

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
            glob_string=glob_string,
            verbose=verbose,
        )
    else:
        raise RuntimeError(
            "Invalid `file_format`. Must be one of " \
            "['xyz', 'extxyz', 'cfg', 'folder']"
        )

    return results if generator else list(results)


# Moved out of static method to avoid changing insert_data* methods
# Could consider changing in the future
def _build_c_update_doc(configuration):
    processed_fields = configuration.configuration_summary()
    c_hash = str(hash(configuration))
    c_update_doc = {
        '$setOnInsert': {
            'hash': c_hash,
            SHORT_ID_STRING_NAME: 'CO_' + c_hash
        },
        '$set': {
            'last_modified': datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
        },
        '$addToSet': {
            'names': {
                '$each': list(configuration.info[ATOMS_NAME_FIELD])
            },

        }
    }
    c_update_doc['$setOnInsert'].update({k: v.tolist() for k, v in configuration.unique_identifiers.items()})
    c_update_doc['$setOnInsert'].update({k: v for k, v in processed_fields.items()})
    return c_update_doc, c_hash


def _build_md_insert_doc(metadata):
    md_set_on_insert = {
        'hash': str(metadata._hash),
        SHORT_ID_STRING_NAME: 'MD_' + str(metadata._hash)
    }

    for gf, gf_dict in metadata.metadata.items():
        if isinstance(gf_dict['source-value'], (int, float, str)):
            # Add directly
            md_set_on_insert[gf] = {
                'source-value': gf_dict['source-value']
            }
        else:
            # Then it's array-like and should be converted to a list
            md_set_on_insert[gf] = {
                'source-value': np.atleast_1d(
                    gf_dict['source-value']
                ).tolist()
            }

        if 'source-unit' in gf_dict:
            md_set_on_insert[gf]['source-unit'] = gf_dict['source-unit']
    return md_set_on_insert


def _build_ca_insert_doc(calculation):
    ca_set_on_insert = {
        'hash': str(calculation._hash),
        SHORT_ID_STRING_NAME: 'DO_' + str(calculation._hash),
        #'relationships.configurations': 'CO_' + calculation.configuration,
        # '$addToSet': {'relationships.property_instances':  {'$each': calculation.properties}},
        # 'ncounts': 0,

    }
    return ca_set_on_insert


def generate_string():
    return get_random_string(12, allowed_chars=string.ascii_lowercase + '1234567890')


class ConcatenationException(Exception):
    pass


class InvalidGroupError(Exception):
    pass


class MissingEntryError(Exception):
    pass


class DuplicateDefinitionError(Exception):
    pass
