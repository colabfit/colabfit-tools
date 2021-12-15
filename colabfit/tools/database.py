import datetime
import warnings
import itertools
import numpy as np
from tqdm import tqdm
from getpass import getpass
from pymongo import MongoClient
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

from colabfit import (
    _CONFIGS_COLLECTION, _PROPS_COLLECTION, _PROPSETTINGS_COLLECTION,
    _CONFIGSETS_COLLECTION, _PROPDEFS_COLLECTION, _DATASETS_COLLECTION,
    ATOMS_NAME_FIELD, ATOMS_LABELS_FIELD, ATOMS_LAST_MODIFIED_FIELD
)
from colabfit.tools.configuration import Configuration, process_species_list
from colabfit.tools.property import Property
from colabfit.tools.configuration_set import ConfigurationSet
from colabfit.tools.converters import CFGConverter, EXYZConverter, FolderConverter
from colabfit.tools.dataset import Dataset

class MongoDatabase(MongoClient):
    """
    A MongoDatabase stores all of the data in Mongo documents, and
    provides additinal functionality like filtering and optimized queries.

    The Mongo database has the following structure

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

    /properties
        /property_name_1
            _id
            each field in the property definition
            last_modified
            aggregated_info
                (from property settings)
                labels
                labels_counts
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
                types
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
    def __init__(self, database_name, drop=False):
        """
        Args:

            database_name (str):
                The name of the database

            drop (bool, default=False):
                If True, deletes the existing Mongo database.

        """
        # super().__init__(repository=client_repo, **kwargs)
        # user = input("mongodb username: ")
        # pwrd = getpass("mongodb password: ")
        user = 'colabfitAdmin'
        pwrd = 'Fo08w3K&VEY&'

        super().__init__(
            'mongodb://{}:{}@localhost:27017/'.format(user, pwrd)
        )

        self.database_name = database_name

        if drop:
            self.drop_database(database_name)

        self.configurations         = self[database_name][_CONFIGS_COLLECTION]
        self.properties             = self[database_name][_PROPS_COLLECTION]
        self.property_definitions   = self[database_name][_PROPDEFS_COLLECTION]
        self.property_settings      = self[database_name][_PROPSETTINGS_COLLECTION]
        self.configuration_sets     = self[database_name][_CONFIGSETS_COLLECTION]
        self.datasets               = self[database_name][_DATASETS_COLLECTION]


    def insert_data(
        self, configurations, property_map=None, property_settings=None,
        generator=False, verbose=True
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

                If None, only loads the configuration information (atomic
                numbers, positions, lattice vectors, and periodic boundary
                conditions).

            property_settings (dict)
                key = property name (same as top-level keys in property_map).
                val = property settings ID that has been previously entered into
                the database using
                :meth:`~colabfit.tools.database.Database.insert_property_settings`

            generator (bool, default=False):
                If True, returns a generator of the results; otherwise returns
                a list.

            verbose (bool, default=False):
                If True, prints a progress bar

        Returns:

            ids (list):
                A list of (config_id, property_id) tuples of the inserted data.
                If no properties were inserted, then property_id will be None.

        """

        if generator:
            return self._insert_data(
                configurations=configurations,
                property_map=property_map,
                property_settings=property_settings,
                verbose=verbose
            )
        else:
            return list(self._insert_data(
                configurations=configurations,
                property_map=property_map,
                property_settings=property_settings,
                verbose=verbose
            ))


    def _insert_data(
        self, configurations, property_map=None, property_settings=None,
        verbose=True
        ):

        if isinstance(configurations, Configuration):
            configurations = [configurations]

        if property_map is None:
            property_map = {}

        if property_settings is None:
            property_settings = {}

        for settings_id in property_settings.values():
            if not self.property_settings.count_documents({'_id': settings_id}):
                raise MissingEntryError(
                    "The property settings object with ID '{}' does"\
                    " not exist in the database".format(settings_id)
                )

        property_definitions = {
            pname: self.get_property_definition(pname)['definition']
            for pname in property_map
        }

        ignore_keys = {
            'property-id', 'property-title', 'property-description',
            'last_modified', 'definition'
        }

        expected_keys = {
            pname: set(
                property_map[pname][f]['field']
                for f in property_definitions[pname].keys() - ignore_keys
                # property_definitions[pname].keys()
            )
            for pname in property_map
        }

        # Add all of the configurations into the Mongo server
        ai = 1
        for atoms in tqdm(
            configurations,
            desc='Adding configurations to Database',
            disable=not verbose,
            ):

            cid = str(hash(atoms))

            processed_fields = process_species_list(atoms)

            # Add if doesn't exist, else update (since last-modified changed)

            self.configurations.update_one(
                {'_id': cid},  # filter
                {  # update document
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
                    }
                },
                upsert=True,  # overwrite if exists already
            )

            available_keys = set().union(atoms.info.keys(), atoms.arrays.keys())

            pid = None

            for pname, pmap in property_map.items():

                # Pre-check to avoid having to delete partially-added properties
                missing_keys = expected_keys[pname] - available_keys
                if missing_keys:
                    warnings.warn(
                        "Configuration {} is missing keys {} during "\
                        "insert_data. Available keys: {}. Skipping".format(
                            ai, missing_keys, available_keys
                        )
                    )
                    continue

                prop = Property.from_definition(
                    pname, property_definitions[pname],
                    atoms, pmap
                )

                pid = str(hash(prop))

                # Attach property settings, if any were given
                settings_id = property_settings[pname] if pname in property_settings else None

                if settings_id:
                    self.property_settings.update_one(
                        {'_id': settings_id},
                        {'$addToSet': {'relationships.properties': pid}}
                    )

            # TODO: resync PSO labels to PR
                setOnInsert = {}
                for k in property_map[pname]:
                    setOnInsert[k] = {
                        'source-value': np.atleast_1d(
                            prop[k]['source-value']
                        ).tolist()
                    }

                    if 'source-unit' in prop[k]:
                        setOnInsert[k]['source-unit'] = prop[k]['source-unit']

                setOnInsert['_id'] = pid

                self.properties[pname].update_one(
                    {'_id': pid},
                    {
                        '$addToSet': {
                            # 'labels': {'$each': labels},
                            # PR -> PSO pointer
                            'relationships.property_settings': {
                                # hack for handling possibly empty case
                                '$each': [settings_id] if settings_id  else []
                            },
                            'relationships.configurations': cid,
                        },
                        '$setOnInsert': setOnInsert,
                        '$set': {
                            'last_modified': datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
                        }
                    },
                    upsert=True
                )

                # Add the backwards arrow
                self.configurations.update_one(
                    {'_id': cid},
                    {'$addToSet': {'relationships.properties': pid}},
                    upsert=True
                )

                yield (cid, pid)

            if not pid:
                # Only yield if something wasn't yielded earlier
                yield (cid, pid)

            ai += 1


    def insert_property_definition(self, definition):
        """
        Inserts a new property definition into the database. Checks that
        definition is valid, then builds all necessary groups in
        :code:`/root/properties`.

        Args:

            definition (dict):
                The map defining the property. See the example below, or the
                `OpenKIM Properties Framework <https://openkim.org/doc/schema/properties-framework/>`_
                for more details.

        Example definition:

        ..code-block:: python

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

        self.property_definitions.update_one(
            {'_id': definition['property-id']},
            {
                '$setOnInsert': {
                    '_id': definition['property-id'],
                    'definition': definition
                }
            },
            upsert=True
        )


    def get_property_definition(self, name):
        return next(self.property_definitions.find({'_id': name}))


    def insert_property_settings(self, pso_object):
        """
        Inserts a new property settings object into the database by creating
        and populating the necessary groups in :code:`/root/property_settings`.

        Args:

            pso_object (PropertySettings)
                The :class:`~colabfit.tools.property_settings.PropertySettings`
                object to insert into the database.


        Returns:

            pso_id (str):
                The ID of the inserted property settings object. Equals the hash
                of the object.
        """

        pso_id = str(hash(pso_object))

        self.property_settings.update_one(
            {'_id': pso_id},
            {
                '$addToSet': {
                    'labels': {'$each': list(pso_object.labels)}
                },
                '$setOnInsert': {
                    '_id': pso_id,
                    'method': pso_object.method,
                    'description': pso_object.description,
                    'files': [
                        {
                            'file_name': ftup[0],
                            'file_contents': ftup[1],
                        } for ftup in pso_object.files
                    ],
                }
            },
            upsert=True
        )

        return pso_id


    def get_property_settings(self, pso_id):
        return next(self.property_settings.find({'_id': pso_id}))


    def get_data(
        self, collection_name, keys,
        ids=None,
        concatenate=False, ravel=False
        ):
        """
        Queries the database and returns the fields specified by `keys`. Returns
        the results in memory.

        Example:

        ..code-block:: python

            database.get_data(
                collection_name='properties/property_name_1',
                keys=['energy', 'forces'],
            )

        Args:

            collection_name (str):
                The name of a collection in the database. Can include nested
                collections using path-like strings (e.g.,
                'properties/property_name_1').

            keys (list or str):
                A keys for indexing the returned objects from
                the Mongo cursor.

            ids (list):
                The list of IDs to return the data for. If None, returns the
                data for the entire collection.

            concatenate (bool):
                If True, concatenates the data before returning. Only available
                if :code:`in_memory==True`.

            ravel (bool):
                If True, concatenates and ravels the data before returning. Only
                available if :code:`in_memory==True`.

        Returns:

            data (dict):
                key = k for k in keys. val = in-memory data
        """
        if ids is None:
            query = {}
        else:
            if isinstance(ids, str):
                ids = [ids]

            query = {'_id': {'$in': ids}}

        if isinstance(keys, str):
            keys = [keys]

        collection = self[self.database_name]
        for cname in collection_name.split('/'):
            collection = collection[cname]

        cursor = collection.find(query, {k: 1 for k in keys})

        data = {k: [] for k in keys}

        for doc in cursor:
            for k in keys:
                if isinstance(doc[k], dict):
                    data[k].append(doc[k]['source-value'])
                else:
                    data[k].append(doc[k])

        if concatenate or ravel:
            for k,v in data.items():
                data[k] = np.concatenate(v)

        if ravel:
            for k,v in data.items():
                data[k] = v.ravel()

        if len(data) == 1:
            return data[keys[0]]
        else:
            return data


    def get_configuration(self, i, verbose=False):
        """
        Returns a single configuration by calling :meth:`get_configurations`
        """
        return self.get_configurations([i])


    def get_configurations(self, ids, generator=False, verbose=False):
        """
        A generator that returns in-memory Configuration objects one at a time
        by loading the atomic numbers, positions, cells, and PBCs.

        Args:

            ids (list or 'all'):
                A list of string IDs specifying which Configurations to return.
                If 'all', returns all of the configurations in the database.

            generator (bool):
                If true, this function becomes a generator which only returns
                the configurations one at a time. This is useful if the
                configurations can't all fit in memory at the same time. Default
                is False.

            verbose (bool):
                If True, prints progress bar

        Returns:

            configurations (iterable):
                A list or generator of the re-constructed configurations
        """

        return (
            Configuration(
                symbols=co_doc['atomic_numbers'],
                positions=co_doc['positions'],
                cell=co_doc['cell'],
                pbc=co_doc['pbc'],
            ) for co_doc in self.configurations.find({'_id': {'$in': ids}})
        )



    def concatenate_configurations(self):
        """
        Concatenates the atomic_numbers, positions, cells, and pbcs groups in
        /configurations.
        """
        self.database.concatenate_configurations()


    def insert_configuration_set(self, ids, description=''):
        """
        Inserts the configuration set of IDs to the database.

        Args:

            ids (list or str):
                The IDs of the configurations to include in the configuartion
                set.

            description (str, optional):
                A human-readable description of the configuration set.
        """

        cs_id = self.database.insert_configuration_set(
            ids=ids, description=description
        )

        aggregated_info = self.aggregate_configuration_info(ids)

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
                    'last_modified': self.database[f'configuration_sets/{cs_id}'].attrs['last_modified']
                },
            },
            upsert=True
        )

        # Add the backwards relationships CO->CS
        for cid in ids:
            self.configurations.update_one(
                {'_id': cid},
                {
                    '$addToSet': {
                        'relationships.configuration_sets': cs_id
                    }
                }
            )

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

        cs_doc = next(self.configuration_sets.find({'_id': cs_id}))

        return {
            'last_modified': cs_doc['last_modified'],
            'configuration_set': ConfigurationSet(
                configuration_ids=cs_doc['relationships']['configurations'],
                description=cs_doc['description'],
                aggregated_info=cs_doc['aggregated_info']
            )
        }


    def resync_configuration_set(self, cs_id):
        """
        Re-synchronizes the configuration set by re-aggregating the information
        from the configurations.

        Args:

            cs_id (str):
                The ID of the configuration set to update

        Returns:

            None; updates the configuration set document in-place

        """

        co_ids = list(self.database[f'configuration_sets/{cs_id}'].attrs['configuration_ids'])

        aggregated_info = self.aggregate_configuration_info(co_ids)

        self.configuration_sets.update_one(
            {'_id': cs_id},
            {'$set': {'aggregated_info': aggregated_info}}
        )


    def resync_property(self, pid):
        """
        Re-synchronizes the property by pulling up labels from any attached
        property settings.

        Args:

            pid (str):
                The ID of the property to update

        Returns:

            None; updates the property document in-place
        """
        pso_ids = self.database.get_data(
            'properties/settings_ids', ids=pid, as_str=True,
            in_memory=True, ravel=True
        ).tolist()

        aggregated_info = self.aggregate_property_settings_info(pso_ids)

        self.properties.update_one(
            {'_id': pid},
            {'$set': {'aggregated_info': aggregated_info}}
        )


    def resync_dataset(self, ds_id):
        """
        Re-synchronizes the dataset by aggregating all necessary data from
        properties and configuration sets. Note that this also calls
        :meth:`colabfit.tools.client.resync_configuration_set` and
        :meth:`colabfit.tools.client.resync_property` for each attached object.

        Args:

            ds_id (str):
                The ID of the dataset to update

        Returns:

            None; updates the dataset document in-place
        """

        cs_ids = self.database[f'datasets/{ds_id}'].attrs['configuration_set_ids'].tolist()
        pr_ids = self.database[f'datasets/{ds_id}'].attrs['property_ids'].tolist()

        for csid in cs_ids:
            self.resync_configuration_set(csid)

        for pid in pr_ids:
            self.resync_property(pid)

        aggregated_info = {}
        for k,v in self.aggregate_configuration_set_info(cs_ids).items():
            if k == 'labels':
                k = 'configuration_labels'
            elif k == 'labels_counts':
                k = 'configuration_labels_counts'

            aggregated_info[k] = v

        for k,v in self.aggregate_property_info(pr_ids).items():
            if k == 'labels':
                k = 'property_labels'
            elif k == 'labels_counts':
                k = 'property_labels_counts'

            aggregated_info[k] = v

        self.datasets.update_one(
            {'_id': ds_id},
            {'$set': {'aggregated_info': aggregated_info}}
        )


    def aggregate_configuration_info(self, ids):
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
        """

        aggregated_info = {
            'nconfigurations': len(ids),
            'nsites': 0,
            'nelements': 0,
            'elements': [],
            'individual_elements_ratios': [],
            'total_elements_ratios': [],
            'labels': [],
            'labels_counts': [],
            'chemical_formula_reduced': set(),
            'chemical_formula_anonymous': set(),
            'chemical_formula_hill': set(),
            'nperiodic_dimensions': set(),
            'dimension_types': set(),
        }

        # TODO: I could convert this to only using HDF5 operations instead.

        # aggregated_info['nconfigurations'] = len(ids)

        # atomic_numbers = self.database.get_data(
        #     'configurations/atomic_numbers',
        #     ids=ids
        # )

        # for cid in ids:
        #     sl = self.database[f'configurations/atomic_numbers/slices/{cid}'][()]

        #     aggregated_info['nsites'] += atomic_numbers[sl].shape[0]
        #     aggregated_info['elements'] = aggregated_info.union(set(np.unique(
        #         atomic_numbers[sl]
        #     )))

        # aggregated_info['nelements'] = len(aggregated_info['elements'])

        # elements = set().union(*[np.unique(a) for a in self.database.])

        for doc in self.configurations.find({'_id': {'$in': ids}}):
            aggregated_info['nsites'] += doc['nsites']

            for e, er in zip(doc['elements'], doc['elements_ratios']):
                if e not in aggregated_info['elements']:
                    aggregated_info['nelements'] += 1
                    aggregated_info['elements'].append(e)
                    aggregated_info['total_elements_ratios'].append(er*doc['nsites'])
                    aggregated_info['individual_elements_ratios'].append(set(
                        [np.round_(er, decimals=2)]
                    ))
                else:
                    idx = aggregated_info['elements'].index(e)
                    aggregated_info['total_elements_ratios'][idx] += er*doc['nsites']
                    aggregated_info['individual_elements_ratios'][idx].add(
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

        aggregated_info['individual_elements_ratios'] = [
            list(_) for _ in aggregated_info['individual_elements_ratios']
        ]

        aggregated_info['total_elements_ratios'] = [
            c/aggregated_info['nsites'] for c in aggregated_info['total_elements_ratios']
        ]

        aggregated_info['individual_elements_ratios'] = list(aggregated_info['individual_elements_ratios'])

        aggregated_info['chemical_formula_reduced'] = list(aggregated_info['chemical_formula_reduced'])
        aggregated_info['chemical_formula_anonymous'] = list(aggregated_info['chemical_formula_anonymous'])
        aggregated_info['chemical_formula_hill'] = list(aggregated_info['chemical_formula_hill'])
        aggregated_info['nperiodic_dimensions'] = list(aggregated_info['nperiodic_dimensions'])
        aggregated_info['dimension_types'] = list(aggregated_info['dimension_types'])

        return aggregated_info


    def aggregate_property_settings_info(self, pso_ids):
        """
        Aggregates the following information from a list of property settings:

            * labels

        Args:

            pso_ids (list or str):
                The IDs of the properties to aggregate the information from

        Returns:

            aggregated_info (dict):
                All of the aggregated info
        """

        if isinstance(pso_ids, str):
            pso_ids = [pso_ids]

        aggregated_info = {
            'labels': [],
        }

        for doc in self.property_settings.find({'_id': {'$in': pso_ids}}):
            for l in doc['labels']:
                if l not in aggregated_info['labels']:
                    aggregated_info['labels'].append(l)

        return aggregated_info


    def aggregate_property_info(self, pr_ids, resync=False):
        """
        Aggregates the following information from a list of properties:

            * types
            * labels
            * labels_counts

        Args:

            pr_ids (list or str):
                The IDs of the configurations to aggregate information from

            resync (bool):
                If True, re-synchronizes the property before aggregating the
                information. Default is False.


        Returns:

            aggregated_info (dict):
                All of the aggregated info
        """

        if isinstance(pr_ids, str):
            pr_ids = [pr_ids]

        if resync:
            for pid in pr_ids:
                self.resync_property(pid)

        aggregated_info = {
            'types': set(),
            'labels': [],
            'labels_counts': []
        }

        for doc in self.properties.find({'_id': {'$in': pr_ids}}):
            aggregated_info['types'].add(doc['type'])

            for l in doc['aggregated_info']['labels']:
                if l not in aggregated_info['labels']:
                    aggregated_info['labels'].append(l)
                    aggregated_info['labels_counts'].append(1)
                else:
                    idx = aggregated_info['labels'].index(l)
                    aggregated_info['labels_counts'][idx] += 1

        aggregated_info['types'] = list(aggregated_info['types'])
        return aggregated_info


    def aggregate_configuration_set_info(self, cs_ids, resync=False):
        """
        Aggregates the following information from a list of configuration sets:

            * nconfigurations
            * nsites
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

            resync (bool):
                If True, re-synchronizes each configuration set before
                aggregating the information. Default is False.

        Returns:

            aggregated_info (dict):
                All of the aggregated info
        """

        # TODO: if the CSs overlap, they'll double count COs...

        if isinstance(cs_ids, str):
            cs_ids = [cs_ids]

        if resync:
            for csid in cs_ids:
                self.resync_configuration_set(csid)

        co_ids = list(set(itertools.chain.from_iterable(
            self.database[f'configuration_sets/{csid}'].attrs['configuration_ids'][()]
            for csid in cs_ids
        )))

        return self.aggregate_configuration_info(co_ids)

        # aggregated_info = {
        #     'nconfigurations': len(co_ids),
        #     'nsites': 0,
        #     'nelements': 0,
        #     'elements': [],
        #     'individual_elements_ratios': [],
        #     'total_elements_ratios': [],
        #     'labels': [],
        #     'labels_counts': [],
        #     'chemical_formula_reduced': set(),
        #     'chemical_formula_anonymous': set(),
        #     'chemical_formula_hill': set(),
        #     'nperiodic_dimensions': set(),
        #     'dimension_types': set(),
        # }


        # for doc in self.configuration_sets.find({'_id': {'$in': cs_ids}}):

        #     agg = doc['aggregated_info']

        #     aggregated_info['nconfigurations'] += agg['nconfigurations']
        #     aggregated_info['nsites'] += agg['nsites']

        #     for e, er, ier in zip(
        #         agg['elements'], agg['total_elements_ratios'],
        #         agg['individual_elements_ratios']
        #         ):
        #         if e not in aggregated_info['elements']:
        #             aggregated_info['nelements'] += 1
        #             aggregated_info['elements'].append(e)
        #             aggregated_info['total_elements_ratios'].append(er*agg['nsites'])
        #             aggregated_info['individual_elements_ratios'].append(
        #                 set(ier)
        #             )
        #         else:
        #             idx = aggregated_info['elements'].index(e)
        #             aggregated_info['total_elements_ratios'][idx] += er*agg['nsites']

        #             old = aggregated_info['individual_elements_ratios'][idx]
        #             new = old.union(set(agg['individual_elements_ratios'][idx]))
        #             aggregated_info['individual_elements_ratios'][idx] = new

        #     for l in agg['labels']:
        #         if l not in aggregated_info['labels']:
        #             aggregated_info['labels'].append(l)
        #             aggregated_info['labels_counts'].append(1)
        #         else:
        #             idx = aggregated_info['labels'].index(l)
        #             aggregated_info['labels_counts'][idx] += 1


        #     for n in ['reduced', 'anonymous', 'hill']:
        #         t = 'chemical_formula_'+n

        #         old = aggregated_info[t]
        #         new = old.union(set(agg[t]))

        #         aggregated_info[t] = new


        #     old = aggregated_info['nperiodic_dimensions']
        #     new = old.union(set(agg['nperiodic_dimensions']))
        #     aggregated_info['nperiodic_dimensions'] = new

        #     old = aggregated_info['dimension_types']
        #     new = old.union(set(tuple(_) for _ in agg['dimension_types']))
        #     aggregated_info['dimension_types'] = new

        # aggregated_info['individual_elements_ratios'] = [list(_) for _ in aggregated_info['individual_elements_ratios']]
        # aggregated_info['chemical_formula_reduced'] = list(aggregated_info['chemical_formula_reduced'])
        # aggregated_info['chemical_formula_anonymous'] = list(aggregated_info['chemical_formula_anonymous'])
        # aggregated_info['chemical_formula_hill'] = list(aggregated_info['chemical_formula_hill'])
        # aggregated_info['nperiodic_dimensions'] = list(aggregated_info['nperiodic_dimensions'])
        # aggregated_info['dimension_types'] = list(aggregated_info['dimension_types'])

        # return aggregated_info


    def insert_dataset(
        self, cs_ids, pr_ids,
        authors=None,
        links=None,
        description='',
        resync=False,
        ):
        """
        Inserts a dataset into the database.

        Args:

            cs_ids (list or str):
                The IDs of the configuration sets to link to the dataset.

            pr_ids (list or str):
                The IDs of the properties to link to the dataset

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

        Returns:

            ds_id (str):
                The ID of the inserted dataset
        """
        ds_id = self.database.insert_dataset(
            cs_ids=cs_ids,
            pr_ids=pr_ids,
            authors=authors,
            links=links,
            description=description
        )

        aggregated_info = {}
        for k,v in self.aggregate_configuration_set_info(cs_ids, resync=resync).items():
            if k == 'labels':
                k = 'configuration_labels'
            elif k == 'labels_counts':
                k = 'configuration_labels_counts'

            aggregated_info[k] = v

        for k,v in self.aggregate_property_info(pr_ids, resync=resync).items():
            if k == 'labels':
                k = 'property_labels'
            elif k == 'labels_counts':
                k = 'property_labels_counts'

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
                    'authors': authors,
                    'links': links,
                    'description': description,
                },
                '$set': {
                    'aggregated_info': aggregated_info,
                    'last_modified': self.database[f'datasets/{ds_id}'].attrs['last_modified']
                },
            },
            upsert=True
        )

        # Add the backwards relationships CS->DS
        for csid in cs_ids:
            self.configuration_sets.update_one(
                {'_id': csid},
                {
                    '$addToSet': {
                        'relationships.configuration_sets': ds_id
                    }
                }
            )

        # Add the backwards relationships PR->DS
        for pid in pr_ids:
            self.properties.update_one(
                {'_id': pid},
                {
                    '$addToSet': {
                        'relationships.configuration_sets': ds_id
                    }
                }
            )

        return ds_id


    def get_dataset(self, ds_id, resync=False):
        """
        Returns the dataset with the given ID.

        Args:

            ds_ids (str):
                The ID of the dataset to return

            resync (bool):
                If True, re-aggregates the configuration set and property
                information before returning. Default is False.

        Returns:

            A dictionary with two keys:
                'last_modified': a datetime string
                'dataset': the dataset object
        """


        if resync:
            self.resync_dataset(ds_id)

        ds_doc = next(self.datasets.find({'_id': ds_id}))

        return {
            'last_modified': ds_doc['last_modified'],
            'dataset': Dataset(
                configuration_set_ids=ds_doc['relationships']['configuration_sets'],
                property_ids=ds_doc['relationships']['properties'],
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


    def apply_configuration_labels(self, query, labels, verbose=False):
        """
        Applies the given labels to all configurations that match the query.

        Args:

            query (dict):
                A Mongo-style query for filtering the configurations. For
                example: :code:`query = {'nsites': {'$lt': 100}}`.

            labels (set or str):
                A set of labels to apply to the matching configurations.

            verbose (bool):
                If True, prints progress bar.

        Pseudocode:
            * Get the IDs of the configurations that match the query
            * Use updateMany to update the MongoDB
            * Iterate over the HDF5 entries.
        """

        if isinstance(labels, str):
            labels = {labels}

        for cdoc in tqdm(
            self.configurations.find(query, {'_id': 1}),
            desc='Applying configuration labels',
            disable=not verbose
            ):
            cid = cdoc['_id']

            self.configurations.update_one(
                {'_id': cid},
                {'$addToSet': {'labels': {'$each': list(labels)}}}
            )

            self.database
            data = self.database[f'configurations/labels/data/{cid}']
            new_labels = set(data.asstr()[()]).union(labels)
            data.resize(
                (len(new_labels),) + data.shape[1:]
            )
            data[:] = list(new_labels)


    def plot_histograms(self, fields=None, ids=None, xscale='linear', yscale='linear'):
        """
        Generates histograms of the given fields.

        Args:

            fields (list or str):
                The names of the fields to plot

            ids (list or str):
                The IDs of the objects to plot the data for
        """

        if fields is None:
            fields = self.property_fields
        elif isinstance(fields, str):
            fields = [fields]

        nfields = len(fields)

        nrows = max(1, int(np.ceil(nfields/3)))
        ncols = max(3, nfields%3)

        # fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=fields)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4*ncols, 2*nrows))
        axes = np.atleast_2d(axes)

        for i, prop in enumerate(fields):
            data = self.get_data(prop, in_memory=True, ravel=True)

            nbins = max(data.shape[0]//1000, 100)

            c = i % 3
            r = i // 3

            ax = axes[r][c]
            _ = ax.hist(data, bins=nbins)

        #     if nrows > 1:
        #         fig.add_trace(
        #             go.Histogram(x=data, nbinsx=nbins),
        #             row=r+1, col=c+1,
        #         )
        #     else:
        #         fig.add_trace(
        #             go.Histogram(x=data, nbinsx=nbins),
        #             row=1, col=c+1
        #         )

        # fig.update_layout(showlegend=False)
        # fig.update_xaxes(type=xscale)
        # fig.update_yaxes(type=yscale)
        plt.tight_layout()


    def get_statistics(self, field):
        """
        Returns the average, standard deviation, minimum, maximum, and average
        absolute value of all entries for the given field .

        Args:

            field (str):
                The name of the field to get statistics for.

        Returns:
            results (dict)::
                ..code-block::
                    {'average': np.average(data), 'std': np.std(data), 'min': np.min(data), 'max': np.max(data), 'average_abs': np.average(np.abs(data))}
        """

        data = self.get_data(field, in_memory=True, ravel=True)

        return {
            'average': np.average(data),
            'std': np.std(data),
            'min': np.min(data),
            'max': np.max(data),
            'average_abs': np.average(np.abs(data)),
        }


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

class ConcatenationException(Exception):
    pass

class InvalidGroupError(Exception):
    pass

class MissingEntryError(Exception):
    pass