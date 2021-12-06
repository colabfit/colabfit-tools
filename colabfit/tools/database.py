import h5py
import json
import warnings
import numpy as np
from ase import Atoms
from copy import deepcopy

from kim_property.definition import check_property_definition
from kim_property.definition import PROPERTY_ID as VALID_KIM_ID

from colabfit import (
    STRING_DTYPE_SPECIFIER
)
from colabfit.tools.configuration import Configuration
from colabfit.tools.property import Property

class Database(h5py.File):
    """
    A Database extends a PyTables (HDF5) file, but provides additional
    functionality for construction, filtering, exploring, ...

    In general, a Database works by grouping the data (coordinates, lattice
    vectors, property fields, ...), maintaining columns of IDs to identify which
    data corresponds to each Configuration/Property/etc., and providing
    additional columns of IDs for mapping from the data to any linked entries
    (configurations, properties, configuration sets, or property settings).

    The underlying HDF5 filesystem has the following structure:

    C  = number of configurations
    I  = number of fields in :attr:`Configuration.info`
    A  = number of fields in :attr:`Configuration.arrays`

    P  = number of properties defined for the Database
    Fp = number of fields for a given property

    S  = number of property settings definitions in the Database

    G  = number of configuration sets in the Database
    D  = number of datasets in the Database

    /root
        /configurations
            This group stores all of the information necessary to construct each
            of the configurations in the database.

            /ids
                /data
                    An array of all Configuration IDs in the Database

            /atomic_numbers
                .attrs
                    concatenated
                        A boolean indicating if /data has been concatenated or
                        not.
                /data
                    The atomic numbers for all configurations in the database.

                    If :code:`concatenated==True`, then /data will contain
                    multiple arrays, one per configuration.

                    If :code:`concatenated==True`, then /data will contain
                    a single array that was generated using something like:

                    ..code-block:: python

                        np.concatenate(
                            # /data/<co_id> for co_id in /data
                        )
                /slices
                        A group where /slices/config_id_1 specifies how to slice
                        /data in order to get the data corresponding to
                        config_id_1.

                        The data can always be extracted using something like:

                        ..code-block:: python

                            slices = database['configurations/field/slices'][()]

                            database.get_data(
                                'configurations/atomic_numbers'
                            )[()][slices]

                        If :code:`concatenated==False`, then /slices/config_id_1
                        will just be config_id_1.

                        If :code:`concatenated==True`, then /slices/config_id_1
                        will be an integer array specifying rows of /data
            /positions
                Same as /atomic_numbers, but for atomic positions
            /cells
                Same as /atomic_numbers, but for lattice vectors
            /pbcs
                Same as /atomic_numbers, but for periodic boundary conditions
            /constraints
                This group is not supported yet. It will be added in the future,
                and will have a similar structure to the /atomic_numbers
        /properties
            This group stores all of the computed properties.

            /ids
                As in /configurations/ids, but for the Property IDs

            /property_1
                .attrs
                    definition
                        An OpenKIM Property Definition as a (serialized)
                        dictionary

                /configuration_ids
                    A list of tuples of Configuration IDs specifying the
                    Configurations associated with each Property. As in
                    /configurations/info/info_field_1/indices, the shape
                    matches data.shape[0], so entries may be duplicated to
                    match the number of rows.

                /settings_ids
                    A list of Property Settings IDs. As in
                    /configuration_ids, entries may be duplicated to match
                    the shape of data.shape[0]

                /field_1
                    /data
                        As in /configurations/info/info_field_1/data
                    /slices
                        As in /configurations/info/info_field_1/slices, but
                        mapping to the Property IDs

                .
                .
                .
                /field_Fp
            .
            .
            .
            /property_P
        /property_settings
            The property settings.

            /property_settings_id_1
                .attrs
                    All of the fields of a PropertySettings object (methods,
                    files, ...)
            .
            .
            .
            /property_settings_id_S
        /configuration_sets
            The configuration sets defining groups of configurations.

            /configuration_set_id_1
                /ids
                    A list of Configuration IDs that belong to the configuration
                    set. Useful for indexing /configurations fields and
                    /properties/property fields.
            .
            .
            .
            /configuration_set_id_G
        /datasets
            /dataset_1
                .attrs
                    authors
                        A list of author names
                    links
                        A list of external links (e.g., journal articles, Git
                        repos, ...)
                    description
                        A human-readable description of the dataset
                    configuration_set_ids
                        The list of configuration set IDs
                    property_ids
                        The list of property IDs
    """

    def __init__(self, name, mode='r', **kwargs):
        """
        Args:
            name (str):
                The path to the database file

            mode (str):
                'r', 'w', 'a', or 'w+'
        """

        super().__init__(name=name, mode=mode, **kwargs)

        # Build all the base groups
        g = self.create_group('configurations')

        for sub_group_name in [
            'ids', 'atomic_numbers', 'positions', 'cells', 'pbcs'
            ]:
            sub_g = g.create_group(sub_group_name)
            sub_g.attrs['concatenated'] = False
            sub_g.create_group('data', track_order=True)

            if sub_group_name != 'ids':
                sub_g.create_group('slices')

        # g.create_group('info')
        # g.create_group('arrays')

        g = self.create_group('properties')
        g_ids = g.create_group('ids')
        g_ids.attrs['concatenate'] = False
        g_ids.create_group('data', track_order=True)

        self.create_group('property_settings')
        self.create_group('configuration_sets')
        self.create_group('datasets')


    def insert_data(
        self,
        configurations, property_map=None, generator=False
        ):
        """
        Inserts the configurations into the databas, and any specified
        properties, and returns an iterable of 2-tuples, where the first entry
        in each tuple is the ID of the inserted property, and the second entry
        is the ID of the associated configuration.
        
        A configuration is added to the database by extracting four fields from
        the Configuration object: atomic numbers, atomic positions, cell lattice
        vectors, and cell periodic boundary conditions.

        A property is added to the database by extracting the fields specified
        by the property definition off of the configuration. Note that an error
        will be thrown if the property hasn't been defined.
        
        Set :code:`generator=True` when the configurations
        can't all fit in memory at the same time. NOTE: if `generator==True`,
        then the configurations will only be added to the dataset once the
        returned IDs have been iterated over

        # TODO: there should also be some way to add properties to existing
        # configurations. Maybe insert_property()

        Args:

            configurations (list or Configuration):
                The list of configurations to be added.

            property_names (list or str):
                The names of the properties that should be loaded off of the

            property_map (dict):
                A dictionary that is used to specify how to load a defined
                property off of a configuration. Note that the top-level keys in
                the map must be the names of properties that have been
                previously defined using
                :meth:`~colabfit.tools.database.Database.add_property_definition`.
                
                If None, only loads the configuration information (atomic
                numbers, positions, lattice vectors, and periodic boundary
                conditions).
                
                Example:

                ..code-block:: python

                    database.add_property_definition(...)

                    property_map = {
                        'property-name-1': {
                            'property-field-1': {
                                'field': 'ase-field-1',
                                'units': 'units-1',
                            }
                        }
                    }

                    database.insert_data(configurations, property_map)

            generator (bool):
                If true, this function becomes a generator which only adds the
                configurations one at a time. This is useful if the
                configurations can't all fit in memory at the same time. Default
                is False.

        Returns:

            additions (iterable):
                An iterable of 2-tuples, where the first entry in each tuple is
                the ID of the inserted property, and the second entry is the ID
                of the associated configuration.
        """

        if isinstance(configurations, Configuration):
            configurations = [configurations]

        if property_map is None:
            property_map = {}

        property_definitions = {
            pname: self.get_property_definition(pname)
            for pname in property_map
        }

        if generator:
            return self._insert_data_gen(
                configurations, property_definitions, property_map
            )
        else:
            return self._insert_data(
                configurations, property_definitions, property_map
            )


    def _insert_data_gen(
        self, configurations, property_definitions, property_map
        ):
        if isinstance(configurations, Configuration):
            configurations = [configurations]

        ignore_keys = {'property-id', 'property-title', 'property-description'}
        expected_keys = {
            pname: set(
                property_definitions[pname].keys()
            ) - ignore_keys
            for pname in property_map
        }

        for ai, atoms in enumerate(configurations):
            config_id = str(hash(atoms))

            # Save the ID
            g = self['configurations/ids/data']
            g.create_dataset(
                name=config_id,
                shape=1,
                data=np.array(config_id, dtype=STRING_DTYPE_SPECIFIER)
            )

            # Save all fundamental information about the configuration
            g = self['configurations/atomic_numbers']
            g['data'].create_dataset(
                name=config_id,
                data=atoms.get_atomic_numbers(),
            )
            g[f'slices/{config_id}'] = np.array(config_id, dtype=STRING_DTYPE_SPECIFIER)

            g = self['configurations/positions']
            g['data'].create_dataset(
                name=config_id,
                data=atoms.get_positions()
            )
            g[f'slices/{config_id}'] = np.array(config_id, dtype=STRING_DTYPE_SPECIFIER)

            g = self['configurations/cells']
            g['data'].create_dataset(
                name=config_id,
                data=np.array(atoms.get_cell())
            )
            g[f'slices/{config_id}'] = np.array(config_id, dtype=STRING_DTYPE_SPECIFIER)

            g = self['configurations/pbcs']
            g['data'].create_dataset(
                name=config_id,
                data=atoms.get_pbc().astype(int),
            )
            g[f'slices/{config_id}'] = np.array(config_id, dtype=STRING_DTYPE_SPECIFIER)

            # Try to load all of the specified properties
            available_keys = set().union(atoms.info.keys(), atoms.arrays.keys())

            for pname, pmap in property_map.items():

                # Pre-check to avoid having to delete partially-added properties
                missing_keys = expected_keys[pname] - available_keys
                if missing_keys:
                    warnings.warn(
                        "Configuration {} is missing keys ({}) during "\
                        "insert_data. Skipping".format(
                            ai, missing_keys
                        )
                    )
                    continue

                prop = Property.from_definition(
                    pname, property_definitions[pname],
                    atoms, pmap
                )

                prop_id = str(hash(prop))

                # Add the data; group should already exist
                for field in expected_keys[pname]:
                    g = self[f'properties/{pname}/{field}']

                    # Try to convert field into either a float or string array
                    v = prop[field]['source-value']

                    if isinstance(v, str):
                        v = np.atleast_1d(v).astype(STRING_DTYPE_SPECIFIER)
                    if isinstance(v, set):
                        # These should always be sets of strings
                        v = np.atleast_1d(list(v)).astype(STRING_DTYPE_SPECIFIER)
                    else:
                        v = np.atleast_1d(v)

                    g['data'].create_dataset(
                        name=prop_id,
                        data=v
                    )
                    g[f'slices/{prop_id}'] = np.array(
                        prop_id, dtype=STRING_DTYPE_SPECIFIER
                    )

                yield (prop_id, config_id)


    def _insert_data(self, configurations, property_definitions, property_map):
        if isinstance(configurations, Configuration):
            configurations = [configurations]

        ignore_keys = {'property-id', 'property-title', 'property-description'}
        expected_keys = {
            pname: set(
                property_definitions[pname].keys()
            ) - ignore_keys
            for pname in property_map
        }

        additions = []

        for ai, atoms in enumerate(configurations):
            config_id = str(hash(atoms))

            # Save the ID
            g = self['configurations/ids/data']
            g.create_dataset(
                name=config_id,
                shape=1,
                data=np.array(config_id, dtype=STRING_DTYPE_SPECIFIER)
            )

            # Save all fundamental information about the configuration
            g = self['configurations/atomic_numbers']
            g['data'].create_dataset(
                name=config_id,
                data=atoms.get_atomic_numbers(),
            )
            g[f'slices/{config_id}'] = np.array(config_id, dtype=STRING_DTYPE_SPECIFIER)

            g = self['configurations/positions']
            g['data'].create_dataset(
                name=config_id,
                data=atoms.get_positions()
            )
            g[f'slices/{config_id}'] = np.array(config_id, dtype=STRING_DTYPE_SPECIFIER)

            g = self['configurations/cells']
            g['data'].create_dataset(
                name=config_id,
                data=np.array(atoms.get_cell())
            )
            g[f'slices/{config_id}'] = np.array(config_id, dtype=STRING_DTYPE_SPECIFIER)

            g = self['configurations/pbcs']
            g['data'].create_dataset(
                name=config_id,
                data=atoms.get_pbc().astype(int),
            )
            g[f'slices/{config_id}'] = np.array(config_id, dtype=STRING_DTYPE_SPECIFIER)

            # Try to load all of the specified properties
            available_keys = set().union(atoms.info.keys(), atoms.arrays.keys())

            for pname, pmap in property_map.items():

                # Pre-check to avoid having to delete partially-added properties
                missing_keys = expected_keys[pname] - available_keys
                if missing_keys:
                    warnings.warn(
                        "Configuration {} is missing keys ({}) during "\
                        "insert_data. Skipping".format(
                            ai, missing_keys
                        )
                    )
                    continue

                prop = Property.from_definition(
                    pname, property_definitions[pname],
                    atoms, pmap
                )

                prop_id = str(hash(prop))

                # Add the data; group should already exist
                for field in expected_keys[pname]:
                    g = self[f'properties/{pname}/{field}']

                    # Try to convert field into either a float or string array
                    v = prop[field]['source-value']

                    if isinstance(v, str):
                        v = np.atleast_1d(v).astype(STRING_DTYPE_SPECIFIER)
                    if isinstance(v, set):
                        # These should always be sets of strings
                        v = np.atleast_1d(list(v)).astype(STRING_DTYPE_SPECIFIER)
                    else:
                        v = np.atleast_1d(v)

                    g['data'].create_dataset(
                        name=prop_id,
                        data=v
                    )
                    g[f'slices/{prop_id}'] = np.array(
                        prop_id, dtype=STRING_DTYPE_SPECIFIER
                    )

                g = self['properties/ids/data']
                g.create_dataset(
                    name=prop_id,
                    shape=1,
                    data=np.array(prop_id, dtype=STRING_DTYPE_SPECIFIER)
                )

                g = self[f'properties/{pname}/configuration_ids/data']
                g.create_dataset(
                    name=prop_id,
                    shape=1,
                    data=np.array(prop_id, dtype=STRING_DTYPE_SPECIFIER)
                )
                g[f'slices/{prop_id}'] = np.array(
                    config_id, dtype=STRING_DTYPE_SPECIFIER
                )

                # TODO: insert_property_settings(); update here


                # yield (prop_id, config_id)
                additions.append((prop_id, config_id))

        return additions




            # for k, v in atoms.info.items():
            #     g = self['configurations/info'].require_group(k)
            #     # Extract data and slices groups
            #     if 'data' in g:
            #         g_data = g['data']
            #     else:
            #         g_data = g.create_group('data', track_order=True)

            #     if 'slices' in g:
            #         g_slices = g['slices']
            #     else:
            #         g_slices = g.create_group('slices', track_order=True)

            #     if isinstance(v, str):
            #         v = np.atleast_1d(v).astype(STRING_DTYPE_SPECIFIER)
            #     if isinstance(v, (set, list)):
            #         # These should always be sets of strings
            #         v = np.atleast_1d(list(v)).astype(STRING_DTYPE_SPECIFIER)
            #     else:
            #         v = np.atleast_1d(v)

            #     # Add in info fields
            #     g_data.create_dataset(
            #         name=config_id,
            #         shape=v.shape,
            #         dtype=v.dtype,
            #         data=v,
            #         maxshape=(None,)+v.shape[1:],
            #     )

            #     g_slices[config_id] = config_id

            #     g.attrs['concatenated'] = False

            # for k, v in atoms.arrays.items():
            #     g = self['configurations/arrays'].require_group(k)
            #     if 'data' in g:
            #         g_data = g['data']
            #     else:
            #         g_data = g.create_group('data', track_order=True)

            #     if 'slices' in g:
            #         g_slices = g['slices']
            #     else:
            #         g_slices = g.create_group('slices', track_order=True)

            #     # Add in arrays fields
            #     g_data.create_dataset(
            #         name=config_id,
            #         shape=v.shape,
            #         dtype=v.dtype,
            #         data=v,  # should already be an array
            #         maxshape=(None,)+v.shape[1:],
            #     )

            #     g_slices[config_id] = config_id

            #     g.attrs['concatenated'] = False

        #     ids.append(config_id)

        # return ids


    def concatenate_group(self, group, chunks=None):
        """
        Attempt to concatenate all of the datasets in a group. Raise an
        exception if the datasets in the group have incompatible shapes.

        Args:

            group_name (str or group):
                The name of a group in the database, or the group object

            chunks (tuple):
                An optional argument describing how to chunk the concatenated
                array. Chunk shapes should be chosed based on the desired access
                pattern. See `chunked storage <https://docs.h5py.org/en/stable/high/dataset.html#chunked-storage>_
                in the h5py documentation for more details.
        """

        if isinstance(group, str):
            group = self[group]

        if 'data' not in group:
            raise InvalidGroupError(
                "The group '{}' does not have a 'data' sub-group".format(
                    group
                )
            )

        # Just initialize an array of the right size
        try:
            example = next(iter(group['data'].values()))
        except StopIteration:
            raise ConcatenationException(
                "Group '{}/data' sub-group is empty. Make sure to consume the "\
                "generator from insert_data() if you used "\
                "generator=True.".format(group.name)
            )

        n = sum((_.shape[0] for _ in group['data'].values()))


        bigshape = (n,) + example.shape[1:]
        data = group.create_dataset(
            name=None, shape=bigshape, dtype=example.dtype, chunks=chunks
        )

        start = 0
        problem_adding = False

        # TODO: need to figure out how to handle the case where the field has
        # already been concatenated before.

        if '_root_concatenated' in group['data']:
            # Copy any already-concatenated data
            ds = group['data/_root_concatenated']
            stop = start + ds.shape[0]
            data[start:stop] = ds
            start += ds.shape[0]

        for ds_name, ds in group['data'].items():
            # if os.path.split(ds.name)[-1] != '_root_concatenated':
            if ds_name == '_root_concatenated':
                # This was added first
                continue

            try:
                # Copy the data over
                stop = start + ds.shape[0]
                data[start:stop] = ds

                if 'slices' in group:
                    # Some groups (like IDs) don't need to be sliced
                    key = ds.name.split("/")[-1]

                    del group[f'slices/{key}']
                    group[f'slices/{key}'] = np.arange(start, stop)

                start += ds.shape[0]
            except Exception as e:
                problem_adding = True
                break

        if problem_adding:
            # Get rid of the new dataset that was added
            del data
            raise ConcatenationException(
                "Expected shape {}, but got shape {} for dataset '{}'".format(
                    (None,)+bigshape[1:], ds.shape, ds.name
                )
            )
        else:
            # Get rid of all of the old stuff
            del group['data']
            g = group.create_group('data', track_order=True)
            g['_root_concatenated'] = data
            group.attrs['concatenated'] = True


    def get_data(self, group, in_memory=False):
        """
        Returns all of the datasets in the 'data' sub-group of
        :code:`<group_name>`.

        Args:

            group_name (str or group):
                The name of a group in the database, or the group object

            in_memory(bool):
                If True, converts each of the datasets to a Numpy array before
                returning.
        """
        if isinstance(group, str):
            group = self[group]

        if 'data' not in group:
            raise InvalidGroupError(
                "The group '{}' does not have a 'data' sub-group".format(
                    group
                )
            )
        else:
            g = group['data']

            if group.attrs['concatenated']:
                return g['_root_concatenated'][()] if in_memory else g['_root_concatenated']
            else:
                return {k: ds[()] if in_memory else ds for k, ds in g.items()}
                # return [ds[()] if in_memory else ds for ds in g.values()]


    def get_configuration(self, i):
        return self.get_configurations([i])

    def get_configurations(self, ids, generator=False):
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

        Returns:

            configurations (iterable):
                A list or generator of the re-constructed configurations
        """

        if generator:
            return self._get_configurations_gen(ids)
        else:
            return self._get_configurations(ids)


    def _get_configurations(self, ids):
        if ids == 'all':
            ids = self.get_data('configurations/ids')

        configurations = []

        for co_id in ids:
            atoms = Atoms(
                symbols=self.get_data(
                    'configurations/atomic_numbers'
                )[self[f'configurations/atomic_numbers/slices/{co_id}'][()]],

                positions=self.get_data(
                    'configurations/positions'
                )[self[f'configurations/positions/slices/{co_id}'][()]],

                cell=self.get_data(
                    'configurations/cells'
                )[self[f'configurations/cells/slices/{co_id}'][()]],

                pbc=self.get_data(
                    'configurations/pbcs'
                )[self[f'configurations/pbcs/slices/{co_id}'][()]],
            )

            configurations.append(Configuration.from_ase(atoms))

        return configurations



    def _get_configurations_gen(self, ids):
        if ids == 'all':
            ids = self.get_data('configurations/ids')

        configurations = []

        for co_id in ids:
            atoms = Atoms(
                symbols=self.get_data(
                    'configurations/atomic_numbers'
                )[self[f'configurations/atomic_numbers/slices/{co_id}'][()]],

                positions=self.get_data(
                    'configurations/positions'
                )[self[f'configurations/positions/slices/{co_id}'][()]],

                cell=self.get_data(
                    'configurations/cells'
                )[self[f'configurations/cells/slices/{co_id}'][()]],

                pbc=self.get_data(
                    'configurations/pbcs'
                )[self[f'configurations/pbcs/slices/{co_id}'][()]],
            )

            yield Configuration.from_ase(atoms)


    def concatenate_configurations(self):
        self.concatenate_group('configurations/atomic_numbers')
        self.concatenate_group('configurations/positions')
        self.concatenate_group('configurations/cells')
        self.concatenate_group('configurations/pbcs')

        # for field in self['configurations/info']:
        #     try:
        #         self.concatenate_group(self['configurations/info'][field])
        #     except ConcatenationException as e:
        #         pass

        # for field in self['configurations/arrays']:
        #     try:
        #         self.concatenate_group(self['configurations/arrays'][field])
        #     except ConcatenationException:
        #         pass


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

        dummy_dict = deepcopy(definition)

        # Spoof if necessary
        if VALID_KIM_ID.match(dummy_dict['property-id']) is None:
            # Invalid ID. Try spoofing it
            dummy_dict['property-id'] = 'tag:@,0000-00-00:property/'
            dummy_dict['property-id'] += definition['property-id']
            warnings.warn(f"Invalid KIM property-id; Temporarily renaming to {dummy_dict['property-id']}")

        check_property_definition(dummy_dict)

        group = self['properties'].require_group(definition['property-id'])
        group.attrs['definition'] = json.dumps(definition)

        ignore_fields = {
            'property-id', 'property-title', 'property-description'
        }

        for subgroup in ['configuration_ids', 'settings_ids']:
            g = group.create_group(subgroup)
            g.create_group('data', track_order=True)
            g.create_group('slices')
            g.attrs['concatenated'] = False

        for field in set(definition.keys()) - ignore_fields:
            g = group.create_group(field)
            g.create_group('data', track_order=True)
            g.create_group('slices')
            g.attrs['concatenated'] = False


    def get_property_definition(self, name):
        return json.loads(self[f'properties/{name}'].attrs['definition'])


    def insert_property_settings(self, pso_object):
        """
        Inserts a new property settings object into the database by creating
        and populating the necessary groups in :code:`/root/property_settings`.

        Args:

            pso_object (PropertySettings)
                The :class:`~colabfit.tools.property_settings.PropertySettings`
                object to insert into the database.
        """
        pass


class ConcatenationException(Exception):
    pass

class InvalidGroupError(Exception):
    pass