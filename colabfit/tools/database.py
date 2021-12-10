import os
import h5py
import json
import warnings
import datetime
import numpy as np
from ase import Atoms
from copy import deepcopy

from kim_property.definition import check_property_definition
from kim_property.definition import PROPERTY_ID as VALID_KIM_ID

from colabfit import (
    ATOMS_LABELS_FIELD,
    ATOMS_LAST_MODIFIED_FIELD,
    ATOMS_NAME_FIELD,
    STRING_DTYPE_SPECIFIER
)
from colabfit.tools.configuration import Configuration
from colabfit.tools.property import Property
from colabfit.tools.property_settings import PropertySettings

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

            /names
                Same as /atomic_numbers, but for configuration names. Note that
                a single configuration may have been given multiple names.
            /labels
                Same as /atomic_numbers, but for labels
            /last_modified
                Same as /atomic_numbers, but for datetime strings specifying
                when the configuration was last modified.
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
                and will have a similar structure to /atomic_numbers
        /properties
            This group stores all of the computed properties.

            /ids
                As in /configurations/ids, but for the Property IDs

            /types
                As in /ids, but the type of the property (e.g., "property_1")

            /settings_ids
                A list of Property Settings IDs. Uses the /data, /slices
                form.

            /last_modified
                Same as configurations/atomic_numbers, but for datetime strings
                specifying when the configuration was last modified.

            /property_1
                .attrs
                    definition:
                        An OpenKIM Property Definition as a (serialized)
                        dictionary

                    last_modified:
                        A datetime string specifying when the object was
                        modified last.

                /configuration_ids
                    A list of tuples of Configuration IDs specifying the
                    Configurations associated with each Property. Uses the
                    /data, /slices form.

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
                    method: VASP, QuantumEspresso, ...

                    description:
                        A description of the calculation

                    last_modified:
                        A datetime string specifying when the object was
                        modified last.

                    labels:
                        Labels applied to the property settings

                /files
                    Any files associated with the property settings, stored as
                    raw text strings. Uses the /data form, as in
                    /configurations/ids.
            .
            .
            .
            /property_settings_id_S
        /configuration_sets
            The configuration sets defining groups of configurations.

            /configuration_set_id_1
                .attrs
                    description:
                        Human-readable description of the set

                    last_modified:
                        A datetime string specifying when the object was
                        modified last.
                /ids
                    A list of Configuration IDs that belong to the configuration
                    set. Useful for indexing /configurations fields and
                    /properties/property fields. Uses the /data form, as in
                    /configurations/ids.
            .
            .
            .
            /configuration_set_id_G
        /datasets
            /dataset_1
                .attrs
                    authors:
                        A list of author names

                    links:
                        A list of external links (e.g., journal articles, Git
                        repos, ...)

                    description:
                        A human-readable description of the dataset

                    configuration_set_ids:
                        The list of configuration set IDs

                    property_ids:
                        The list of property IDs

                    last_modified:
                        A datetime string specifying when the object was
                        modified last.
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
            'ids', 'atomic_numbers', 'positions', 'cells', 'pbcs',
            'names', 'labels', 'last_modified'
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
        g_ids.attrs['concatenated'] = False
        g_ids.create_group('data', track_order=True)

        g_types = g.create_group('types')
        g_types.attrs['concatenated'] = False
        g_types.create_group('data', track_order=True)

        g_settings = g.create_group('settings_ids')
        g_settings.attrs['concatenated'] = False
        g_settings.create_group('data', track_order=True)

        self.create_group('property_settings')
        self.create_group('configuration_sets')
        self.create_group('datasets')


    def insert_data(
        self,
        configurations, property_map=None, property_settings=None,
        generator=False
        ):
        """
        Inserts the configurations into the databas, and any specified
        properties, and returns an iterable of 2-tuples, where the first entry
        in each tuple is the ID of the inserted property, and the second entry
        is the ID of the associated configuration.

        A configuration is added to the database by extracting seven fields from
        the Configuration object:

            * atomic numbers (:meth:`get_atomic_numbers`)
            * atomic positions (:meth:`get_positions`)
            * cell lattice vectors (:meth:`get_cell`)
            * cell periodic boundary conditions (:meth:`get_pbc`)
            * names (:attr:`info['_name'])
            * labels (:attr:`info['_labels'])
            * last_modified (:meth:`datetime.now`)

        A property is added to the database by extracting the fields specified
        by the property definition off of the configuration. Note that an error
        will be thrown if the property hasn't been defined yet using
        :meth:`~colabfit.tools.database.Database.add_property_definition`.

        Set :code:`generator=True` when the configurations
        can't all fit in memory at the same time. NOTE: if `generator==True`,
        then the configurations will only be added to the dataset once the
        returned IDs have been iterated over

        Note that new properties can be added to existing configurations by
        passing the same configurations to insert_data(), but using a new
        property map.

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

            pso_id_1 = database.add_property_settings(
                PropertySettings(...)
            )

            property_settings = {
                'property-name': pso_id_1
            }

            database.insert_data(
                configurations, property_map, property_settings
            )


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

        if property_settings is None:
            property_settings = {}

        for settings_id in property_settings.values():
            if settings_id not in self['property_settings']:
                raise MissingEntryError(
                    "The property settings object with ID '{}' does"\
                    " not exist in the database".format(settings_id)
                )

        property_definitions = {
            pname: self.get_property_definition(pname)
            for pname in property_map
        }

        if generator:
            return self._insert_data_gen(
                configurations,
                property_definitions, property_map, property_settings
            )
        else:
            return self._insert_data(
                configurations,
                property_definitions, property_map, property_settings
            )


    def _insert_data_gen(
        self, configurations,
        property_definitions, property_map, property_settings
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

            g = self['configurations/ids/data']

            if config_id in g:  # Identical CO already exists in the dataset
                # So just update /names, /labels, and /last_modified
                for f2 in ['names', 'labels', 'last_modified']:
                    g = self[f'configurations/{f2}']
                    if g.attrs['concatenated']:
                        raise ConcatenationException(
                            "Trying to update a configuration after "\
                            "concatenating is not allowed."
                        )

                # Now append to existing datasets
                # Names
                data = self[f'configurations/names/data/{config_id}']
                new_name = atoms.info[ATOMS_NAME_FIELD]
                if new_name == '':
                    new_name = []
                else:
                    new_name = [new_name]
                names_set = set(new_name) - set(data.asstr()[()])
                data.resize((data.shape[0]+len(names_set),) + data.shape[1:])
                data[-len(names_set):] = np.array(
                    list(names_set), dtype=STRING_DTYPE_SPECIFIER
                )

                # Labels
                data = self[f'configurations/labels/data/{config_id}']
                labels = atoms.info[ATOMS_LABELS_FIELD]
                data.resize(
                    (data.shape[0]+len(labels),) + data.shape[1:]
                )
                data[-len(labels):] = list(labels)

                # Last modified
                data = self[f'configurations/names/data/{config_id}']
                data = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')

            else:
                # Adding a new CO
                # Names
                g = self['configurations/names']
                name = atoms.info[ATOMS_NAME_FIELD]
                if name == '':
                    name = []
                else:
                    name = [name]
                g['data'].create_dataset(
                    name=config_id,
                    shape=(len(name),),
                    maxshape=(None,),
                    dtype=STRING_DTYPE_SPECIFIER,
                    data=name
                )
                g[f'slices/{config_id}'] = np.array(
                    config_id, dtype=STRING_DTYPE_SPECIFIER
                )
                # Labels
                g = self['configurations/labels']
                labels = list(atoms.info[ATOMS_LABELS_FIELD])
                g['data'].create_dataset(
                    name=config_id,
                    shape=(len(labels),),
                    maxshape=(None,),
                    dtype=STRING_DTYPE_SPECIFIER,
                    data=labels
                )
                g[f'slices/{config_id}'] = np.array(
                    config_id, dtype=STRING_DTYPE_SPECIFIER
                )
                # Last modified
                g = self['configurations/last_modified']
                g['data'].create_dataset(
                    name=config_id,
                    shape=(1,),
                    maxshape=(None,),
                    dtype=STRING_DTYPE_SPECIFIER,
                    data=datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
                )
                g[f'slices/{config_id}'] = np.array(
                    config_id, dtype=STRING_DTYPE_SPECIFIER
                )

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

            # Flag for tracking if we need to return (None, config_id)
            returned_something = False

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

                # Check for duplicate property. Note that if even a single field
                # is changed, it is considered a new property. This can lead to
                # duplicate data (for the unchanged fields), but is still the
                # desired behaviour.
                if prop_id in self['properties/ids/data']:
                    yield (config_id, prop_id)
                    continue

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

                g = self[f'properties/{pname}/configuration_ids']
                g['data'].create_dataset(
                    name=prop_id,
                    shape=1,
                    data=np.array(config_id, dtype=STRING_DTYPE_SPECIFIER)
                )
                g[f'slices/{prop_id}'] = np.array(
                    prop_id, dtype=STRING_DTYPE_SPECIFIER
                )

                # Attach property settings, if any were given
                if pname in property_settings:
                    settings_id = property_settings[pname]

                    # TODO: doesn't support multiple settings per prop
                    g = self[f'properties/settings_ids/data']
                    g.create_dataset(
                        name=prop_id,
                        shape=1,
                        data=np.array(settings_id, dtype=STRING_DTYPE_SPECIFIER)
                    )
                    g[f'slices/{prop_id}'] = np.array(
                        prop_id, dtype=STRING_DTYPE_SPECIFIER
                    )

                yield (config_id, prop_id)
                returned_something = True

            if not returned_something:
                yield (config_id, None)


    def _insert_data(
        self, configurations,
        property_definitions, property_map, property_settings
        ):
        """
        TODO:

        * Add support for duplicate configurations. If the CO ID already exists,
        that means the COs are identical (since the ID is the hash), so all that
        needs to be done is to update /names, /labels, and /last_modified.

        * Add support for duplicate properties

        """
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

            g = self['configurations/ids/data']

            if config_id in g:  # Identical CO already exists in the dataset
                # So just update /names, /labels, and /last_modified
                for f2 in ['names', 'labels', 'last_modified']:
                    g = self[f'configurations/{f2}']
                    if g.attrs['concatenated']:
                        raise ConcatenationException(
                            "Trying to update a configuration after "\
                            "concatenating is not allowed."
                        )

                # Now append to existing datasets
                # Names
                data = self[f'configurations/names/data/{config_id}']
                new_names = atoms.info[ATOMS_NAME_FIELD]
                names_set = set(new_names) - set(data.asstr()[()])
                data.resize((data.shape[0]+len(names_set),) + data.shape[1:])
                data[-len(names_set):] = np.array(
                    list(names_set), dtype=STRING_DTYPE_SPECIFIER
                )

                # Labels
                data = self[f'configurations/labels/data/{config_id}']
                labels = atoms.info[ATOMS_LABELS_FIELD]
                data.resize(
                    (data.shape[0]+len(labels),) + data.shape[1:]
                )
                data[-len(labels):] = list(labels)

                # Last modified
                data = self[f'configurations/names/data/{config_id}']
                data = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')

            else:
                # Adding a new CO
                # Names
                g = self['configurations/names']
                names = list(
                    atoms.info[ATOMS_NAME_FIELD]
                )
                g['data'].create_dataset(
                    name=config_id,
                    shape=(len(names),),
                    maxshape=(None,),
                    dtype=STRING_DTYPE_SPECIFIER,
                    data=names
                )
                g[f'slices/{config_id}'] = np.array(
                    config_id, dtype=STRING_DTYPE_SPECIFIER
                )
                # Labels
                g = self['configurations/labels']
                labels = list(atoms.info[ATOMS_LABELS_FIELD])
                g['data'].create_dataset(
                    name=config_id,
                    shape=(len(labels),),
                    maxshape=(None,),
                    dtype=STRING_DTYPE_SPECIFIER,
                    data=labels
                )
                g[f'slices/{config_id}'] = np.array(
                    config_id, dtype=STRING_DTYPE_SPECIFIER
                )
                # Last modified
                g = self['configurations/last_modified']
                g['data'].create_dataset(
                    name=config_id,
                    shape=(1,),
                    maxshape=(None,),
                    dtype=STRING_DTYPE_SPECIFIER,
                    data=datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
                )
                g[f'slices/{config_id}'] = np.array(
                    config_id, dtype=STRING_DTYPE_SPECIFIER
                )

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

            # Flag for tracking if we need to return (None, config_id)
            returned_something = False

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

                # Check for duplicate property. Note that if even a single field
                # is changed, it is considered a new property. This can lead to
                # duplicate data (for the unchanged fields), but is still the
                # desired behaviour.
                if prop_id in self['properties/ids/data']:
                    additions.append((config_id, prop_id))
                    continue

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

                g = self[f'properties/types/data/{prop_id}'] = pname

                g = self[f'properties/{pname}/configuration_ids']
                g['data'].create_dataset(
                    name=prop_id,
                    shape=1,
                    data=np.array(config_id, dtype=STRING_DTYPE_SPECIFIER)
                )
                g[f'slices/{prop_id}'] = np.array(
                    prop_id, dtype=STRING_DTYPE_SPECIFIER
                )

                # Attach property settings, if any were given
                if pname in property_settings:
                    settings_id = property_settings[pname]

                    g = self[f'properties/settings_ids/{pname}/data']
                    g.create_dataset(
                        name=prop_id,
                        shape=1,
                        data=np.array(settings_id, dtype=STRING_DTYPE_SPECIFIER)
                    )
                    g[f'slices/{prop_id}'] = np.array(
                        prop_id, dtype=STRING_DTYPE_SPECIFIER
                    )

                now = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
                self[f'properties/last_modified/data/{prop_id}'] = now

                # yield (prop_id, config_id)
                additions.append((config_id, prop_id))
                returned_something = True

            if not returned_something:
                additions.append((config_id, None))

        return additions


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

        if '_root_concatenated' in group['data']:
            # Copy any already-concatenated data
            ds = group['data/_root_concatenated']
            stop = start + ds.shape[0]
            data[start:stop] = ds
            start += ds.shape[0]

        for ds_name, ds in group['data'].items():
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


    def get_data(
        self, group, in_memory=False,
        concatenate=False, ravel=False, as_str=False
        ):
        """
        Returns all of the datasets in the 'data' sub-group of
        :code:`<group_name>`.

        Args:

            group_name (str or group):
                The name of a group in the database, or the group object

            in_memory (bool):
                If True, converts each of the datasets to a Numpy array before
                returning.

            concatenate (bool):
                If True, concatenates the data before returning. Only available
                if :code:`in_memory==True`.

            ravel (bool):
                If True, concatenates and ravels the data before returning. Only
                available if :code:`in_memory==True`.

            as_str (bool):
                If True, tries to call :code:`asstr()` to convert from an HDF5
                bytes array to an array of strings
        """
        if isinstance(group, str):
            group = self[group]

        if concatenate or ravel:
            if not in_memory:
                raise ConcatenationException(
                    "Cannot use concatenate=True or ravel=True without "\
                    "in_memory=True"
                )

        if 'data' not in group:
            raise InvalidGroupError(
                "The group '{}' does not have a 'data' sub-group".format(
                    group
                )
            )
        else:
            g = group['data']

            if group.attrs['concatenated']:
                data = g['_root_concatenated']

                if as_str:
                    data = data.asstr()
                if in_memory:
                    data = data[()]
                if ravel:
                    data = data.ravel()

                return data
            else:
                keys = g.keys()
                data = g.values()

                if as_str:
                    data = [_.asstr() for _ in data]

                if concatenate or ravel:
                    return np.concatenate(list(data))

                if ravel:
                    return data.ravel()

                return {
                    # encode since /slices will have bytes
                    k.encode('utf-8'): ds for k, ds in zip(keys, data)
                }


    def get_configuration(self, i):
        """
        Returns a single configuration by calling :meth:`get_configurations`
        """
        return self.get_configurations([i])[0]

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
            ids = [
                ds.asstr()[0]
                for ds in self.get_data('configurations/ids').values()
            ]

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

            atoms.info[ATOMS_NAME_FIELD] = set(self.get_data(
                'configurations/names'
            )[self[f'configurations/names/slices/{co_id}'][()]].asstr())

            atoms.info[ATOMS_LABELS_FIELD] = set(self.get_data(
                'configurations/labels'
            )[self[f'configurations/labels/slices/{co_id}'][()]].asstr())

            atoms.info[ATOMS_LAST_MODIFIED_FIELD] = self.get_data(
                'configurations/last_modified'
            )[self[f'configurations/last_modified/slices/{co_id}'][()]].asstr()[0]

            configurations.append(Configuration.from_ase(atoms))

        return configurations



    def _get_configurations_gen(self, ids):
        if ids == 'all':
            ids = [
                ds.asstr()[0]
                for ds in self.get_data('configurations/ids').values()
            ]

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
        """
        Concatenates the atomic_numbers, positions, cells, and pbcs groups in
        /configurations.
        """
        self.concatenate_group('configurations/atomic_numbers')
        self.concatenate_group('configurations/positions')
        self.concatenate_group('configurations/cells')
        self.concatenate_group('configurations/pbcs')


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

        now = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
        group.attrs['last_modified'] = now

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
        """
        Returns:
            A dictionary with two keys:
                'last_modified': a datetime string
                'definition': the dictionary form of the property definition
        """
        return {
            'last_modified': self[f'properties/{name}'].attrs['last_modified'],
            'definition': json.loads(
                self[f'properties/{name}'].attrs['definition']
            )
        }


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

        if pso_id in self['property_settings']:
            return pso_id

        g = self['property_settings'].create_group(pso_id)

        now = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
        g.attrs['last_modified']    = now
        g.attrs['method']           = pso_object.method
        g.attrs['description']      = pso_object.description
        g.attrs['labels']           = json.dumps(list(pso_object.labels))

        g = g.create_group('files')
        for (fname, contents) in pso_object.files:
            g[fname] = contents

        return pso_id


    def get_property_settings(self, pso_id):
        """
        Returns:
            A dictionary with two keys:
                'last_modified': a datetime string
                'settings': the PropertySettings object with the given ID
        """
        g = self[f'property_settings/{pso_id}']

        return {
            'last_modified': g.attrs['last_modified'],
            'settings': PropertySettings(
                method=g.attrs['method'],
                description=g.attrs['description'],
                labels=set(json.loads(g.attrs['labels'])),
                files=[(fname, ds.asstr()[()]) for fname, ds in g['files'].items()]
            )
        }


    def get_properties(self, ids):
        """
        This function should take a list of PR IDs, and re-construct the
        properties from them.
        """
        pass


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

        if isinstance(ids, str):
            ids = [ids]

        cs_id = str(hash(tuple(ids)))

        # Check for duplicates
        if cs_id in self['configuration_sets']:
            return cs_id

        # Make sure all of the configurations exist
        for co_id in ids:
            if co_id not in self['configurations/ids/data']:
                raise MissingEntryError(
                    "The configuration with ID '{}' is not in the "\
                    "database".format(co_id)
                )

        g = self['configuration_sets'].create_group(cs_id)
        g.attrs['description'] = description
        g = g.create_group('ids')
        g.attrs['concatenated'] = True
        g = g.create_group('data', track_order=True)
        g.create_dataset(
            name='_root_concatenated',
            data=np.array(ids, dtype=STRING_DTYPE_SPECIFIER)
        )

        now = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
        g.attrs['last_modified']    = now

        return cs_id


    def insert_dataset(
        self, cs_ids, pr_ids,
        authors, links, description,
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
        """

        if isinstance(cs_ids, str):
            cs_ids = [cs_ids]

        if isinstance(pr_ids, str):
            pr_ids = [pr_ids]

        if isinstance(authors, str):
            authors = [authors]

        if isinstance(links, str):
            links = [links]

        ds_id = str(hash((
            hash(tuple(cs_ids)), hash(tuple(pr_ids))
        )))

        # Check for duplicates
        if ds_id in self['datasets']:
            return ds_id

        # Make sure all of the configuration sets and properties exist
        for cs_id in cs_ids:
            if cs_id not in self['configuration_sets']:
                raise MissingEntryError(
                    "The configuration set with ID '{}' is not in the "\
                    "database".format(cs_id)
                )

        for pr_id in pr_ids:
            if pr_id not in self['properties/ids/data']:
                raise MissingEntryError(
                    "The configuration set with ID '{}' is not in the "\
                    "database".format(pr_id)
                )

        g = self['datasets'].create_group(ds_id)
        g.attrs['authors'] = authors
        g.attrs['links'] = links
        g.attrs['description'] = description

        g = g.create_group('configuration_set_ids')
        g.attrs['concatenated'] = True
        g = g.create_group('data', track_order=True)
        g.create_dataset(
            name='_root_concatenated',
            data=np.array(cs_ids, dtype=STRING_DTYPE_SPECIFIER)
        )

        g = g.create_group('property_ids')
        g.attrs['concatenated'] = True
        g = g.create_group('data', track_order=True)
        g.create_dataset(
            name='_root_concatenated',
            data=np.array(pr_ids, dtype=STRING_DTYPE_SPECIFIER)
        )

        now = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
        g.attrs['last_modified']    = now

        return ds_id

class ConcatenationException(Exception):
    pass

class InvalidGroupError(Exception):
    pass

class MissingEntryError(Exception):
    pass