import h5py
import numpy as np

from colabfit import (
    MAX_STRING_LENGTH,
    STRING_DTYPE_SPECIFIER
)
from colabfit.tools.configuration import Configuration


class Database(h5py.File):
    """
    A Database extends a PyTables (HDF5) file, but provides additional
    functionality for construction, filtering, exploring, ...

    In general, a Database works by grouping the data (coordinates, lattice
    vectors, property fields, ...), maintaining columns of IDs to identify which
    data corresponds to each Configuration/Property/etc., and providing
    additional columns of IDs for mapping from the data to any linked entries
    (configurations, properties, configuration sets, or property settings).

    The underlying PyTables filesystem has the following structure:

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

            /info
                This is all of the content stored in the
                :attr:`Configuration.info` dictionaries.

                /info_field_1
                    .attrs
                        concatenated
                            A boolean indicating if /data has been concatenated
                            or not
                    /data
                        The :code:`<info_field_1>` data for all Configurations
                        in the dataset.

                        If possible, the data will be concatenated together
                        using something similar to:

                        ..code-block:: python

                            np.concatenate(
                                [c.info[<field_name>] for c in configurations],
                                axis=0
                            )

                        In this case, /data will be a multi-dimensional array
                        where the first dimension is along configurations.

                        If the data cannot be concatenated together, then /data
                        will be a group of individual arrays indexed by their
                        corresponding configuration IDs
                    /slices
                        A group where /slices/config_id_1 specifies how to slice
                        /data in order to get the data corresponding to
                        config_id_1.

                        Regardless of if 

                        If /info_field_1.attrs['concatenated'] == False:
                            
                            Then /slices/config_id_1 will just be config_id_1.

                        If /info_field_1.attrs['concatenated'] == True:

                            Then /slices/config_id_1 will be an integer array.

                        Regardless of if /data has been concatenated or not,
                        the data for config_id_1 can be extracted using:

                        ..code-block:: python

                            group = database['configurations/info/info_field_1']
                            data = group['data'][group['slices']]
                
                        Some things to note:

                            * If the given info field CAN be concatenated as
                            described in /data, then the IDs in /slices
                            will be duplicated based on how many rows each
                            Configuration contributes.
                            * If the given info field can't be concatenated as
                            described in /data, then /data will be a simple
                            list of arrays, so /slices will just map an
                            entire array to a single Configuration.
                            * This use of lists of IDs is favorable because it
                            allows for quick filtering on /data and returning
                            the names of all associated IDs.
                .
                .
                .
                /info_field_I
            /arrays
                This group is structured in the same way as /info, but is used
                for the :attr:`Configuration.arrays` fields instead
            /constraints
                This group is not supported yet. It will be added in the future,
                and will have a similar structure to /info and /arrays
        /properties
            This group stores all of the computed properties.

            /ids
                As in /configurations/ids, but for the Property IDs

            /property_1
                .attrs
                    definition
                        An OpenKIM Property Definition in dictionary format

                /field_1
                    /data
                        As in /configurations/info/info_field_1/data
                    /indices
                        As in /configurations/info/info_field_1/indices, but
                        mapping to the Property IDs
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
                .attributes
                    All of the fields of a PropertySettings object
            .
            .
            .
            /property_settings_id_S
        /configuration_sets
            The configuration sets defining groups of configurations.

            /configuration_set_id_1
                /ids
                    A list of Configuration IDs that belong to the configuration
                    set. Useful for indexing /configurations/info fields,
                    /configurations/arrays fields, and /properties/property
                    fields.
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
        g.create_group('ids/data', track_order=True)
        g.create_group('info')
        g.create_group('arrays')

        g = self.create_group('properties')
        g.create_group('ids/data', track_order=True)

        self.create_group('property_settings')
        self.create_group('configuration_sets')
        self.create_group('datasets')


    def add_property_definition(self, definition):
        pass

    def add_configurations(self, configurations, concatenate=False):
        """
        Adds the configurations into the database, concatenating each of the
        fields in their info/arrays dictionaries whenever possible.
        """

        """
        Things to worry about:

        * This should work even if you can't hold all of the configurations in
        memory at the same time

        * Having to resize arrays might slow things down by a lot

        Pseudocode:
            For each configuration
                For each of its info fields
                    Attach the field as a new dataset in /configurations/info
                For each of its arrays fields
                    Attach the field as a new dataset in /configurations/arrays

            If concatenate==True:
                For each field in /configurations/info
                    Try to concatenate_groups

                For each field in /configurations/arrays
                    Try to concatenate_groups
        """

        for atoms in configurations:
            config_id = str(hash(atoms))

            # Save the ID
            g = self[f'configurations/ids/data']
            g.create_dataset(
                name=config_id,
                shape=1,
                data=np.array(config_id, dtype=STRING_DTYPE_SPECIFIER)
            )

            for k, v in atoms.info.items():
                g = self['configurations/info'].require_group(k)
                # Extract data and slices groups
                if 'data' in g:
                    g_data = g['data']
                else:
                    g_data = g.create_group('data', track_order=True)

                if 'slices' in g:
                    g_slices = g['slices']
                else:
                    g_slices = g.create_group('slices', track_order=True)

                if isinstance(v, str):
                    v = np.atleast_1d(v).astype(STRING_DTYPE_SPECIFIER)
                if isinstance(v, set):
                    # These should always be sets of strings
                    v = np.atleast_1d(list(v)).astype(STRING_DTYPE_SPECIFIER)
                else:
                    v = np.atleast_1d(v)

                # Add in info fields
                g_data.create_dataset(
                    name=config_id,
                    shape=v.shape,
                    dtype=v.dtype,
                    data=v,
                    maxshape=(None,)+v.shape[1:],
                )

                g_slices[config_id] = config_id

                g.attrs['concatenated'] = False

            for k, v in atoms.arrays.items():
                g = self['configurations/arrays'].require_group(k)
                if 'data' in g:
                    g_data = g['data']
                else:
                    g_data = g.create_group('data', track_order=True)

                if 'slices' in g:
                    g_slices = g['slices']
                else:
                    g_slices = g.create_group('slices', track_order=True)

                # Add in arrays fields
                g_data.create_dataset(
                    name=config_id,
                    shape=v.shape,
                    dtype=v.dtype,
                    data=v,  # should already be an array
                    maxshape=(None,)+v.shape[1:],
                )

                g_slices[config_id] = config_id

                g.attrs['concatenated'] = False

        # IDs should always be concatenated
        self.concatenate_group(self['configurations/ids'])

        if concatenate:
            for g in self['configurations/info']:
                try:
                    self.concatenate_group(g)
                except Exception as e:
                    print(e)

            for g in self['configurations/arrays']:
                    try:
                        self.concatenate_group(g)
                    except Exception as e:
                        print(e)


    def concatenate_group(self, group, chunks=None):
        """
        Attempt to concatenate all of the datasets in a group. Raise an
        exception if the datasets in the group have incompatible shapes.
        """

        if isinstance(group, str):
            group = self[group]

        # Just initialize an array of the right size
        example = next(iter(group['data'].values()))

        n = sum((_.shape[0] for _ in group['data'].values()))


        bigshape = (n,) + example.shape[1:]
        data = group.create_dataset(
            name=None, shape=bigshape, dtype=example.dtype, chunks=chunks
        )

        start = 0
        problem_adding = False

        # TODO: need to figure out how to handle the case where the field has
        # already been concatenated before.

        for ds in group['data'].values():
            try:
                # Handle IDs differently? or give them slices

                # Copy the data over
                stop = start + ds.shape[0]
                data[start:stop] = ds

                if 'slices' in group:
                    # Some groups (like IDs) don't need to be sliced

                    if ds.name is not '_root_concatenated':
                        # Ignore the one that's already been concatenated
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
            group['data']['_root_concatenated'] = data
            group.attrs['concatenated'] = True


    def get_configuration(self, i):
        return self.get_configurations([i])

    
    def get_configurations(self, ids):
        """
        A generator that returns in-memory Configuration objects one at a time
        by loading the info/array fields

        Args:

            ids (list):
                A list of string IDs specifying which Configurations to return.
        """

        for co_id in ids:
            co = Configuration()

            g = self['configuration/info']
            for field in g:
                if g[field].attrs['concatenated']:
                    # It's all one big dataset
                    pass

                co.info[field] = g[f'{field}/data'][()]
                co.info.setdefault(field, )


    def parse_data(self):
        # Move the data off of the Configurations and into the Properties
        # NOTE: this should just mean moving fields out of /configurations/info
        # or /configurations/arrays and into /properties/property_id_xxx

        pass

class ConcatenationException(Exception):
    pass