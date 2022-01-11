=============
Basic example
=============

This example corresponds to the :code:`colabfit/examples/basic_example.ipynb`
Jupyter notebook in the GitHub repo, which can be run in Google Colab.

Initialize the database
=======================

The :class:`~colabfit.tools.database.MongoDatabase` opens a connection to a
running Mongo server and attaches to a database with the given name.

.. code-block:: python

    from colabfit.tools.database import MongoDatabase

    client = MongoDatabase('colabfit_database')

Attaching a property definition
===============================

Property definitions are data structures that are used to concretely define
material properties that are stored in the Database. Note that the
:code:`property-id` field must be unique for all property definitions in the
Database.

Ideally, an existing Property Definition from the `OpenKIM Property Definition
list <https://openkim.org/properties>`_ should be used by passing the "Property
Definition ID" listed on a Property Definition page to the
:meth:`insert_property_definition` function. For example:

.. code-block:: python

    client.insert_property_definition(
        'tag:staff@noreply.openkim.org,2014-04-15:property/bulk-modulus-isothermal-cubic-crystal-npt'
    )

However, if none of the existing properties seem appropriate, a custom property
definition can be provided by passing a valid dictionary instead. See
`the OpenKIM documentation
<https://openkim.org/doc/schema/properties-framework/>`_ for more details on how
to write a valid property definition.

.. code-block:: python

    client.insert_property_definition({
        'property-id': 'energy-forces',
        'property-title': 'A default property for storing energies and forces',
        'property-description': 'Energies and forces computed using DFT',
        'energy': {'type': 'float', 'has-unit': True, 'extent': [], 'required': True, 'description': 'Cohesive energy'},
        'forces': {'type': 'float', 'has-unit': True, 'extent': [':',3], 'required': True, 'description': 'Atomic forces'},
    })


Adding Data
===========

Configurations and Properties can be inserted into a database by passing a
list of Configurations and a property map (for parsing the data) to
:meth:`~colabfit.tools.database.MongoDatabase.insert_data`. Note that a
property definition must have been already attached to the Database (see above).

Loading Configurations
^^^^^^^^^^^^^^^^^^^^^^

Load the configurations by either manually constructing the Configurations or
using :meth:`~colabfit.tools.database.load_data`, which calls a pre-made
:class:`~colabfit.tools.converters.BaseConverter` and returns a list of
:class:`~colabfit.tools.configuration.Configuration` objects.

.. code-block:: python

    # Manually

    import numpy as np
    from ase import Atoms

    images = []
    for i in range(1, 1000):
        atoms = Atoms('H'*i, positions=np.random.random((i, 3)))

        atoms.info['_name'] = 'configuration_' + str(i)

        atoms.info['dft_energy'] = i*i
        atoms.arrays['dft_forces'] = np.random.normal(size=(i, 3))

        images.append(atoms)

Note that when using :meth:`load_data`, the file_format must be specified and a
name_field (or None) should be provided to specify the names of the loaded
configurations.

.. code-block:: python

	from ase.io import write

	write('/content/example.extxyz', images)

.. code-block:: python

	from colabfit.tools.database import load_data

	images = load_data(
		file_path='/content/example.extxyz',
		file_format='xyz',
		name_field='_name',
		elements=['H'],
		default_name=None,
	)

Defining a property_map
^^^^^^^^^^^^^^^^^^^^^^^

A property map is used to specify how parse a Property from a Configuration.
Below, we define a property map that extracts the `'energy'` and `'forces'` keys
in the `'energy-forces'` property defined above from the `'dft_energy'` and
`'dft_forces'` fields in the info and arrays attributes of a given
Configuration.

.. code-block:: python

    property_map = {
        # property name
        'energy-forces': {
            # property field: {'field': configuration info/arrays field, 'units': field units}
            'energy': {'field': 'dft_energy', 'units': 'eV'},
            'forces': {'field': 'dft_forces', 'units': 'eV/Ang'},
        }
    }


Inserting the data
==================

:meth:`~colabfit.tools.database.MongoDatabase.insert_data` takes in a list of
Configurations and adds each Configuration into the :code:`'configurations'`
collection of the Database. It also uses :code:`property_map` to parse the
Properties from each Configuration and add them into the :code:`'properties'`
collection. :meth:`~colabfit.tools.database.MongoDatabase.insert_data` will
return a list of tuples of :code:`(<configuration_id>, <property_id>)`, which
can be useful for accessing and manipulating the new data.

.. code-block:: python

    from colabfit.tools.property_settings import PropertySettings

    ids = list(client.insert_data(
        images,
        property_map=property_map,
        property_settings={
            'energy-forces': PropertySettings(
                                method='VASP',
                                description='A basic VASP calculation',
                                files=None,
                                labels=['PBE', 'GGA'],
                            ),
        },
        generator=False,
        verbose=True
    ))

    all_co_ids, all_pr_ids = list(zip(*ids))

Note that the :code:`property_settings` argument can also be used to specify a
dictionary of :class:`~colabfit.tools.property_settings.PropertySettings`
objects for providing additional metadata regarding the Properties being loaded.

Creating a ConfigurationSet
===========================

A :class:`~colabfit.tools.configuration_sets.ConfigurationSet` can be used to
create groups of configurations for organizational purposes.

First, use :meth:`~colabfit.tools.database.MongoDatabase.get_data` to extract
the :code:`_id` fields for all of the configurations with fewer than 100 atoms.
A Mongo query is passed in as the :code:`query` argument (see
:ref:`Mongo usage` for more details).

.. code-block:: python

    co_ids = client.get_data(
        'configurations',
        fields='_id',
        query={'_id': {'$in': all_co_ids}, 'nsites': {'$lt': 100}},
        ravel=True
    ).tolist()

Then use :meth:`~colabfit.tools.database.MongoDatabase.insert_configuration_set`
to add the ConfigurationSet into the Database, specifying the list of
Configuration IDs to include, and a description of the ConfigurationSet.

.. code-block:: python

    cs_id = client.insert_configuration_set(
        co_ids,
        description='Configurations with fewer than 100 atoms'
    )

Note that :meth:`insert_configuration_set` returns the ID of the inserted
ConfigurationSet, which can be used to obtain the newly-added ConfigurationSet

.. code-block:: python

    client.get_configuration_set(cs_id)['configuration_set']

    print(cs.description)


A ConfigurationSet aggregates some key information from its linked
Configurations upon insertion.

.. code-block:: python

    for k,v in cs.aggregated_info.items():
        print(k, v)

Creating a Dataset
==================

A :class:`~colabfit.tools.dataset.Dataset` can be constructed by providing a
list of ConfigurationSet and Property IDs to
:meth:`~colabfit.tools.database.MongoDatabase.insert_dataset`.

First, define two ConfigurationSets:

.. code-block:: python

    co_ids1 = client.get_data(
        'configurations',
        fields='_id',
        query={'_id': {'$in': all_co_ids}, 'nsites': {'$lt': 100}},
        ravel=True
    ).tolist()

    co_ids2 = client.get_data(
        'configurations',
        fields='_id',
        query={'_id': {'$in': all_co_ids}, 'nsites': {'$gte': 100}},
        ravel=True
    ).tolist()

    # Note: CS IDs depend upon the description, so cs_id1 will not match cs_id
    # from above
    cs_id1 = client.insert_configuration_set(co_ids1, 'Small configurations')
    cs_id2 = client.insert_configuration_set(co_ids2, 'Big configurations')

Then extract the Property IDs that are linked to the given Configurations.

.. code-block:: python

    pr_ids = client.get_data(
        'properties',
        fields='_id',
        query={
            'relationships.configurations': {'$elemMatch': {'$in':
            co_ids1+co_ids2}}
        },
        ravel=True

    ).tolist()

Finally, add the Dataset into the Database

.. code-block:: python

    ds_id = client.insert_dataset(
        cs_ids=[cs_id1, cs_id2],
        pr_ids=pr_ids,
        name='basic_example',
        authors=['ColabFit User'],
        links=['https://colabfit.openkim.org/'],
        description="This is an example dataset",
        resync=True
    )

Just as a ConfigurationSet aggregates information from a list of Configurations,
a Dataset aggregates information from a list of ConfigurationSets and a list of
Properties. Note the use of :code:`resync=True`. This ensures that the ConfigurationSets
re-aggregate all of their data from their linked Configurations before the
Dataset aggregates information from the ConfigurationSets.

.. code-block:: python

    ds = client.get_dataset(ds_id)['dataset']

    for k,v in ds.aggregated_info.items():
        print(k, v)


Applying labels to configurations
=================================

Additional metadata can be applied to individual configurations or properties
using :meth:`~colabfit.tools.database.MongoDatabase.apply_labels`. This
function queries on the specified collection and attaches the given labels to
any matching entries.

.. code-block:: python

    client.apply_labels(
        dataset_id=ds_id,
        collection_name='configurations',
        query={'nsites': {'$lt': 100}},
        labels={'small'},
        verbose=True
    )

Note the use of :code:`dataset_id=ds_id` which ensures that the labels are only
applied to the entries attached to the specified Dataset.

When extracting a ConfigurationSet whose linked Configurations have been
modified, :code:`resync=True` should be used to ensure that all necessary
information (such as Configuration labels) is re-aggregated.

.. code-block:: python

    cs = client.get_configuration_set(cs_id, resync=True)['configuration_set']

Exploring the dataset
=====================

Use :meth:`~colabfit.tools.database.MongoDatabase.get_statistics` to see basic
statistics about a selection of property fields:

.. code-block:: python

    client.get_statistics(
        # For getting statistics about all property fields on the dataset
        dataset.aggregated_info['property_fields'],
        # For getting statistics only about the properties attached to the
        # dataset
        ids=dataset.property_ids
    )

Use :meth:`~colabfit.tools.database.MongoDatabase.plot_histograms` to quickly
visualize the property fields.

.. code-block:: python

    client.plot_histograms(
        dataset.aggregated_info['property_fields'],
        ids=dataset.property_ids
    )

Applying transformations to properties
======================================

:meth:`~colabfit.tools.database.MongoDatabase.apply_transformation` can be used
to modify Properties that have already been loaded into the Database.

The :code:`update_map` argument of this function is a dictionary where the key is a
field from a property definition, and the value is a function that takes two
inputs: (1) the current value taken by the specified key, and (2) a Property
document (as stored in the Mongo database). The output of this function will be
the new value used to overwrite the existing one. Note that the Property
document will be augmented to include the linked Configuration as specified by
:code:`configuration_ids`.

.. code-block:: python

    all_co_ids, all_pr_ids = list(zip(*ids))

    # Convert to per-atom energies
    client.apply_transformation(
        dataset_id=ds_id,
        property_ids=all_pr_ids,
        update_map={
            'energy-forces.energy':
                lambda current_val, pr_doc: current_val/pr_doc['configuration']['nsites']
        },
        configuration_ids=all_co_ids,
    )

Filtering
=========

The :meth:`~colabfit.tools.database.MongoDatabase.filter_on_properties`
and :meth:`~colabfit.tools.database.MongoDatabase.filter_on_configurations`
methods can be used to filter lists of ConfigurationSets and Properties based on
arbitrary criterion. This is useful for obtaining subsets of a Dataset.

Here we show an example of using :meth:`~colabfit.tools.database.MongoDatabase.filter_on_properties`.
:meth:`~colabfit.tools.database.MongoDatabase.filter_on_configurations` works in a similar manner; see the
documentation for more details.

First, define a function to use for filtering the Properties. This function
should have a single argument which is the Property document, and should return
True if the Property should be included in the filtered data. Note that the
ConfigurationSets will be filtered by only including Configurations that are
linked to at least one Property that returned :code:`True` for the filter
function.

.. code-block:: python

	def ff(pr_doc):
		emax = np.max(np.abs(pr_doc['energy-forces']['energy']['source-value']))
		fmax = np.max(np.abs(pr_doc['energy-forces']['forces']['source-value']))
		return (emax < 100) and (fmax < 3)

Next, get the filtered ConfigurationSets and Properties.

.. code-block:: python

	clean_config_sets, clean_property_ids = client.filter_on_properties(
		ds_id,
		filter_fxn=ff,
		fields=['energy-forces.energy', 'energy-forces.forces'],
		verbose=True
	)

Add the newly-filtered ConfigurationSets into the Database

.. code-block:: python

    new_cs_ids = []
    for cs in clean_config_sets:
        if len(cs.configuration_ids):
            new_cs_ids.append(
                client.insert_configuration_set(
                    cs.configuration_ids,
                    cs.description, verbose=True
                )
            )

And finally, define a new Dataset with the filtered data

.. code-block:: python

    ds_id_clean = client.insert_dataset(
        cs_ids=new_cs_ids,
        pr_ids=clean_property_ids,
        name='basic_example_filtered',
        authors=['ColabFit'],
        links=[],
        description="A dataset generated during a basic filtering example",
        resync=True,
        verbose=True,
    )
