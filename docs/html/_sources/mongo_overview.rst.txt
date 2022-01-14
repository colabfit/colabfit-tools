==============
Mongo overview
==============

MongoDB structure
=================

The structure of the underlying Mongo database is described below. Lines
preceded by a forward slash (:code:`/`) denote `Mongo collections
<https://docs.mongodb.com/manual/core/databases-and-collections/#collections>`_.
All other lines denote queryable fields. Indentation denotes sub-fields.

* :code:`/configurations`
    * :code:`_id`: a unique string identifier
    * :code:`atomic_numbers`: the atomic numbers for each atom
    * :code:`positions`: the Cartesian coordinates of each atom
    * :code:`cell`: the cell lattice vectors
    * :code:`pbc`: periodic boundary conditions along each cell vector
    * :code:`names`: human-readable names for the Configuration
    * :code:`labels`: labels applied to the Configuration to improve queries
    * :code:`elements`: the set of element types in the Configuration
    * :code:`nelements`: the number of unique element types in the Configuration
    * :code:`elements_ratios`: the relative concentrations of each element type
    * :code:`chemical_formula_reduced`: a reduced chemical formula
    * :code:`chemical_formula_anonymous`: an anonymous chemical formula (without
      specific element types)
    * :code:`chemical_formula_hill`: the chemical formula in Hill notation
    * :code:`nsites`: the number of sites (atoms) in the Configuration
    * :code:`dimension_types`: same as :code:`pbc`
    * :code:`nperiodic_dimensions`: the number of periodic dimensions
    * :code:`latice_vectors`: same as :code:`cell`
    * :code:`last_modified`: timestamp of when the entry was modified last
    * :code:`relationships`: pointers to linked entries
        * :code:`properties`: IDs of linked Properties
        * :code:`configuration_sets`: IDs of linked ConfigurationSets
* :code:`/properties`
    * :code:`_id`: a unique string identifier
    * :code:`type`: the property type
    * :code:`<property_name>`: the property, with the same name as the contents of :code:`type`
        * :code:`field_name`: the values of each field in Property definition
    * :code:`methods`: duplication of the :code:`method` field of any linked
      PropertySettings
    * :code:`labels`: duplication of the :code:`labels` field of any linked
      PropertySettings
    * :code:`last_modified`: timestamp of when the entry was modified last
    * :code:`relationships`: pointers to linked entries
        * :code:`property_settings`: IDs of linked PropertySettings
        * :code:`configurations`: IDs of linked Configurations
* :code:`/property_definitions`
    * :code:`_id`: a unique string identifier
    * :code:`definition`: the full contents of a Property definition
* :code:`/property_settings`
    * :code:`_id`: a unique string identifier
    * :code:`method`: the method used in the calculation/experiment
    * :code:`decription`: a human-readable description of the
      calculation/experiment
    * :code:`labels`: labels to improve queries
    * :code:`files`: linked files
        * :code:`file_name`: the name of the file
        * :code:`file_contents`: the contents of the file
    * :code:`relationships`: pointers to linked entries
        * :code:`properties`: IDs of linked Properties
* :code:`/configuration_sets`
    * :code:`_id`: a unique string identifier
    * :code:`description`: a human-readable description
    * :code:`last_modified`: timestamp of when the entry was modified last
    * :code:`aggregated_info`: information gathered by aggregating the corresponding fields from the linked Configurations
        * :code:`nconfigurations`
        * :code:`nsites`
        * :code:`nelements`
        * :code:`chemical_systems`
        * :code:`elements`
        * :code:`individual_elements_ratios`
        * :code:`total_elements_ratios`
        * :code:`labels`
        * :code:`labels_counts`
        * :code:`chemical_formula_reduced`
        * :code:`chemical_formula_anonymous`
        * :code:`chemical_formula_hill`
    * :code:`relationships`: pointers to linked entries
        * :code:`configurations`: IDs of linked Configurations
        * :code:`datasets`: IDs of linked Datasets
* :code:`/datasets`
    * :code:`_id`: a unique string identifier
    * :code:`name`: the name of the Dataset
    * :code:`authors`: the authors of the Dataset
    * :code:`description`: a human-readable description of the Dataset
    * :code:`links`: external inks associated with the Dataset
    * :code:`last_modified`: timestamp of when the entry was modified last
    * :code:`aggregated_info`: information gathered by aggregating the corresponding fields from the linked Configurations and Properties
        * :code:`nconfigurations`
        * :code:`nsites`
        * :code:`nelements`
        * :code:`chemical_systems`
        * :code:`elements`
        * :code:`individual_elements_ratios`
        * :code:`total_elements_ratios`
        * :code:`configuration_labels`
        * :code:`configuration_labels_counts`
        * :code:`chemical_formula_reduced`
        * :code:`chemical_formula_anonymous`
        * :code:`chemical_formula_hill`
        * :code:`nperiodic_dimensions`
        * :code:`dimension_types`
        * :code:`property_types`
        * :code:`property_types_counts`
        * :code:`property_fields`
        * :code:`property_fields_counts`
        * :code:`methods`
        * :code:`methods_counts`
        * :code:`property_labels`
        * :code:`property_labels_counts`
    * :code:`relationships`: pointers to linked entries
        * :code:`properties`: IDs of linked Properties
        * :code:`configuration_sets`: IDs of linked ConfigurationSets

Mongo usage
===========

This section provides examples on how to perform various operations on the
Database using Mongo. For more details, it is highly suggested that you visit
`the MongoDB documentation <https://docs.mongodb.com/manual/>`_.

Queries
^^^^^^^

It is extremely important to be able to understand how to formulate at least
basic Mongo queries. If you are a newcomer to Mongo, one of the best places to
start would be to look over some of the
`query tutorials from the official Mongo manual
<https://docs.mongodb.com/manual/tutorial/query-documents/>`_.

Structure
^^^^^^^^^

Recall that when opening a connection to the Database, for example with the
following code:

.. code-block:: python

    from colabfit.tools.database import MongoDatabase

    client = MongoDatabase('colabfit_database')

the :code:`client` object is a Mongo Client connected to the
:code:`'colabfit_database'` Database in a running Mongo server. This Database
will have the following collections: :code:`'configurations'`,
:code:`'properties'`, :code:`'property_settings'`, :code:`'configuration_sets'`,
and :code:`'datasets'`. which are accessible as attributes. See :ref:`MongoDB
structure` for more details.

Find one
^^^^^^^^

Get an example of a single document in a collection that satisfies the given
query.

.. code-block:: python

    # Find a Property document that is linked to the Dataset with an ID of ds_id
    client.properties.find_one({'relationships.datasets': ds_id})

Count documents
^^^^^^^^^^^^^^^

Count the number of documents in a collection.

.. code-block:: python

    # Count the number of Configurations in the Database
    client.configurations.count_documents({})

Get all documents
^^^^^^^^^^^^^^^^^

Get a list of all of the Datasets in the Database, then sort by name.

.. code-block:: python

    sorted(
        list(
            client.datasets.find({}, {'name'})
        ),
        key=lambda x: x['name'].lower()
    )


Check for multiple links
^^^^^^^^^^^^^^^^^^^^^^^^

Similar to what is done in :ref:`detecting duplicates <Detecting duplicates>`,
the :code:`'relationships'` field can be useful for finding documents that are
linked to multiple other documents.

For example, for finding how many ConfigurationSets are linked to more than one
Dataset:

.. code-block:: python

    client.configuration_sets.count_documents(
        {'relationships.datasets.1': {'$exists': True}}
    )

Get distinct fields
^^^^^^^^^^^^^^^^^^^

Get a set of all existing values of a given field:

.. code-block:: python

    # Get a list of the unique property types in the Database
    client.properties.distinct('type')

Count occurrences
^^^^^^^^^^^^^^^^^

`Aggregation pipelines <https://docs.mongodb.com/manual/aggregation/>`_ can be
extremely useful, but may be more difficult to understand for new users of
Mongo. The example below shows how to use aggregation to count the occurrences
of each Configuration label.

.. code-block:: python

	cursor = client.configurations.aggregate([
        # by default, matches to all documents in the collection
        # $unwind: create a new document, once for each value in the 'labels'
        # field
		{'$unwind': '$labels'},
        # $group: group the documents based on their label field, and count
		{'$group': {'_id': '$labels', 'count': {'$sum': 1}}}
	])

	sorted(cursor, key=lambda x: x['count'], reverse=True)

Get Datasets linked to ConfigurationSets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The example below shows how to use aggregation to obtain a list of all
ConfigurationSets in the Database, with the names of their linked Datasets.

.. code-block:: python

    cursor = client.configuration_sets.aggregate([
        # $project: only return the requested fields for each document
        {'$project': {'relationships.datasets': 1}},
        # $unwind: create a new document for each element in an array
        {'$unwind': '$relationships.datasets'},
        # $project: only return the renamed field
        {'$project': {'ds_id': '$relationships.datasets'}},
        # $lookup: pull the Dataset document with the given ID
        {'$lookup': {
            # pull from the 'datasets' collection
            'from': 'datasets',
            # match the local field 'ds_id' to the '_id' field in 'datasets'
            'localField': 'ds_id',
            'foreignField': '_id',
            # attach the Dataset document under the name 'linked_ds'
            'as': 'linked_ds'
        }},
        # $project: only return the name of the linke Dataset
        {'$project': {'ds_name': '$linked_ds.name'}}
    ])

    sorted(list(cursor), key=lambda x: x['ds_name'][0].lower())
