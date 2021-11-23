=====
Usage
=====

* have all of the special dictionaries described somewhere.

Property map
============

Note that the keys should be one of the following:
    1. The name of an OpenKIM Property Definition from the `list of approved OpenKIM Property Definitions <https://openkim.org/properties>`_
    2. The name of a locally-defined property (see :ref:`Custom properties`)
    3. The string :code:`'default'`. Note that :code:`'default'` means that an
       existing property will be used with support for basic fields like
       'energy', 'forces', and 'stress'.
       
'field' is used to specify the key for extracting the property from :meth:`Configuration.info` or :meth:`Configuartion.arrays`.
:meth:`~colabfit.tools.dataset.Dataset.parse_data()` will extract the fields
provided in property_map from the configuration, and store them as Property
objects in the dataset.data list. If a custom property is used, the
:attr:`~colabfit.tools.dataset.Dataset.custom_definitions` dictionary of a
Dataset must be updated to either point to the local EDN file or a Python
dictionary representation of the contents of the EDN file. See
:ref:`Custom properties` for more details


Applying Configuration labels
=============================

* Point to Si_PRX_GAP for an example of how to add configuration labels and
  build configuration sets.

Custom properties
=================

* Point to the QM9 for an example of how to use custom properties
* Update the link in basic_example to piont to this section

Reading/writing Datasets with Markdown
======================================


Visualization
=============


Filtering a Dataset
===================


Checking for subsets
====================

Data transformation
===================

Train/test splits
=================

Loading data
============

* XYZ, CFG, JSON, HDF5

Building configuration sets
===========================
