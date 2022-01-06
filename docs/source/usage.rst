=====
Usage
=====

Detecting duplicates
====================

This section describes the core usage of the :code:`colabfit-tools` package.

Synchronizing a Dataset
=======================
When working with a Dataset, it is important to make sure that the dataset has
been "synchronized" in order to ensure that all of the data (configuration
labels, configuration sets, aggregated metadata, ...) have been properly updated
to reflect any recent changes to the Dataset. However, this often requires
looping over (potentially large) lists of Property and Configuration objects.

A :meth:`~colabfit.tools.dataset.Dataset.resync` function is provided in order
to allow Datasets to be synchronized without performing these iterations at
unnecessary times.

**Important**: the :meth:`resync` function should be called after making any
major changes to a Dataset (e.g., loading data, or updating one of the
:ref:`regex dictionaries <Regex dictionaries>`).

Note: this function will be called automatically before many core operations
(like :meth:`~colabfit.tools.dataset.Dataset.to_markdown`).

Custom properties
=================
See :ref:`an example custom property definition <Custom property
definition example>` for how to write custom property definitions.


Regex dictionaries
================================

Datasets have four important dictionaries as attributes that make it easier to
apply metadata to subsets of their configurations. For all of these
dictionaries, the key is a string that will be used to perform regex matching on
the :attr:`ASE_NAME_FIELD` of a Configuration, and the value is a representation
of the metadata to be applied to all matching Configurations.

* :attr:`~colabfit.tools.dataset.Dataset.property_map`: key = regex, value = dictionary
* :attr:`~colabfit.tools.dataset.Dataset.configuration_label_regexes`: key = regex, value = string OR list of strings
* :attr:`~colabfit.tools.dataset.Dataset.configuration_set_regexes`: key = regex, value = string
* :attr:`~colabfit.tools.dataset.Dataset.property_settings_regexes`: key = regex, value = :class:`~colabfit.tools.property_settings.PropertySettings` instance

See below for more details of how to use each dictionary.

These dictionaries will affect what happens when
:meth:`~colabfit.tools.dataset.Dataset.resync` and/or 
:meth:`~colabfit.tools.dataset.Dataset.parse_data` are called.

See the following sections for more details about how to use each of these
dictionaries:

* :attr:`~colabfit.tools.dataset.Dataset.property_map`: :ref:`Parsing data`
* :attr:`~colabfit.tools.dataset.Dataset.configuration_label_regexes`: :ref:`Applying Configuration labels`
* :attr:`~colabfit.tools.dataset.Dataset.configuration_set_regexes`: :ref:`Building configuration sets`
* :attr:`~colabfit.tools.dataset.Dataset.property_settings_regexes`: :ref:`Attaching property settings`

Parsing data
============

The most common way to parse data is using the
:attr:`~colabfit.tools.dataset.Dataset.property_map` attribute of a Dataset
instance. A property map is used to define how a
:class:`~colabfit.tools.property.Property` should be parsed from a
:class:`~colabfit.tools.configuration.Configuration` during :meth:`parse_data`.

A property map should have the following structure:

.. code-block:: python
    
    {
        <property_name>: {
            <property_field>: {
                'field': <key_for_info_or_arrays>,
                'units': <ase_readable_units>
            }
        }
    }

See below for the definitions of each of the above keys/values:

* :attr:`<property_name>` should be one of the following:

    1. The name of an OpenKIM Property Definition from the `list of approved OpenKIM Property Definitions <https://openkim.org/properties>`_
    2. The name of a locally-defined property (see :ref:`Custom properties`)
    3. The string :code:`'default'`. Note that :code:`'default'` means that an
       existing property will be used with support for basic fields like
       'energy', 'forces', and 'stress'.
* :attr:`<property_field>` should be the name of a field from an OpenKIM Property
  Definition (see :ref:`Custom properties` for more details).
* :attr:`<key_for_info_or_arrays` should be a key for indexing the :attr:`info` or
  :attr:`arrays` dictionaries on a Configuration (see :ref:`Configuration info and
  arrays fields`)
* :attr:`'field'` is used to specify the key for extracting the property from
  :attr:`Configuration.info` or :attr:`Configuartion.arrays`.
* :attr:`'units'` should be a string matching one of the units names in
  `ase.units <https://wiki.fysik.dtu.dk/ase/ase/units.html>`_. See :ref:`Units`
  for more details.

Applying configuration labels
=============================

The most common method for applying configuration labels is to use the
:attr:`~colabfit.tools.dataset.Dataset.configuration_label_regexes`
attribute of a Dataset. It determines which labels will be applied to
configurations when :meth:`resync` is called. See :ref:`Basics of Configurations`
for more details on Configuration labels.


An example :attr:`configuration_label_regexes`:

.. code-block:: python
    
    dataset.configuration_label_regexes = {
        # apply the labels 'aimd' and '100K' to all Configurations that
        # match the pattern 'md_snap.*'
        'md_snap.*': ['aimd', '100K'],

        # apply the label 'surface' to all Configurations with 'surf' in their
        # names
        'surf' : 'surface',

        # apply the label 'fcc' to all Configurations
        '.*': 'fcc',
    }

See :meth:`~colabfit.tools.dataset.Dataset.attach_configuration_labels` for an
alternative method.

Building configuration sets
===========================

The most common method for building configuration sets is to use the
:attr:`~colabfit.tools.dataset.Dataset.configuration_set_regexes`
attribute of a Dataset. It determines which
:class:`~colabfit.tools.configuration_sets.ConfigurationSet` objects will be
constructed  when :meth:`resync` is called. See :ref:`Basics of Configuration
Sets` for more details on Configuration labels.

An example :attr:`configuration_set_regexes`:

.. code-block:: python
    
    dataset.configuration_set_regexes = {
        'aimd': "AIMD snapshots of liquid Cu at 2000K",
        'vac':  "Bulk FCC Cu supercells with a single vacancy defect",
        'surf': "Cu surface structures"
    }

See :meth:`~colabfit.tools.dataset.Dataset.define_configuration_set` for an
alternative method.

Attaching property settings
===========================

The most common method for attaching property settings is to use the
:attr:`~colabfit.tools.dataset.Dataset.property_settings_regexes`
attribute of a Dataset. It determines which
:class:`~colabfit.tools.property_settings.PropertySettings` objects will be
applied to the matching Properties when :meth:`resync` is called.
See :ref:`PropertySettings` for more details on Configuration labels.

.. code-block:: python

    from colabfit.tools.property_settings import PropertySettings

    dataset.property_settings_regexes = {
        '.*':
            PropertySettings(
                method='VASP',
                description='energy/force calculations',
                files=['/path/to/INCAR'],
                labels=['PBE', 'GGA'],
            )
    }

See :meth:`~colabfit.tools.dataset.Dataset.attach_property_settings` for an
alternative method.

Reading/writing Datasets with Markdown
======================================

One of the best ways to store and distribute a Dataset is by using a Datasets'
:meth:`~colabfit.tools.dataset.Dataset.to_markdown` and 
:meth:`~colabfit.tools.dataset.Dataset.from_markdown` functions. These functions
will write/read the data to one of the `supported file formats <Supported file
formats>`.

See :ref:`writing an input Markdown file <Writing an input Markdown file>` for
an example of how to write a Markdown file for an un-processed dataset, and the
:ref:`QM9 example` for an example of writing un-processed data out to XYZ format
after it has been loaded into a Dataset.

Data exploration
================

A Dataset's :meth:`~colabfit.tools.dataset.Dataset.plot_histograms`,
:meth:`~colabfit.tools.dataset.Dataset.get_data`,
and :meth:`~colabfit.tools.dataset.Dataset.get_statistics` functions can be
extremely useful for quickly visualizing your data and detecting outliers.

.. code-block:: python

    energies = dataset.get_data('energy', ravel=True)
    forces   = dataset.get_data('forces', concatenate=True)

.. code-block:: python
    
    # From the QM9 example

    dataset.plot_histograms([
        'a', 'b', 'c', 'mu', 'alpha', 'homo', 'lumo', 'r2', 'zpve', 'u0', 'u',
        'h', 'g', 'cv'
    ])

.. image:: qm9_histograms.png
    :align: center

.. code-block:: python
    
    # From the QM9 example

    print(dataset.get_statistics('a'))
    print(dataset.get_statistics('b'))
    print(dataset.get_statistics('c'))

    # {'average': 9.814382088508797, 'std': 1809.4589082320583, 'min': 0.0, 'max': 619867.68314, 'average_abs': 9.814382088508797}
    # {'average': 1.4060972645920002, 'std': 1.5837889998648804, 'min': 0.33712, 'max': 437.90386, 'average_abs': 1.4060972645920002}
    # {'average': 1.1249210272988013, 'std': 1.0956136904779634, 'min': 0.33118, 'max': 282.94545, 'average_abs': 1.1249210272988013}

See the :ref:`QM9 example` and the :ref:`Si PRX GAP example` to further explore
the benefits of these functions.

Filtering a Dataset
===================

Datasets can be easily filtered to remove unwanted entries or extract subsets of
interest. Filtering can be done using a Dataset's
:meth:`~colabfit.tools.dataset.Dataset.filter` function, which allows a user to
either filter on the :code:`"data"` or :code:`"configurations"` of a Dataset
using a user-provided lambda function. The lambda function will iterate over
either the list of Property objects or Configurations objects, and should return
:code:`True` if the given entry should be included in the filtered dataset.

Note: by default :meth:`filter` does _not_ copy the underlying data, so if the
data is modified in the filtered dataset, it will also alter the data in the
original dataset (and vice-versa).


.. code-block:: python

    # From the QM9 example

    clean = dataset.filter(
        'data',
        lambda p: (p['a']['source-value'] < 20) and (p['b']['source-value'] < 10),
        verbose=True
    )

Checking for subsets
====================

It may sometimes be useful to know if two Datasets are subsets of each other.
This can be helpful to avoid including duplicate data, and to properly maintain
original authorships over Datasets that are composed of one another.

To perform set operations with Datasets, use the
:meth:`colabfit.tools.dataset.Dataset.issubset`,
:meth:`colabfit.tools.dataset.Dataset.issuperset`, and
:meth:`colabfit.tools.dataset.Dataset.isdisjoint` functions. By default, these
functions work by hashing the Configuration and Property objects on a Dataset.
These functions also include the :attr:`configurations_only` option to only
compare the Configurations (not the Properties) of a Dataset.

Data transformations
====================

It is often necessary to transform the data in a Dataset in order to improve
performance when fitting models to the data, or to convert the data into a
different format. This can be done using a Dataset's
:meth:`~colabfit.tools.dataset.Dataset.apply_transform` function:

.. code-block:: python

    from colabfit.tools.transformations import (
        BaseTransform, Sequential, SubtractDivide, PerAtomEnergies
    )

    class ConvertToStress(BaseTransform):
        # For converting from energies to stresses
        def __init__(self):
            super(ScaleVirial, self).__init__(
                lambda x, c: (-np.array(x)/c.cell.volume*160.21766208)
            )

    reference_energy = -3.14159  # eV/atom

    # Keys should be property_fields as specified in Dataset.property_map
    dataset.apply_transform('stress', ConvertToStress())

    dataset.apply_transform(
        'energy',
        Sequential(
            # built-in to convert to per-atom energies
            PerAtomEnergies(),
            # subtract a reference energy
            SubtractDivide(sub=reference_energy, div=1)
        )
    )

Building parent datasets
========================

The :attr:`~colabfit.tools.dataset.Dataset.data` field of a Dataset can have two
forms:

1. a list of :class:`~colabfit.tools.property.Property` objects
2. OR a list of :class:`~colabfit.tools.dataset.Dataset` objects

In the second case, the top-level dataset is referred to as a "parent" Dataset,
and the lower-level datesets are referred to as "child" Datasets. This use-case
occurs frequently when combining Datasets from different sources.

All of the core functions of a Dataset have been written to support recursively
iterating through a tree of parent Datasets, so for the most part the behavior
of a Dataset is unchanged if it is a parent or a child. However, there are some
additional functions that can be used with a parent dataset:

* :meth:`~colabfit.tools.dataset.Dataset.attach_dataset` for attaching a child
  dataset to a parent dataset (use this function instead of modifying
  :attr:`Dataset.data`)
* :meth:`~colabfit.tools.dataset.Dataset.flatten` for flattening a nested tree
  of parent datasets into a single dataset (see
  :meth:`~colabfit.tools.dataset.Dataset.merge` for details on how the merging
  is performed)

Train/test splits
=================

A Dataset can be split into two Datasets for training/testing using the
:meth:`~colabfit.tools.dataset.Dataset.train_test_split` function:

.. code-block:: python

        train, test = train_test_split(self, train_frac)

Note that as with the :meth:`filter` function, :meth:`train_test_split` does
_not_ copy the underlying data by default.

Units
=====

:code:`colabfit-tools` defines a set of standard units for some of the common
data types:

* energy: eV
* forces: eV/Ang
* pressure: GPa

If a user wants to specify different units (for example in a 
ref:`Parsing data`), then those units should follow `the unit
conventions provided by ASE <https://wiki.fysik.dtu.dk/ase/ase/units.html>`_.

In general, whenever a function uses :code:`convert_units=True`, the units will
be converted into the defined :code:`colabfit-tools` units. Note that in order
for the units to be properly converted, they should use only ASE-readable unit
names, asterisks (:code:`*`) to denote multiplication, and forward slashes
(:code:`/`) to denote division.

Supported file formats
======================

Ideally, raw data should be stored in `Extended XYZ format
<https://wiki.fysik.dtu.dk/ase/ase/io/formatoptions.html#extxyz>`_. This is the
default format used by :code:`colabfit-tools`, and should be suitable for almost
all use cases. CFG files (used by Moment Tensor Potentials) are also supported,
but are not recommended.

Data that is in a custom format (e.g., JSON, HDF5, ...) that cannot be easily
read by
`ase.io.read <https://wiki.fysik.dtu.dk/ase/ase/io/io.html#ase.io.read>`_ will
require the use of a :class:`~colabfit.tools.converters.FolderConverter`
instance, which needs to be supplied with a custom :meth:`reader` function for
parsing the data.
