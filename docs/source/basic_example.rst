=============
Basic example
=============

This example corresponds to the :code:`colabfit/examples/basic_example.ipynb`
Jupyter notebook in the GitHub repo, which can be run in Google Colab.


Creating a Dataset from scratch
===============================

Initialize the Dataset and add basic metadata.

.. code-block:: python

    from colabfit.tools.dataset import Dataset

    dataset = Dataset(name='example')

    dataset.authors = [
        'J. E. Lennard-Jones',
    ]

    dataset.links = [
        'https://en.wikipedia.org/wiki/John_Lennard-Jones'
    ]

    dataset.description = "This is an example dataset"

Adding Configurations
=====================

Load the configurations onto the Dataset by either manually constructing the
Configurations and assigning them to the :attr:`configurations` attribute, or
using :meth:`~colabfit.tools.dataset.load_data`, which calls a pre-made Converter and
returns a list of Configuration objects.

Manually
^^^^^^^^

.. code-block:: python

    import numpy as np
    from ase import Atoms

    images = []
    for i in range(1, 1000):
        atoms = Atoms('H'*i, positions=np.random.random((i, 3)))

        atoms.info['_name'] = 'configuration_' + str(i)

        atoms.info['dft_energy'] = i*i
        atoms.arrays['dft_forces'] = np.random.normal(size=(i, 3))

        images.append(atoms)

.. code-block:: python

    from colabfit.tools.configuration import Configuration

    dataset.configurations = [
        Configuration.from_ase(atoms) for atoms in images
    ]

Using :meth:`~colabfit.dataset.load_data`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Note that the file_format must be specified, and a name_field (or None) should
be provided to specify the name of the loaded configurations.

.. code-block:: python

	from ase.io import write

	write('/content/example.extxyz', images)

.. code-block:: python

	from colabfit.tools.dataset import load_data

	dataset.configurations = load_data(
		file_path='/content/example.extxyz',
		file_format='xyz',
		name_field='_name',
		elements=['H'],
		default_name=None,
	)


Applying labels to configurations
=================================

Labels can be specified as lists or single strings (which will be wrapped in a list)
Metadata can be applied to individual configurations using labels. Labels are
applied by matching a regular expression to
:code:`configuration.info[ASE_NAME_FIELD]` for each configuration. Regex
mappings are provided by setting the configuration_label_regexes dictionary.

.. code-block:: python

    dataset.configuration_label_regexes = {
        'configuration_[1-10]': 'small',
        '.*': 'random',
    }

Configuration sets using regexes
================================

A :class:`~colabfit.tools.configuration_sets.ConfigurationSet` can be used to
create groups of configurations for organizational purposes. This can be done in
a similar manner to how configuration labels are applied, but using the
:code:`configuration_set_regexes` dictionary. Note that a configuration may
exist in multiple sets at the same time.

.. code-block:: python

    dataset.configuration_set_regexes = {
        'configuration_[1-499]':   "The first configuration set",
        'configuration_[500-999]': "The second configuration set",
    }

Synchronizing the dataset
=========================

A Dataset is a pool of configurations and properties, where the configurations
are further organized by grouping them into configuration sets, and the
properties are linked to property settings. A Dataset then aggregates
information up from the configurations, properties, and property settings. In
order to ensure that the information applied by specifying
:code:`configuration_label_regexes`, :code:`configuration_set_regexes`, and
:code:`property_settings_regexes` are up-to-date, :meth:`dataset.resync()`
should be called before performing critical operations like saving a Dataset.
Some core functions will call :meth:`resync()` automatically.

.. code-block:: python

    dataset.resync()

Parsing the data
================

Parse the properties by specifying a :code:`property_map`, which is a special
dictionary on a Dataset. See :ref:`Property map` for more details.


.. code-block:: python

    dataset.property_map = {
        'default': {
            'energy': {'field': 'dft_energy', 'units': 'eV'},
            'forces': {'field': 'dft_forces', 'units': 'eV/Ang'},
        }
    }

.. code-block:: python

    dataset.parse_data(convert_units=False, verbose=True)

Exploring the data
==================

Use :meth:`~colabfit.tools.dataset.Dataset.get_data()` to obtain a list of the
given property field where each element has been wrapped in a numpy array.

.. code-block:: python

    energies = dataset.get_data('energy', ravel=True)
    forces   = dataset.get_data('forces', ravel=True)

Basic statistics can be obtained using
:meth:`~colabfit.tools.dataset.Dataset.get_statistics()`.

.. code-block:: python

    # Returns: {'average': ..., 'std': ..., 'min': ..., 'max':, ..., 'average_abs': ...}
    dataset.get_statistics('energy')

The :meth:`~colabfit.tools.dataset.Dataset.plot_histograms` function is useful
for quickly visualizing the distribution of your data.

.. code-block:: python

    dataset.plot_histograms(['energy', 'forces'])

Providing calculation metadata
==============================

Metadata for computing properties can be provided by constructing a
:class:`~colabfit.tools.property_settings.PropertySettings` object and matching
it to a property by regex matching on the property's linked configurations.
It is good practice to always attach a PropertySettings object to every
Property to improve reproducibility of the data. It is especially useful to
include example input files using the :attr:`files` field.

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
