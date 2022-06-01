.. module:: colabfit.tools.configuration

=============
Configuration
=============

Configuration info and arrays fields
=====================================================

Configurations inherit directly from `ase.Atoms
<https://wiki.fysik.dtu.dk/ase/ase/atoms.html?highlight=self%20arrays#the-atoms-object>`_ objects.
Because of this, a Configuration can use its :attr:`info` and :attr:`arrays`
dictionaries to store a large variety of information about the inputs/outputs
and metadata of a calculation.

* The :attr:`Configuration.info` dictionary uses strings as keys, and stores any
  arbitrary information about the Configuration as a whole (e.g., its name,
  computed energy, metadata labels, ...)
* The :attr:`Configuration.arrays` dictionary uses strings as keys, and stores
  arrays of per-atom data (e.g., atomic numbers, positions, computed forces,
  ...)

Basics of Configurations
========================

A :class:`Configuration` stores all of the information about an atomic
structure, i.e., the input to a calculation. The :class:`Configuration` class
inherits from the :class:`ase.Atoms` class, and populates some required fields
in its :attr:`Atoms.info` dictionary.

Configurations have four required fields, that are automatically created when a
Configuration is instantiated:

* :attr:`~colabfit.ASE_ID_FIELD` = :code:`"_id"`:
    An `ObjectId <https://pymongo.readthedocs.io/en/stable/api/bson/objectid.html>`_
    that is created when the Configuration is instantiated, then never changed.
    Useful for uniquely identifying the Configuration when stored in a database.
* :attr:`~colabfit.ASE_NAME_FIELD` = :code:`"_name"`:
    An arbitrary string, often specified by the creator of the Configuration. This
    should usually be a human-readable string.
* :attr:`~colabfit.ASE_LABELS_FIELD` = :code:`"_labels"`:
    A set of short strings used for providing queryable metadata about the
    Configuration.
* :attr:`~colabfit.ASE_CONSTRAINTS_FIELD` = :code:`"_constraints"`:
    A set of strings that can be used as keys for :attr:`Configuration.info` or
    :attr:`Configuration.arrays`, specifying that the given fields should be
    considered as additional inputs to a calculation

Building Configurations
=======================

The most common way a :class:`Configuration` object is built is by loading data
files using one of the Converter objects provided in this package. For example,
if working with an Extended XYZ file containing multiple structures, a list of
Configurations can be created as below:

.. code-block:: python

   from colabfit.tools.converters import EXYZConverter

   converter = EXYZConverter()

   configurations = converter.load(...)

or, equivalently, using the
:meth:`~colabfit.tools.database.load_data` function.

.. code-block:: python
    
    from colabfit.tools.database import load_data

    configurations = load_data(...)

.. autoclass:: colabfit.tools.configuration.Configuration
    :members:
    :special-members:

.. autoclass:: colabfit.tools.configuration.BaseConfiguration
    :members:
    :special-members:
    :exclude-members: __weakref__

.. autoclass:: colabfit.tools.configuration.AtomicConfiguration
    :members:
    :special-members:
