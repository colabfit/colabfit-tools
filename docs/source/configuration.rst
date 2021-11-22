.. module:: colabfit.tools.configuration

A :class:`Configuration` stores all of the information about an atomic
structure. The :class:`Configuration` class inherits from the :class:`ase.Atoms`
class, and populates some required fields in its :attr:`Atoms.info` dictionary.

The most common use-case for building :class:`Configuration` objects is to use
the :meth:`~colabfit.tools.converters.BaseConverter.load` method of a
:class:`colabfit.tools.BaseConverter` instance, which will call
:meth:`Configuration.from_ase` on an existing :class:`ase.Atoms` object.

.. code-block:: python

   from colabfit.tools.converters import EXYZConverter

   converter = EXYZConverter()

   configurations = converter.load(...)

=============
Configuration
=============

.. autoclass:: colabfit.tools.configuration.Configuration
    :members:
    :special-members: __hash__
