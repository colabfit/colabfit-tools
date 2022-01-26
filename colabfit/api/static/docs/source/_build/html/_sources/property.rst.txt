.. module:: colabfit.tools.property

========
Property
========

Basics of a Property
====================

A Property stores all of the information about the output of a calculation. A
Property is usually extracted from a
:class:`~colabfit.tools.configuration.Configuration` object using the
:meth:`~colabfit.tools.dataset.Dataset.parse_data()` of a
:meth:`~colabfit.tools.dataset.Dataset` object. For more details about how this
data is stored on a Configuration, see
:ref:`Configuration info and arrays fields`. For more details on how to specify
the mapping for loading data off of a Configuration, see
:ref:`Parsing data`.

Best practice is to also include a
:class:`~colabfit.tools.property_settings.PropertySettings` object.

.. autoclass:: colabfit.tools.property.Property
   :members:
   :special-members:
