.. module:: colabfit.tools.property_settings

It is best practice to attach a :class:`PropertySettings` object to a
:class:`~colabfit.tools.property.Property` instance in order to better document
the conditions under which the property was computed. This would often include
information such as the DFT software package, a description of the calculation,
and an example file for running the calculation.

=================
PropertySettings
=================

.. autoclass:: colabfit.tools.property_settings.PropertySettings
   :members:
   :special-members:
