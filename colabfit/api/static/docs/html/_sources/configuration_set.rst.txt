.. module:: colabfit.tools.configuration_set

=================
ConfigurationSet
=================

A ConfigurationSet defines a relationship between a group of Configurations.
This is useful when organizing a Dataset and can help future users of the
dataset better understand its contents.

The most important functionality of a ConfigurationSet is provided by its
:meth:`~colabfit.tools.configuration_set.ConfigurationSet.aggregate` method,
which accumulates critical information about its group of configurations.

Basics of Configuration Sets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The most often use-case is for a ConfigurationSet to be built by specifying the
:attr:`~colabfit.tools.dataset.Dataset.configuration_set_regexes` dictionary of
a Dataset. For more details about how to build ConfigurationSets,
see :ref:`Building configuration sets`.

.. autoclass:: colabfit.tools.configuration_set.ConfigurationSet
   :members:
   :special-members:
