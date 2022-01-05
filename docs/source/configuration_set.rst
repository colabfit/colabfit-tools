<<<<<<< HEAD
<<<<<<< HEAD
.. module:: colabfit.tools.configuration_set
=======
.. module:: colabfit.tools.configuration_sets
>>>>>>> a36a6b69ee420948c36e804cdf0331ac810ac3e3
=======
.. module:: colabfit.tools.configuration_sets
>>>>>>> a36a6b69ee420948c36e804cdf0331ac810ac3e3

=================
ConfigurationSet
=================

A ConfigurationSet defines a relationship between a group of Configurations.
This is useful when organizing a Dataset and can help future users of the
dataset better understand its contents.

The most important functionality of a ConfigurationSet is provided by its
<<<<<<< HEAD
<<<<<<< HEAD
:meth:`~colabfit.tools.configuration_set.ConfigurationSet.aggregate` method,
=======
:meth:`~colabfit.tools.configuration_sets.ConfigurationSet.aggregate` method,
>>>>>>> a36a6b69ee420948c36e804cdf0331ac810ac3e3
=======
:meth:`~colabfit.tools.configuration_sets.ConfigurationSet.aggregate` method,
>>>>>>> a36a6b69ee420948c36e804cdf0331ac810ac3e3
which accumulates critical information about its group of configurations.

Basics of Configuration Sets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The most often use-case is for a ConfigurationSet to be built by specifying the
:attr:`~colabfit.tools.dataset.Dataset.configuration_set_regexes` dictionary of
a Dataset. For more details about how to build ConfigurationSets,
see :ref:`Building configuration sets`.

<<<<<<< HEAD
<<<<<<< HEAD
.. autoclass:: colabfit.tools.configuration_set.ConfigurationSet
=======
.. autoclass:: colabfit.tools.configuration_sets.ConfigurationSet
>>>>>>> a36a6b69ee420948c36e804cdf0331ac810ac3e3
=======
.. autoclass:: colabfit.tools.configuration_sets.ConfigurationSet
>>>>>>> a36a6b69ee420948c36e804cdf0331ac810ac3e3
   :members:
   :special-members:
