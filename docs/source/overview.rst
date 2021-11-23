========
Overview
========

Dataset Standard
================

The :class:`~colabfit.tools.dataset.Dataset` class was designed to be able to
be as flexible as possible to incorporate different types of data
(computational/experimental) while also making the datasets efficient to query,
store, and manipulate. A :class:`~colabfit.tools.dataset.Dataset` is constructed using five core data
structures:

* :class:`~colabfit.tools.configuration.Configuration` (CO):
    The information necessary to uniquely define the input to a calculation
    (e.g., atomic types, positions, lattice vectors, constraints, ...)
* :class:`~colabfit.tools.property.Property` (PR):
      The outputs from a calculation (e.g., DFT-computed
      energies/forces/stresses)
* :class:`~colabfit.tools.property_settings.PropertySettings` (PSO):
      Additional metadata that useful for setting up the calculation (e.g.,
      software version, input files, ...)
* :class:`~colabfit.tools.configuration_sets.ConfigurationSet` (CS):
      An object defining a group of
      :class:`~colabfit.tools.configuration.Configuration` instances and
      providing useful metadata for organizing datasets (e.g., "Snapshots from a
      molecular dynamics run at 1000K")
* :class:`~colabfit.tools.dataset.Dataset` (DS):
      The top-level dataset, which aggregates information up from the
      configuration sets and sub-datasets for improved discoverability

.. figure:: ds_diagram.svg
    :align: center
    :alt: A diagram showing the relationship between the five core data structures

    A diagram showing the relationship between the five core data structures
