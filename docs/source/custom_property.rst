==================================
Custom property definition example
==================================

If you are working with properties that:

    1. Are not part of the core properties like "energy", "force", "stress", and "charge"
    2. AND none of the existing properties available from the `OpenKIM Property List <https://openkim.org/properties>`_ fit your needs

Then it will be necessary to write a custom property definition to make sure
that a Dataset is able to properly read your data, and to provide future users
of your data with good documentation of the computed values.

Custom properties can be defined by writing an EDN file that has been formatted
according to the
`KIM Properties Framework <https://openkim.org/doc/schema/properties-framework/>`_,
or by using a Python dictionary with the same format.

An example EDN file (which could also be stored as a Python dictionary) is
shown below, and can be modified to fit most use cases.

.. code-block:: markdown

    {
      "property-id" "my-custom-property"

      "property-title" "A custom, user-provided Property Definition. See https://openkim.org/doc/schema/properties-framework/ for instructions on how to build these files."

      "property-description" "Some human-readable description"

      "a-custom-field-name" {
        "type"         "string"
        "has-unit"     false
        "extent"       []
        "required"     false
        "description"  "The description of the custom field"
      }
      "a-custom-1d-array" {
        "type"         "float"
        "has-unit"     true
        "extent"       [":"]
        "required"     true
        "description"  "This should be a 1D vector of floats"
      }
      "a-custom-per-atom-array" {
        "type"         "float"
        "has-unit"     true
        "extent"       [":",3]
        "required"     true
        "description"  "This is a 2D array of floats, where the second dimension has a length of 3"
      }
    }


Note that when using a custom property definition, the
:attr:`~colabfit.tools.dataset.Dataset.custom_definitions` dictionary on a
Dataset must be updated:

.. code-block:: python

    dataset.custom_definitions['my-custom-definition'] = '/path/to/EDN/file'

    # OR

    my_custom_definition = ...  # the dictionary version of an EDN file
    dataset.custom_definitions['my-custom-definition'] = my_custom_definition

A real-world example definition can be found in the :ref:`QM9 example`
