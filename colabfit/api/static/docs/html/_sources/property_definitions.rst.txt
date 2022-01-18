====================
Property definitions
====================

Property definitions are necessary to make sure that your data can be loaded
properly, and to provide future users of your data with good documentation of
the computed values. In most cases one of the existing properties available from 
the `OpenKIM Property List <https://openkim.org/properties>`_ will fit your
needs, but if no suitable definition already exists in the Database you can add
your own custom definition.

Custom properties can be defined using
:meth:`~colabfit.tools.database.MongoDatabase.insert_property_definition` and
supplying a dictionary that has been formatted according to the
`KIM Properties Framework <https://openkim.org/doc/schema/properties-framework/>`_.

An example dictionary is shown below, and can be modified to fit most use cases.

.. code-block:: python

    definition = {
      "property-id": "my-custom-property",

      "property-title": "Some short title",

      "property-description": "A custom, user-provided Property Definition. See https://openkim.org/doc/schema/properties-framework/ for instructions on how to build these files.",

      "a-custom-field-name" {
        "type":         "string",
        "has-unit":     false,
        "extent":       [],
        "required":     false,
        "description":  "The description of the custom field",
      },
      "a-custom-1d-array" {
        "type":         "float",
        "has-unit":     true,
        "extent":       [":"],
        "required":     true,
        "description":  "This should be a 1D vector of floats",
      },
      "a-custom-per-atom-array" {
        "type":         "float",
        "has-unit":     true,
        "extent":       [":",3],
        "required":     true,
        "description":  "This is a 2D array of floats, where the second dimension has a length of 3",
      },
    }

A real-world example definition can be found in the :ref:`QM9 example`.
