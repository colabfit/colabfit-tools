==========
Si PRX GAP
==========

This example will be used to highlight some of the more advanced features of the
Dataset class using the popular `Si GAP dataset <https://www.repository.cam.ac.uk/handle/1810/317974>`_.
It is suggested that you go through the :ref:`basic example <Basic example>` first. The complete
code will not be shown in this example (for the complete code, see the Jupyter
notebook at :code:`colabfit/examples/si_prx_gap.ipynb`); instead, only the additional features will be
discussed here.

Note that this example assumes that the raw data has already been downloaded
using the following commands:

.. code-block:: console

    $ mkdir si_prx_gap
    $ cd si_prx_gap && wget -O Si_PRX_GAP.zip https://www.repository.cam.ac.uk/bitstream/handle/1810/317974/Si_PRX_GAP.zip?sequence=1&isAllowed=yield
    $ cd si_prx_gap && unzip Si_PRX_GAP.zip

Loading from a file
===================

This example uses :meth:`~colabfit.tools.dataset.load_data` to load the data
from an existing Extended XYZ file. Note that the raw data includes the
:code:`config_type` field, which is used to generate the names of the loaded
Configurations. A :attr:`default_name` is also provided to handle the
configurations that do not have a :code:`config_type` field.
:code:`verbose=True` is used here since the dataset is large enough to warrant a
progress bar.

.. code-block:: python

	dataset.configurations = load_data(
		file_path='./si_prx_gap/gp_iter6_sparse9k.xml.xyz',
		file_format='xyz',
		name_field='config_type',  # key in Configuration.info to use as the Configuration name
		elements=['Si'],
		default_name=dataset.name,  # default name with `name_field` not found
		verbose=True
	)

Manually constructed ConfigurationSets
======================================

Since this dataset was manually constructed by its authors, a large amount of
additional information has been provided to better identify the Configurations
(see Table I. in `the original paper <https://journals.aps.org/prx/abstract/10.1103/PhysRevX.8.041048>`_).
In order to retain this information, we define ConfigurationSets by regex
matching on the Configuration names (see
:ref:`Building configuration sets` for more details).

.. code-block:: python

    dataset.configuration_set_regexes = {
        'isolated_atom': 'Reference atom',
        'bt': 'Beta-tin',
        'dia': 'Diamond',
        'sh': 'Simple hexagonal',
        'hex_diamond': 'Hexagonal diamond',
        'bcc': 'Body-centered-cubic',
        'bc8': 'BC8',
        'fcc': 'Face-centered-cubic',
        'hcp': 'Hexagonal-close-packed',
        'st12': 'ST12',
        'liq': 'Liquid',
        'amorph': 'Amorphous',
        'surface_001': 'Diamond surface (001)',
        'surface_110': 'Diamond surface (110)',
        'surface_111': 'Diamond surface (111)',
        'surface_111_pandey': 'Pandey reconstruction of diamond (111) surface',
        'surface_111_3x3_das': 'Dimer-adatom-stacking-fault (DAS) reconstruction',
        '111adatom': 'Configurations with adatom on (111) surface',
        'crack_110_1-10': 'Small (110) crack tip',
        'crack_111_1-10': 'Small (111) crack tip',
        'decohesion': 'Decohesion of diamond-structure Si along various directions',
        'divacancy': 'Diamond divacancy configurations',
        'interstitial': 'Diamond interstitial configurations',
        'screw_disloc': 'Si screw dislocation core',
        'sp': 'sp bonded configurations',
        'sp2': 'sp2 bonded configurations',
        'vacancy': 'Diamond vacancy configurations'
    }

Manually applied Configuration labels
=====================================

Similarly, this additional knowledge about the types of Configurations in the
dataset can be used to apply metadata labels to the Configurations, which is
useful for enabling querying over the data by future users. See
:ref:`Applying configuration labels` for more details.

.. code-block:: python

    dataset.configuration_label_regexes = {
        'isolated_atom': 'isolated_atom',
        'bt': 'a5',
        'dia': 'diamond',
        'sh': 'sh',
        'hex_diamond': 'sonsdaleite',
        'bcc': 'bcc',
        'bc8': 'bc8',
        'fcc': 'fcc',
        'hcp': 'hcp',
        'st12': 'st12',
        'liq': 'liquid',
        'amorph': 'amorphous',
        'surface_001': ['surface', '001'],
        'surface_110': ['surface', '110'],
        'surface_111': ['surface', '111'],
        'surface_111_pandey': ['surface', '111'],
        'surface_111_3x3_das': ['surface', '111', 'das'],
        '111adatom': ['surface', '111', 'adatom'],
        'crack_110_1-10': ['crack', '110'],
        'crack_111_1-10': ['crac', '111'],
        'decohesion': ['diamond', 'decohesion'],
        'divacancy': ['diamond', 'vacancy', 'divacancy'],
        'interstitial': ['diamond', 'interstitial'],
        'screw_disloc': ['screw', 'dislocation'],
        'sp': 'sp',
        'sp2': 'sp2',
        'vacancy': ['diamond', 'vacancy']
    }

Renaming Configuration fields
=============================

In order to ensure that :meth:`~colabfit.tools.dataset.Dataset.parse_data` is
able to properly parse the data, the fields in :attr:`Configuration.info` and 
:attr:`Configuration.arrays` should match those used in
:attr:`Dataset.property_map`. In the case of the Si GAP dataset, some of the
data has incorrectly labeled energy/force/virial fields, where it uses
"DFT_*" instead of "dft_*" (lowercase) like the rest of the Configurations. In
order to fix this, we use the
:meth:`~colabfit.tools.dataset.Dataset.rename_configuration_field` function:

.. code-block:: python

    dataset.rename_configuration_field('DFT_energy', 'dft_energy')
    dataset.rename_configuration_field('DFT_force', 'dft_force')
    dataset.rename_configuration_field('DFT_virial', 'dft_virial')

Filtering based on XC-functional
================================

In the Si GAP dataset, some of the data was computed using a PBE functional,
and some was computed using a PW91 functional. This information is stored in the
:code:`xc_functional` field of the :attr:`Configuration.info` array.

.. code-block:: python

    set(dataset.get_configuration_field('xc_functional'))
    # output: {'PBE', 'PW91'}

A user may want to only work with subsets of the data that were computed with
the exact same DFT settings. To facilitate this, we break the original Dataset
into three separate datasets using the
:meth:`~colabfit.tools.dataset.Dataset.filter` function (see :ref:`Filtering a
Dataset` for more details).

.. code-block:: python

    no_xc_data = dataset.filter(
        'configurations',
        lambda c: c.info.get('xc_functional', None) is None
    )

    pbe_data = dataset.filter(
        'configurations',
        lambda c: c.info.get('xc_functional', None) == 'PBE'
    )

    pw91_data = dataset.filter(
        'configurations',
        lambda c: c.info.get('xc_functional', None) == 'PW91'

    )
