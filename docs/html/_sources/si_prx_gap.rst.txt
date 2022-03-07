==================
Si PRX GAP example
==================

This example will be used to highlight some of the more advanced features of the
Dataset class using the popular `Si GAP dataset <https://www.repository.cam.ac.uk/handle/1810/317974>`_.
It is suggested that you go through the :ref:`basic example <Basic example>` first. The complete
code will not be shown in this example (for the complete code, see the Jupyter
notebook at :code:`colabfit/examples/Si_PRX_GAP/si_prx_gap.ipynb`); instead, only the additional features will be
discussed here.

Note that this example assumes that the raw data has already been downloaded
using the following commands:

.. code-block:: console

    $ mkdir si_prx_gap
    $ cd si_prx_gap && wget -O Si_PRX_GAP.zip https://www.repository.cam.ac.uk/bitstream/handle/1810/317974/Si_PRX_GAP.zip?sequence=1&isAllowed=yield
    $ cd si_prx_gap && unzip Si_PRX_GAP.zip

Loading from a file
===================

This example uses :meth:`~colabfit.tools.database.load_data` to load the data
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
		default_name='Si_PRX_GAP',  # default name with `name_field` not found
		verbose=True
	)

Cleaning data
=============

Some of the Configurations loaded in by :meth:`load_data` need to be cleaned before they
are ready to be used. Specifically:

1. The `'per-atom'` field should be added to each configuration
1. Some fields are inconsistently named, using both `'-'` and `'_'`
2. Some fields need to be converted from strings to floats
3. Stress vectors should be reshaped to have size `(3, 3)`

We will address this by writing a function for modifying the Configurations
in-place.

.. code-block:: python

	# Data stored on atoms needs to be cleaned
	def tform(img):
		img.info['per-atom'] = False
		
		# Renaming some fields to be consistent
		info_items = list(img.info.items())
		
		for key, v in info_items:
			if key in ['_name', '_labels', '_constraints']:
				continue
				
			del img.info[key]
			img.info[key.replace('_', '-').lower()] = v

		arrays_items = list(img.arrays.items())
		for key, v in arrays_items:
			del img.arrays[key]
			img.arrays[key.replace('_', '-').lower()] = v
		
		# Converting some string values to floats
		for k in [
			'md-temperature', 'md-cell-t', 'smearing-width', 'md-delta-t',
			'md-ion-t', 'cut-off-energy', 'elec-energy-tol',
			]:
			if k in img.info:
				try:
					img.info[k] = float(img.info[k].split(' ')[0])
				except:
					pass
		
		# Reshaping shape (9,) stress vector to (3, 3) to match definition
		if 'dft-virial' in img.info:
			img.info['dft-virial'] = img.info['dft-virial'].reshape((3,3))
			
		if 'gap-virial' in img.info:
				img.info['gap-virial'] = img.info['gap-virial'].reshape((3,3))

The :meth:`tform` function can be passed to :meth:`insert_data` using the
:code:`transform` argument, which will call :meth:`tform` on each Configuration
before doing any additional processing.

Handling different property settings
====================================

This Dataset contains the common energy/forces/virial data, but also includes a
large amount of additional data/information for each calculation which can be
stored as PropertySettings objects. This Dataset also has energy/forces/virial
data computed using multiple methods (DFT and a trained GAP model). In this
section we will discuss how to use the :code:`property_map` argument property
with the :meth:`insert_data` function.

To begin with, we first write a property definition for storing computed
energy/forces/virial data. Note that this same definition will be used for both
the DFT-computed and the GAP-computed data.

.. code-block:: python

	base_definition = {
		'property-id': 'energy-forces-stress',
		'property-title': 'Basic outputs from a static calculation',
		'property-description':
			'Energy, forces, and stresses from a calculation of a '\
			'static configuration. Energies must be specified to be '\
			'per-atom or supercell. If a reference energy has been '\
			'used, this must be specified as well.',

		'energy': {
			'type': 'float',
			'has-unit': True,
			'extent': [],
			'required': False,
			'description':
				'The potential energy of the system.'
		},
		'forces': {
			'type': 'float',
			'has-unit': True,
			'extent': [":", 3],
			'required': False,
			'description':
				'The [x,y,z] components of the force on each particle.'
		},
		'stress': {
			'type': 'float',
			'has-unit': True,
			'extent': [3, 3],
			'required': False,
			'description':
				'The full Cauchy stress tensor of the simulation cell'
		},

		'per-atom': {
			'type': 'bool',
			'has-unit': False,
			'extent': [],
			'required': True,
			'description':
				'If True, "energy" is the total energy of the system, '\
				'and has NOT been divided by the number of atoms in the '\
				'configuration.'
		},
		'reference-energy': {
			'type': 'float',
			'has-unit': True,
			'extent': [],
			'required': False,
			'description':
				'If provided, then "energy" is the energy (either of '\
				'the whole system, or per-atom) LESS the energy of '\
				'a reference configuration (E = E_0 - E_reference). '\
				'Note that "reference-energy" is just provided for '\
				'documentation, and that "energy" should already have '\
				'this value subtracted off. The reference energy must '\
				'have the same units as "energy".'
		},
	}


We will then prepare two separate maps. One for loading any DFT-computed
properties:

.. code-block:: python

	dft_map = {
		# Property Definition field: {'field': ASE field, 'units': ASE-readable units}
		'energy': {'field': 'dft-energy', 'units': 'eV'},
		'forces': {'field': 'dft-force',  'units': 'eV/Ang'},
		'stress': {'field': 'dft-virial', 'units': 'GPa'},
		'per-atom': {'field': 'per-atom', 'units': None},
	}

And a separate one for loading GAP-computed properties:

.. code-block:: python

	gap_map = {
		# Property Definition field: {'field': ASE field, 'units': ASE-readable units}
		'energy': {'field': 'gap-energy', 'units': 'eV'},
		'forces': {'field': 'gap-force',  'units': 'eV/Ang'},
		'stress': {'field': 'gap-virial', 'units': 'GPa'},
		'per-atom': {'field': 'per-atom', 'units': None},
	}
	
Next, we will create a list of all of the fields that should be stored on a
PropertySettings object rather than on a Property:

.. code-block:: python

	settings_keys = [
		'mix-history-length',
		'castep-file-name',
		'grid-scale',
		'popn-calculate',
		'n-neighb',
		'oldpos',
		'i-step',
		'md-temperature',
		'positions',
		'task',
		'data-distribution',
		'avg-ke',
		'force-nlpot',
		'continuation',
		'castep-run-time',
		'calculate-stress',
		'minim-hydrostatic-strain',
		'avgpos',
		'frac-pos',
		'hamiltonian',
		'md-cell-t',
		'cutoff-factor',
		'momenta',
		'elec-energy-tol',
		'mixing-scheme',
		'minim-lattice-fix',
		'in-file',
		'travel',
		'thermostat-region',
		'time',
		'temperature',
		'kpoints-mp-grid',
		'cutoff',
		'xc-functional',
		'smearing-width',
		'pressure',
		'reuse',
		'fix-occupancy',
		'map-shift',
		'md-num-iter',
		'damp-mask',
		'opt-strategy',
		'spin-polarized',
		'nextra-bands',
		'fine-grid-scale',
		'masses',
		'iprint',
		'finite-basis-corr',
		'enthalpy',
		'opt-strategy-bias',
		'force-ewald',
		'num-dump-cycles',
		'velo',
		'md-delta-t',
		'md-ion-t',
		'force-locpot',
		'numbers',
		'max-scf-cycles',
		'mass',
		'minim-constant-volume',
		'cut-off-energy',
		'virial',
		'nneightol',
		'max-charge-amp',
		'md-thermostat',
		'md-ensemble',
		'acc',
	]

We will also specify any units on the fields:

.. code-block:: python

	units = {
		'energy': 'eV',
		'forces': 'eV/Ang',
		'virial': 'GPa',
		'oldpos': 'Ang',
		'md-temperature': 'K',
		'positions': 'Ang',
		'avg-ke': 'eV',
		'force-nlpot': 'eV/Ang',
		'castep-run-time': 's',
		'avgpos': 'Ang',
		'md-cell-t': 'ps',
		'time': 's',
		'temperature': 'K',
		'gap-force': 'eV/Ang',
		'gap-energy': 'eV',
		'cutoff': 'Ang',
		'smearing-width': 'eV',
		'pressure': 'GPa',
		'gap-virial': 'GPa',
		'masses': '_amu',
		'enthalpy': 'eV',
		'force-ewald': 'eV/Ang',
		'velo': 'Ang/s',
		'md-delta-t': 'fs',
		'md-ion-t': 'ps',
		'force-locpot': 'eV/Ang',
		'mass': 'g',
		'cut-off-energy': 'eV',
		'virial': 'GPa',
	}

We will also create dictionaries for constructing the DFT settings:

.. code-block:: python

	dft_settings_map = {
		k: {'field': k, 'units': units[k] if k in units else None} for k in settings_keys
	}

	dft_settings_map['_method'] = 'CASTEP'
	dft_settings_map['_description'] = 'DFT calculations using the CASTEP software'
	dft_settings_map['_files'] = None
	dft_settings_map['_labels'] = ['Monkhorst-Pack']

And the GAP settings:

.. code-block:: python

	gap_settings_map = dict(dft_settings_map)

	gap_settings_map['_method'] = 'GAP'
	gap_settings_map['_description'] = 'Predictions using a trained GAP potential'
	gap_settings_map['_files'] = None
	gap_settings_map['_labels'] = ['GAP', 'classical']

Each of these settings maps will be attached to their corresponding property
maps:

.. code-block:: python

	dft_map['_settings'] = dft_settings_map
	gap_map['_settings'] = gap_settings_map

Finally, they will both be merged into a single map, which will be passed
directly to :meth:`insert_data`:

.. code-block:: python

	property_map = {
		'energy-forces-stress': [
			dft_map,
			gap_map,
		]
	}

	ids = client.insert_data(
		images,
		property_map=property_map,
		transform=tform,
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

    configuration_set_regexes = {
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

.. code-block:: python

	cs_ids = []

	for i, (regex, desc) in enumerate(configuration_set_regexes.items()):
		co_ids = client.get_data(
			'configurations',
			fields='_id',
			query={'names': {'$regex': regex}},
			ravel=True
		).tolist()

		print(f'Configuration set {i}', f'({regex}):'.rjust(22), f'{len(co_ids)}'.rjust(7))

		cs_id = client.insert_configuration_set(co_ids, description=desc, verbose=True)

		cs_ids.append(cs_id)

Manually applied Configuration labels
=====================================

Similarly, additional knowledge provided by the authors about the types of
Configurations and Properties in the dataset can be used to apply metadata
labels to the Configurations, which is useful for enabling querying over the
data by future users. See :ref:`Applying configuration labels` for more details.

Second, applying labels to the Configurations based on author-provided
information.

.. code-block:: python

    configuration_label_regexes = {
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

.. code-block:: python

    for regex, labels in configuration_label_regexes.items():
        client.apply_labels(
            dataset_id=ds_id,
            collection_name='configurations',
            query={'names': {'$regex': regex}},
            labels=labels,
            verbose=True
        )

.. Filtering based on XC-functional
.. ================================

.. In the Si GAP dataset, some of the data was computed using a PBE functional,
.. and some was computed using a PW91 functional. This information is stored in the
.. :code:`xc_functional` field of the :attr:`Configuration.info` array.

.. .. code-block:: python

..     set(
..         client.get_data(
..             'properties',
..             'si-prx-gap-data.xc-functional',
..             ravel=True
..         )
..     )

..     # Output: {'PBE', 'PW91'}

.. A user may want to only work with subsets of the data that were computed with
.. the exact same DFT settings. To facilitate this, we break the original Dataset
.. into three separate datasets using the
.. :meth:`~colabfit.tools.dataset.Dataset.filter` function (see :ref:`Filtering a
.. Dataset` for more details).

.. .. code-block:: python

.. 	no_xc_config_sets, no_xc_pr_ids = client.filter_on_properties(
.. 		ds_id,
.. 		query={'si-prx-gap-data.xc-functional.source-value': {'$exists': False}},
.. 	)

.. 	new_cs_ids = []
.. 	for cs in no_xc_config_sets:
.. 		new_cs_ids.append(client.insert_configuration_set(cs.configuration_ids, cs.description, verbose=True))

.. 	no_xc_ds_id = client.insert_dataset(
.. 		cs_ids=new_cs_ids,
.. 		pr_ids=no_xc_pr_ids,
.. 		name='Si_PRX_GAP-no-xc',
.. 		authors=dataset.authors,
.. 		links=dataset.links,
.. 		description="A subset of the Si_PRX_GAP dataset that only contains data without a specified XC functional",
.. 		resync=True,
.. 		verbose=True,
.. 	)

.. .. code-block:: python

.. 	pbe_config_sets, pbe_pr_ids = client.filter_on_properties(
.. 		ds_id,
.. 		query={'si-prx-gap-data.xc-functional.source-value': 'PBE'},
.. 	)

.. 	new_cs_ids = []
.. 	for cs in pbe_config_sets:
.. 		if cs.configuration_ids:
.. 			new_cs_ids.append(client.insert_configuration_set(cs.configuration_ids, cs.description, verbose=True))
			
.. 	pbe_ds_id = client.insert_dataset(
.. 		cs_ids=new_cs_ids,
.. 		pr_ids=pbe_pr_ids,
.. 		name='Si_PRX_GAP-pbe',
.. 		authors=dataset.authors,
.. 		links=dataset.links,
.. 		description="A subset of the Si_PRX_GAP dataset that only contains data computed using the PBE XC functional",
.. 		resync=True,
.. 		verbose=True,
.. 	)

.. .. code-block:: python

.. 	pw91_config_sets, pw91_pr_ids = client.filter_on_properties(
.. 		ds_id,
.. 		query={'si-prx-gap-data.xc-functional.source-value': 'PW91'},
.. 	)

.. 	new_cs_ids = []
.. 	for cs in pw91_config_sets:
.. 		if cs.configuration_ids:
.. 			new_cs_ids.append(client.insert_configuration_set(cs.configuration_ids, cs.description, verbose=True))
			
.. 	pw91_ds_id = client.insert_dataset(
.. 		cs_ids=new_cs_ids,
.. 		pr_ids=pw91_pr_ids,
.. 		name='Si_PRX_GAP-pw91',
.. 		authors=dataset.authors,
.. 		links=dataset.links,
.. 		description="A subset of the Si_PRX_GAP dataset that only contains data computed using the PW91 XC functional",
.. 		resync=True,
.. 		verbose=True,
.. 	)
