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

Using multiple property definitions
===================================

This Dataset contains the common energy/forces/virial data, but also includes a
large amount of additional data/information for each calculation which can be
stored as a separate Property. Note that this information is not suitable for a
PropertySettings object because some of it dependent upon the output of a
calculation (not the input), and is therefore better suited as a Property.

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
   

.. code-block:: python

	extra_stuff_definition = {
		'property-id': 'si-prx-gap-data',
		'property-title': 'Si PRX GAP data',
		'property-description': 'A property for storing all of the additional information provided for the Si PRX GAP dataset',

		'mix_history_length':         {'type': 'float',  'has-unit': False, 'extent': [],      'required': False, 'description': ''},
		'castep_file_name':           {'type': 'string', 'has-unit': False, 'extent': [],      'required': False, 'description': ''},
		'grid_scale':                 {'type': 'float',  'has-unit': False, 'extent': [],      'required': False, 'description': ''},
		'popn_calculate':             {'type': 'bool',   'has-unit': False, 'extent': [],      'required': False, 'description': ''},
		'n_neighb':                   {'type': 'int',    'has-unit': False, 'extent': [":"],   'required': False, 'description': ''},
		'oldpos':                     {'type': 'float',  'has-unit': True,  'extent': [":",3], 'required': False, 'description': ''},
		'i_step':                     {'type': 'int',    'has-unit': False, 'extent': [],      'required': False, 'description': ''},
		'md_temperature':             {'type': 'float',  'has-unit': True,  'extent': [],      'required': False, 'description': ''},
		'positions':                  {'type': 'float',  'has-unit': True,  'extent': [":",3], 'required': False, 'description': ''},
		'task':                       {'type': 'string', 'has-unit': False, 'extent': [],      'required': False, 'description': ''},
		'data_distribution':          {'type': 'string', 'has-unit': False, 'extent': [],      'required': False, 'description': ''},
		'avg_ke':                     {'type': 'float',  'has-unit': True,  'extent': [":"],   'required': False, 'description': ''},
		'force_nlpot':                {'type': 'float',  'has-unit': True,  'extent': [":",3], 'required': False, 'description': ''},
		'continuation':               {'type': 'string', 'has-unit': False, 'extent': [],      'required': False, 'description': ''},
		'castep_run_time':            {'type': 'float',  'has-unit': True,  'extent': [],      'required': False, 'description': ''},
		'calculate_stress':           {'type': 'bool',   'has-unit': False, 'extent': [],      'required': False, 'description': ''},
		'Minim_Hydrostatic_Strain':   {'type': 'bool',   'has-unit': False, 'extent': [],      'required': False, 'description': ''},
		'avgpos':                     {'type': 'float',  'has-unit': True,  'extent': [":",3], 'required': False, 'description': ''},
		'frac_pos':                   {'type': 'float',  'has-unit': False, 'extent': [":",3], 'required': False, 'description': ''},
		'hamiltonian':                {'type': 'float',  'has-unit': False, 'extent': [],      'required': False, 'description': ''},
		'md_cell_t':                  {'type': 'float',  'has-unit': True,  'extent': [],      'required': False, 'description': ''},
		'cutoff_factor':              {'type': 'float',  'has-unit': False, 'extent': [],      'required': False, 'description': ''},
		'momenta':                    {'type': 'float',  'has-unit': False, 'extent': [":",3], 'required': False, 'description': ''},
		'elec_energy_tol':            {'type': 'float',  'has-unit': False, 'extent': [],      'required': False, 'description': ''},
		'mixing_scheme':              {'type': 'string', 'has-unit': False, 'extent': [],      'required': False, 'description': ''},
		'Minim_Lattice_Fix':          {'type': 'float',  'has-unit': False, 'extent': [9],     'required': False, 'description': ''},
		'in_file':                    {'type': 'string', 'has-unit': False, 'extent': [],      'required': False, 'description': ''},
		'travel':                     {'type': 'float',  'has-unit': False, 'extent': [":",3], 'required': False, 'description': ''},
		'thermostat_region':          {'type': 'float',  'has-unit': False, 'extent': [":"],   'required': False, 'description': ''},
		'time':                       {'type': 'float',  'has-unit': True,  'extent': [],      'required': False, 'description': ''},
		'temperature':                {'type': 'float',  'has-unit': True,  'extent': [],      'required': False, 'description': ''},
		'kpoints_mp_grid':            {'type': 'float',  'has-unit': False, 'extent': [3],     'required': False, 'description': ''},
		'gap_force':                  {'type': 'float',  'has-unit': True,  'extent': [":",3], 'required': False, 'description': ''},
		'gap_energy':                 {'type': 'float',  'has-unit': True,  'extent': [],      'required': False, 'description': ''},
		'cutoff':                     {'type': 'float',  'has-unit': True,  'extent': [],      'required': False, 'description': ''},
		'xc_functional':              {'type': 'string', 'has-unit': False, 'extent': [],      'required': False, 'description': ''},
		'smearing_width':             {'type': 'float',  'has-unit': True,  'extent': [],      'required': False, 'description': ''},
		'pressure':                   {'type': 'float',  'has-unit': True,  'extent': [],      'required': False, 'description': ''},
		'gap_virial':                 {'type': 'float',  'has-unit': True,  'extent': [9],     'required': False, 'description': ''},
		'reuse':                      {'type': 'string', 'has-unit': False, 'extent': [],      'required': False, 'description': ''},
		'fix_occupancy':              {'type': 'bool',   'has-unit': False, 'extent': [],      'required': False, 'description': ''},
		'map_shift':                  {'type': 'float',  'has-unit': False, 'extent': [":",3], 'required': False, 'description': ''},
		'md_num_iter':                {'type': 'int',    'has-unit': False, 'extent': [], 'required': False, 'description': ''},
		'damp_mask':                  {'type': 'float',  'has-unit': False, 'extent': [":"],   'required': False, 'description': ''},
		'opt_strategy':               {'type': 'string', 'has-unit': False, 'extent': [],      'required': False, 'description': ''},
		'spin_polarized':             {'type': 'bool',   'has-unit': False, 'extent': [],      'required': False, 'description': ''},
		'nextra_bands':               {'type': 'int',    'has-unit': False, 'extent': [],      'required': False, 'description': ''},
		'fine_grid_scale':            {'type': 'float',  'has-unit': False, 'extent': [],      'required': False, 'description': ''},
		'masses':                     {'type': 'float',  'has-unit': True,  'extent': [":"],   'required': False, 'description': ''},
		'iprint':                     {'type': 'int',    'has-unit': False, 'extent': [],      'required': False, 'description': ''},
		'finite_basis_corr':          {'type': 'string', 'has-unit': False, 'extent': [],      'required': False, 'description': ''},
		'enthalpy':                   {'type': 'float',  'has-unit': True,  'extent': [],      'required': False, 'description': ''},
		'opt_strategy_bias':          {'type': 'int',    'has-unit': False, 'extent': [],      'required': False, 'description': ''},
		'force_ewald':                {'type': 'float',  'has-unit': True,  'extent': [":",3], 'required': False, 'description': ''},
		'num_dump_cycles':            {'type': 'int',    'has-unit': False,  'extent': [],     'required': False, 'description': ''},
		'velo':                       {'type': 'float',  'has-unit': True,  'extent': [":",3], 'required': False, 'description': ''},
		'md_delta_t':                 {'type': 'float',  'has-unit': True,  'extent': [],      'required': False, 'description': ''},
		'md_ion_t':                   {'type': 'float',  'has-unit': True,  'extent': [],      'required': False, 'description': ''},
		'force_locpot':               {'type': 'float',  'has-unit': True,  'extent': [":",3], 'required': False, 'description': ''},
		'numbers':                    {'type': 'int',    'has-unit': False, 'extent': [":"],   'required': False, 'description': ''},
		'max_scf_cycles':             {'type': 'int',    'has-unit': False, 'extent': [],      'required': False, 'description': ''},
		'mass':                       {'type': 'float',  'has-unit': True,  'extent': [":"],      'required': False, 'description': ''},
		'Minim_Constant_Volume':      {'type': 'bool',   'has-unit': False, 'extent': [],      'required': False, 'description': ''},
		'cut_off_energy':             {'type': 'float',  'has-unit': True,  'extent': [],      'required': False, 'description': ''},
		'virial':                     {'type': 'float',  'has-unit': True,  'extent': [3,3],   'required': False, 'description': ''},
		'nneightol':                  {'type': 'float',  'has-unit': False, 'extent': [],      'required': False, 'description': ''},
		'max_charge_amp':             {'type': 'float',  'has-unit': False, 'extent': [],      'required': False, 'description': ''},
		'md_thermostat':              {'type': 'string', 'has-unit': False, 'extent': [],      'required': False, 'description': ''},
		'md_ensemble':                {'type': 'string', 'has-unit': False, 'extent': [],      'required': False, 'description': ''},
		'acc':                        {'type': 'float',  'has-unit': False, 'extent': [":",3], 'required': False, 'description': ''},
	}

In order to satisfy the formatting requirements specified by the `OpenKIM
Properties Framework <https://openkim.org/doc/schema/properties-framework/>`_,
the field names in the property defintion should not include underscores
(:code:`'_'`).

.. code-block:: python

	# Can't use underscores in field names
	extra_stuff_definition = {
		k.replace('_', '-').lower(): v for k,v in extra_stuff_definition.items()
	}

.. code-block:: python

    client.insert_property_definition(base_definition)
    client.insert_property_definition(extra_stuff_definition)

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

Some of the data fields need to be cleaned before use

.. code-block:: python

	# Data stored on atoms needs to be cleaned
	def tform(img):
		img.info['per-atom'] = False
		
		# Renaming some fields to be consistent
		if 'DFT_energy' in img.info:
			img.info['dft_energy'] = img.info['DFT_energy']
			del img.info['DFT_energy']
			
		if 'DFT_force' in img.arrays:
			img.arrays['dft_force'] = img.arrays['DFT_force']
			del img.arrays['DFT_force']
			
		if 'DFT_virial' in img.info:
			img.info['dft_virial'] = img.info['DFT_virial']
			del img.info['DFT_virial']
			
		# Converting some string values to floats
		for k in [
			'md_temperature', 'md_cell_t', 'smearing_width', 'md_delta_t',
			'md_ion_t', 'cut_off_energy', 'elec_energy_tol',
			]:
			if k in img.info:
				try:
					img.info[k] = float(img.info[k].split(' ')[0])
				except:
					pass
		
		# Reshaping shape (9,) stress vector to (3, 3) to match definition
		if 'dft_virial' in img.info:
			img.info['dft_virial'] = img.info['dft_virial'].reshape((3,3))

Now we can build the property map to tell :meth:`insert_data` how to build the
properties.

.. code-block:: python

	units = {
		'energy': 'eV',
		'forces': 'eV/Ang',
		'virial': 'GPa',
		'oldpos': 'Ang',
		'md_temperature': 'K',
		'positions': 'Ang',
		'avg_ke': 'eV',
		'force_nlpot': 'eV/Ang',
		'castep_run_time': 's',
		'avgpos': 'Ang',
		'md_cell_t': 'ps',
		'time': 's',
		'temperature': 'K',
		'gap_force': 'eV/Ang',
		'gap_energy': 'eV',
		'cutoff': 'Ang',
		'smearing_width': 'eV',
		'pressure': 'GPa',
		'gap_virial': 'GPa',
		'masses': '_amu',
		'enthalpy': 'eV',
		'force_ewald': 'eV/Ang',
		'velo': 'Ang/s',
		'md_delta_t': 'fs',
		'md_ion_t': 'ps',
		'force_locpot': 'eV/Ang',
		'mass': 'g',
		'cut_off_energy': 'eV',
		'virial': 'GPa',
	}

.. code-block:: python

	property_map = {
		'energy-forces-virial': {
			# Property Definition field: {'field': ASE field, 'units': ASE-readable units}
			'energy': {'field': 'dft_energy', 'units': 'eV'},
			'forces': {'field': 'dft_force', 'units': 'eV/Ang'},
			'virial': {'field': 'dft_virial', 'units': 'GPa'}
		},
		'si-prx-gap-data': {
			k.replace('_', '-').lower(): {'field': k , 'units': units[k] if k in units else None}
			for k in extra_stuff_definition if k not in {'property-id', 'property-title', 'property-description'}
		}
	}

Identifying duplicate configurations
====================================

Note: this dataset has four pairs of duplicate configurations. This can be seen
by counting the number of configurations that have twice as many linked
properties as expected (expected is 2).

.. code-block:: python

	client.configurations.count_documents(
		{'relationships.properties.2': {'$exists': True}}
	)

	# Output: 4

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


First, adding labels to the Property objects based on the XC-functional used.

.. code-block:: python

    client.apply_labels(
        dataset_id=ds_id, collection_name='properties',
        query={'si-prx-gap-data.xc-functional.source-value': 'PW91'},
        labels='PW91',
        verbose=True
    )

    client.apply_labels(
        dataset_id=ds_id, collection_name='properties',
        query={'si-prx-gap-data.xc-functional.source-value': 'PBE'},
        labels='PBE',
        verbose=True
    )

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
