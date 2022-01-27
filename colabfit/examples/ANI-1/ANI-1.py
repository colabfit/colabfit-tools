import os
import numpy as np
import sys
from ase import Atoms

from colabfit.tools.database import MongoDatabase, load_data

client = MongoDatabase('test', nprocs=1, drop_database=True)

my_path_to_pyanitools = '/colabfit/data/AL_Al'
sys.path.append(my_path_to_pyanitools)


import pyanitools as pya
import glob

from colabfit import ATOMS_NAME_FIELD, ATOMS_LABELS_FIELD
from tqdm import tqdm
from colabfit.tools.configuration import Configuration
import h5py

def reader(path):
        
    adl = pya.anidataloader(path)

    n = 0
    for data in adl:
        smiles = ''.join(data['smiles'])
        
        progress_bar = tqdm(range(data['coordinates'].shape[0]), 'Loading configurations')
        for ai in progress_bar:
            atoms = Configuration(symbols=data['species'], positions=data['coordinates'][ai])

            atoms.info['_name'] = [
                '{}{}_configuration_{}'.format(path, data['path'], ai),
                smiles
            ]
            atoms.info['energy'] = data['energies'][ai]
            atoms.info['smiles'] = smiles

            yield atoms
            
            n += 1
            progress_bar.set_description('Loaded {} configurations'.format(n))
            
            
base_definition = {
    'property-id': 'energy-forces-stress',
    'property-title': 'A default property for storing energies, forces, and stress',
    'property-description': 'Energies and forces computed using DFT',

    'energy': {'type': 'float', 'has-unit': True, 'extent': [],      'required': False, 'description': 'Cohesive energy'},
    'forces': {'type': 'float', 'has-unit': True, 'extent': [':',3], 'required': False, 'description': 'Atomic forces'},
    'stress': {'type': 'float', 'has-unit': True, 'extent': [':',3], 'required': False, 'description': 'Stress'},
}

smiles_definition = {
    'property-id': 'smiles',
    'property-title': 'SMILES',
    'property-description': 'A SMILES string of a molecule',

    'smiles': {'type': 'string', 'has-unit': False, 'extent': [], 'required': True, 'description': 'SMILES string'},
}

client.insert_property_definition(base_definition)
client.insert_property_definition(smiles_definition)

property_map = {
    'energy-forces-stress': {
        # Property Definition field: {'field': ASE field, 'units': ASE-readable units}
        'energy': {'field': 'energy', 'units': 'Hartree'},
    },
    'smiles': {
        'smiles': {'field': 'smiles', 'units': None}
    }
}

from colabfit.tools.property_settings import PropertySettings

pso = PropertySettings(
    method='Gaussian09',
    description='ANI-1 property settings calculation',
    files=[],
    labels=['DFT', 'wb97x', '6-31G(d)'],
)

images = load_data(
    file_path='/colabfit/data/ANI-1_release',
    file_format='folder',
    name_field='_name',  # key in Configuration.info to use as the Configuration name
    elements=['C', 'H', 'N', 'O'],    # order matters for CFG files, but not others
    default_name='ani-1',  # default name with `name_field` not found
    reader=reader,
    glob_string='*.h5',
#     verbose=True
)

ids = list(client.insert_data(
    images,
    property_map=property_map,
    property_settings={'energy-forces-stress': pso, 'smiles': pso},
    generator=True,
    verbose=True
))