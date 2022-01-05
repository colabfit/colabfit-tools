#!/usr/bin/env python
# coding: utf-8

# This notebook serves as an example of how to load and manipulate the [ANI-1 dataset](https://github.com/isayev/ANI1_dataset) using a `Dataset` object.

# In[1]:


import os
import numpy as np

from ase import Atoms


# # Initialize the database

# In[2]:


from colabfit.tools.database import MongoDatabase, load_data

client = MongoDatabase('colabfit_database', nprocs=1, drop_database=True)


# # Setup

# This dataset uses the [ANI-1 format](https://github.com/isayev/ANI1_dataset) for loading. Before running this example, you should make sure that [pyanitools.py](https://github.com/isayev/ANI1_dataset/blob/master/readers/lib/pyanitools.py) is in `PYTHONPATH` so that you can use it for loading from the ANI-formatted HDF5 files.

# In[3]:


import sys

my_path_to_pyanitools = '../../../svreg_data/AlZnMg/AL_Al/'
sys.path.append(my_path_to_pyanitools)


# # Custom reader
# 
# Since ANI-1 is not stored in one of the core file formats, a user-specified `reader` function must be provided to `load_data` in order to read the data.

# In[4]:


import pyanitools as pya
import glob

base_path = '/home/jvita/scripts/colabfit/data/ANI-1_release/'

files = sorted(glob.glob(base_path + '*.h5'))

counter = 0
all_paths = []
for fpath in files:
    print(fpath)
    adl = pya.anidataloader(fpath)

    for data in adl:
        for i in range(data['coordinates'].shape[0]):
            all_paths.append((data['path'], i))
        
        counter += data['coordinates'].shape[0]
        print(data['path'], counter)


# In[5]:


len(all_paths)


# In[6]:


chunked_paths = [all_paths[i*10000:(i+1)*10000] for i in range(int(np.ceil(len(all_paths)/10000)))]
print(len(chunked_paths[0]))
sum(len(c) for c in chunked_paths)


# # Data loading

# In[13]:


from colabfit import ATOMS_NAME_FIELD, ATOMS_LABELS_FIELD
from tqdm import tqdm
from colabfit.tools.configuration import Configuration
import h5py

def reader(
    list_of_hdf5_paths,
    **kwargs,
    ):
    
    # Get the s0* number for the first file
    fi = list_of_hdf5_paths[0][0][9]
    path_template = '/home/jvita/scripts/colabfit/data/ANI-1_release/ani_gdb_s0{}.h5'
    current_file = path_template.format(fi)
    short_name = os.path.split(current_file)[-1]
    h5pyFile = h5py.File(current_file, 'r')
    
    for atoms_path, ai in tqdm(
        list_of_hdf5_paths,
        desc='Loading configurations',
        disable=True
        ):
        # Check if configuration is is new file
        if atoms_path[9] != fi:
            # Close the current file
            h5pyFile.close()
            
            # And open a new file
            fi = atoms_path[9]
            current_file = path_template.format(fi)
            short_name = os.path.split(current_file)[-1]
            h5pyFile = h5py.File(current_file, 'r')
            
        # Extract configuration
        data = h5pyFile[atoms_path]
        
        atoms = Atoms(symbols=data['species'].asstr()[()], positions=data['coordinates'][ai][()])

        atoms.info['_name'] = '{}{}_configuration_{}'.format(short_name, atoms_path, ai)
        atoms.info['energy'] = data['energies'][ai][()]
        atoms.info['smiles'] = ''.join(data['smiles'].asstr()[()])
        atoms.info['_labels'] = set()

        yield Configuration.from_ase(atoms)
        
    # Close the last file
    h5pyFile.close()


# In[14]:


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


# In[15]:


client.insert_property_definition(base_definition)
client.insert_property_definition(smiles_definition)


# In[16]:


property_map = {
    'energy-forces-stress': {
        # Property Definition field: {'field': ASE field, 'units': ASE-readable units}
        'energy': {'field': 'energy',    'units': 'Hartree'},
    },
    'smiles': {
        'smiles': {'field': 'smiles', 'units': None}
    }
}


# In[17]:


from colabfit.tools.property_settings import PropertySettings

pso = PropertySettings(
    method='Gaussian09',
    description='ANI-1 property settings calculation',
    files=[],
    labels=['DFT', 'wb97x', '6-31G(d)'],
)


# In[18]:


import multiprocessing as mp
from functools import partial

pool = mp.Pool(6)

user = 'colabfitAdmin'
pwrd = 'Fo08w3K&VEY&'
mongo_login = 'mongodb://{}:{}@localhost:27017/'.format(user, pwrd)

def wrapper(chunk):
    configurations = reader(chunk)

    client._insert_data(
        configurations=configurations,
        mongo_login=mongo_login,
        database_name=client.database_name,
        property_map=property_map,
        property_settings={'energy-forces-stress': pso, 'smiles': pso},
        verbose=True
    )

pool = mp.Pool(6)

pool.map(wrapper, chunked_paths)
