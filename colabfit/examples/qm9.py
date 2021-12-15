#!/usr/bin/env python
# coding: utf-8

# This notebook serves as an example of how to load and manipulate the [QM9 dataset](https://figshare.com/collections/Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978904) using a `Dataset` object.


import os
import numpy as np

from mpi4py import MPI

from ase import Atoms

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# # Initialize the `HDF5Client`

# In[2]:


from colabfit.tools.client import HDF5Client

client = HDF5Client('profiling_qm9.hdf5', 'w', driver='mpio',
        comm=comm,
        drop_mongo=True,
        libver='latest')


# # Data loading

# ## Define the properties and reader functions

# In[3]:


client.insert_property_definition({
    'property-id': 'qm9-property',
    'property-title': 'A, B, C, mu, alpha, homo, lumo, gap, r2, zpve, U0, U, H, G, Cv',
    'property-description': 'Geometries minimal in energy, corresponding harmonic frequencies, dipole moments, polarizabilities, along with energies, enthalpies, and free energies of atomization',
    'a':     {'type': 'float', 'has-unit': True, 'extent': [], 'required': True, 'description': 'Rotational constant A'},
    'b':     {'type': 'float', 'has-unit': True, 'extent': [], 'required': True, 'description': 'Rotational constant B'},
    'c':     {'type': 'float', 'has-unit': True, 'extent': [], 'required': True, 'description': 'Rotational constant C'},
    'mu':    {'type': 'float', 'has-unit': True, 'extent': [], 'required': True, 'description': 'Dipole moment'},
    'alpha': {'type': 'float', 'has-unit': True, 'extent': [], 'required': True, 'description': 'Isotropic polarizability'},
    'homo':  {'type': 'float', 'has-unit': True, 'extent': [], 'required': True, 'description': 'Energy of Highest occupied molecular orbital (HOMO)'},
    'lumo':  {'type': 'float', 'has-unit': True, 'extent': [], 'required': True, 'description': 'Energy of Lowest occupied molecular orbital (LUMO)'},
    'gap':   {'type': 'float', 'has-unit': True, 'extent': [], 'required': True, 'description': 'Gap, difference between LUMO and HOMO'},
    'r2':    {'type': 'float', 'has-unit': True, 'extent': [], 'required': True, 'description': 'Electronic spatial extent'},
    'zpve':  {'type': 'float', 'has-unit': True, 'extent': [], 'required': True, 'description': 'Zero point vibrational energy'},
    'u0':    {'type': 'float', 'has-unit': True, 'extent': [], 'required': True, 'description': 'Internal energy at 0 K'},
    'u':     {'type': 'float', 'has-unit': True, 'extent': [], 'required': True, 'description': 'Internal energy at 298.15 K'},
    'h':     {'type': 'float', 'has-unit': True, 'extent': [], 'required': True, 'description': 'Enthalpy at 298.15 K'},
    'g':     {'type': 'float', 'has-unit': True, 'extent': [], 'required': True, 'description': 'Free energy at 298.15 K'},
    'cv':    {'type': 'float', 'has-unit': True, 'extent': [], 'required': True, 'description': 'Heat capacity at 298.15 K'},
    'smiles-relaxed':    {'type': 'string', 'has-unit': False, 'extent': [], 'required': True, 'description': 'SMILES for relaxed geometry'},
    'inchi-relaxed':     {'type': 'string', 'has-unit': False, 'extent': [], 'required': True, 'description': 'InChI for relaxed geometry'},
})


# In[4]:


client.get_property_definition('qm9-property')['definition']


# In[5]:


from colabfit.tools.property_settings import PropertySettings

pso_id = client.insert_property_settings(
    PropertySettings(
        method='DFT/B3LYP/6-31G(2df,p)',
        description='QM9 property settings calculation',
        files=[],
        labels=['DFT', 'B3LYP', '6-31G(2df,p)'],
    )
)

pso_id


# ## Defining a `property_map`

# In[6]:


property_map = {
    'qm9-property': {
        # Property Definition field: {'field': ASE field, 'units': ASE-readable units}
        'a':     {'field': 'A',     'units': 'GHz'},
        'b':     {'field': 'B',     'units': 'GHz'},
        'c':     {'field': 'C',     'units': 'GHz'},
        'mu':    {'field': 'mu',    'units': 'Debye'},
        'alpha': {'field': 'alpha', 'units': 'Bohr*Bohr*Bohr'},
        'homo':  {'field': 'homo',  'units': 'Hartree'},
        'lumo':  {'field': 'lumo',  'units': 'Hartree'},
        'gap':   {'field': 'gap',   'units': 'Hartree'},
        'r2':    {'field': 'r2',    'units': 'Bohr*Bohr'},
        'zpve':  {'field': 'zpve',  'units': 'Hartree'},
        'u0':    {'field': 'U0',    'units': 'Hartree'},
        'u':     {'field': 'U',     'units': 'Hartree'},
        'h':     {'field': 'H',     'units': 'Hartree'},
        'g':     {'field': 'G',     'units': 'Hartree'},
        'cv':    {'field': 'Cv',    'units': 'cal/mol/K'},
        'smiles-relaxed': {'field': 'SMILES_relaxed', 'units': None},
        'inchi-relaxed': {'field': 'SMILES_relaxed',  'units': None},
    }
}


# In[7]:


def reader(file_path):
    # A function for returning a list of ASE a
    
    properties_order = [
        'tag', 'index', 'A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv'
    ]
        
#     images = []
    with open(file_path, 'r') as f:
        lines = [_.strip() for _ in f.readlines()]
        
        na = int(lines[0])
        properties = lines[1].split()
        
        symbols = []
        positions = []
        partial_charges = []
        
        for line in lines[2:2+na]:
            split = line.split()
            split = [_.replace('*^', 'e') for _ in split]  # Python-readable scientific notation
            
            # Line order: symbol, x, y, z, charge
            symbols.append(split[0])
            positions.append(split[1:4])
            partial_charges.append(split[-1])
            
        positions = np.array(positions)
        partial_charges = np.array(partial_charges, dtype=float)
                
        atoms = Atoms(symbols=symbols, positions=positions)
        
        atoms.info['mulliken_partial_charges'] = partial_charges
        
        name = os.path.splitext(os.path.split(file_path)[-1])[0]

        atoms.info['name'] = name
        
        for pname, val in zip(properties_order[2:], properties[2:]):
            atoms.info[pname] = float(val)
            
        frequencies = np.array(lines[-3].split(), dtype=float)
        atoms.info['frequencies'] = frequencies
                
        smiles = lines[-2].split()
        inchi  = lines[-1].split()
        
        atoms.info['SMILES']    = smiles[0]
        atoms.info['SMILES_relaxed'] = smiles[1]
        atoms.info['InChI']     = inchi[0]
        atoms.info['InChI_relaxed']  = inchi[1]
        
        yield atoms
#         images.append(atoms)
    
#     return images


# ## Load and insert the data

# In[8]:


from colabfit.tools.client import load_data

images = list(load_data(
    file_path='/home/jvita/scripts/colabfit/data/quantum-machine/qm9',
    file_format='folder',
    name_field='name',  # key in Configuration.info to use as the Configuration name
    elements=['H', 'C', 'N', 'O', 'F'],    # order matters for CFG files, but not others
    default_name='qm9',  # default name with `name_field` not found
    reader=reader,
    glob_string='*.xyz',
    verbose=True
))


# In[ ]:


ids = list(client.insert_data(
    images,
    property_map=property_map,
    property_settings={'energy-forces': pso_id},
    generator=True,
    verbose=True
))
