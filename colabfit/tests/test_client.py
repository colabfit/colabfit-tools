import tempfile
import numpy as np
np.random.seed(42)
import random
random.seed(42)
from ase import Atoms

from colabfit.tools.configuration import Configuration
from colabfit.tools.client import HDF5Client
from colabfit.tools.property_settings import PropertySettings


def build_n(n):
    images              = []
    energies            = []
    stress              = []
    names               = []
    nd_same_shape       = []
    nd_diff_shape       = []
    forces              = []
    nd_same_shape_arr   = []
    nd_diff_shape_arr   = []

    for i in range(1, n+1):
        atoms = Atoms(f'H{i}', positions=np.random.random((i, 3)))

        atoms.info['energy'] = np.random.random()
        atoms.info['stress'] = np.random.random(6)
        atoms.info['name'] = f'configuration_{i}'
        atoms.info['nd-same-shape'] = np.random.random((2, 3, 5))
        atoms.info['nd-diff-shapes'] = np.random.random((
            i+np.random.randint(1, 4),
            i+1+np.random.randint(1, 4),
            i+2+np.random.randint(1, 4),
        ))

        energies.append(atoms.info['energy'])
        stress.append(atoms.info['stress'])
        names.append(atoms.info['name'])
        nd_same_shape.append(atoms.info['nd-same-shape'])
        nd_diff_shape.append(atoms.info['nd-diff-shapes'])

        atoms.arrays['forces'] = np.random.random((i, 3))
        atoms.arrays['nd-same-shape-arr'] = np.random.random((i, 2, 3))
        atoms.arrays['nd-diff-shapes-arr'] = np.random.random((
            i,
            i+np.random.randint(1, 4),
            i+1+np.random.randint(1, 4),
        ))

        forces.append(atoms.arrays['forces'])
        nd_same_shape_arr.append(atoms.arrays['nd-same-shape-arr'])
        nd_diff_shape_arr.append(atoms.arrays['nd-diff-shapes-arr'])

        images.append(Configuration.from_ase(atoms))

    return (
        images,
        energies, stress, names, nd_same_shape, nd_diff_shape,
        forces, nd_same_shape_arr, nd_diff_shape_arr,
    )


class TestClient:
    def test_insert_data(self):

        with tempfile.TemporaryFile() as tmpfile:
            client = HDF5Client(tmpfile, mode='w')

            images = build_n(10)[0]

            client.insert_property_definition(
                {
                    'property-id': 'default',
                    'property-title': 'A default property used for testing',
                    'property-description': 'A description of the property',
                    'energy': {'type': 'float', 'has-unit': True, 'extent': [], 'required': True, 'description': 'empty'},
                    'stress': {'type': 'float', 'has-unit': True, 'extent': [6], 'required': True, 'description': 'empty'},
                    'name': {'type': 'string', 'has-unit': False, 'extent': [], 'required': True, 'description': 'empty'},
                    'nd-same-shape': {'type': 'float', 'has-unit': True, 'extent': [2,3,5], 'required': True, 'description': 'empty'},
                    'nd-diff-shapes': {'type': 'float', 'has-unit': True, 'extent': [":", ":", ":"], 'required': True, 'description': 'empty'},
                    'forces': {'type': 'float', 'has-unit': True, 'extent': [":", 3], 'required': True, 'description': 'empty'},
                    'nd-same-shape-arr': {'type': 'float', 'has-unit': True, 'extent': [':', 2, 3], 'required': True, 'description': 'empty'},
                    'nd-diff-shapes-arr': {'type': 'float', 'has-unit': True, 'extent': [':', ':', ':'], 'required': True, 'description': 'empty'},
                }
            )

            property_map = {
                'default': {
                    'energy': {'field': 'energy', 'units': 'eV'},
                    'stress': {'field': 'stress', 'units': 'GPa'},
                    'name': {'field': 'name', 'units': None},
                    'nd-same-shape': {'field': 'nd-same-shape', 'units': 'eV'},
                    'nd-diff-shapes': {'field': 'nd-diff-shapes', 'units': 'eV'},
                    'forces': {'field': 'forces', 'units': 'eV/Ang'},
                    'nd-same-shape-arr': {'field': 'nd-same-shape-arr', 'units': 'eV/Ang'},
                    'nd-diff-shapes-arr': {'field': 'nd-diff-shapes-arr', 'units': 'eV/Ang'},
                }
            }

            pso = PropertySettings(
                method='VASP',
                description='A basic test calculation',
                files=[('dummy_name', 'dummy file contents')],
            )


            pso_id = client.insert_property_settings(pso)

            ids = client.insert_data(
                images,
                property_map=property_map,
                property_settings={'default': pso_id}
            )

            for cid, pid in ids:
                cn = next(client.configurations.find({'_id': cid}))['natoms']
                pn = client.database[f'properties/default/forces/data/{pid}'].shape[0]

                assert cn == pn