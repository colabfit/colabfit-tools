import tempfile
import numpy as np
np.random.seed(42)
import random
random.seed(42)
from ase import Atoms

from colabfit import ATOMS_NAME_FIELD, ATOMS_LABELS_FIELD
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
    def test_insert_pso_definition_data(self):

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
                labels=['pso_label1', 'pso_label2']
            )


            pso_id = client.insert_property_settings(pso)

            for i, img in enumerate(images):
                img.info[ATOMS_NAME_FIELD].add(f'config_{i}')
                img.info[ATOMS_LABELS_FIELD].add('a_label')

            ids = client.insert_data(
                images,
                property_map=property_map,
                property_settings={'default': pso_id}
            )

            pso_doc = next(client.property_settings.find({'_id': pso_id}))

            for i, ((cid, pid), config) in enumerate(zip(ids, images)):
                config_doc = next(client.configurations.find({'_id': cid}))
                prop_doc   = next(client.properties.find({'_id': pid}))

                pn = client.database[f'properties/default/forces/data/{pid}'].shape[0]

                na = len(config)
                assert config_doc['nsites'] == na
                assert pn == na

                assert config_doc['chemical_formula_anonymous'] == 'A'
                assert config_doc['chemical_formula_hill'] == config.get_chemical_formula()
                assert config_doc['chemical_formula_reduced'] == 'H'
                assert config_doc['dimension_types'] == [0, 0, 0]
                assert config_doc['elements'] == ['H']
                assert config_doc['elements_ratios'] == [1.0]
                assert {'a_label'}.issubset(config_doc['labels'])
                np.testing.assert_allclose(
                    config_doc['lattice_vectors'],
                    np.array(config.get_cell())
                )
                assert config_doc['names'] == [f'config_{i}']
                assert config_doc['nsites'] == len(config)
                assert config_doc['nelements'] == 1
                assert config_doc['nperiodic_dimensions'] == 0
                assert {pid}.issubset(config_doc['relationships']['properties'])

                assert {'pso_label2', 'pso_label1'}.issubset(set(prop_doc['labels']))
                assert prop_doc['type'] == 'default'
                assert {cid}.issubset(prop_doc['relationships']['configurations'])
                assert {pso_id}.issubset(prop_doc['relationships']['property_settings'])
                
                assert {pid}.issubset(pso_doc['relationships']['properties'])


    def test_insert_cs(self):

        with tempfile.TemporaryFile() as tmpfile:
            client = HDF5Client(tmpfile, mode='w')

            images = build_n(10)[0]

            for i, img in enumerate(images):
                img.info[ATOMS_NAME_FIELD].add(f'config_{i}')
                img.info[ATOMS_LABELS_FIELD].add('a_label')

            ids = client.insert_data(images)

            co_ids = list(zip(*ids))[0]

            cs_id = client.insert_configuration_set(co_ids, 'a description')

            cs_doc = next(client.configuration_sets.find({'_id': cs_id}))

            agg_info = cs_doc['aggregated_info']

            assert cs_doc['description'] == 'a description'

            assert agg_info['nconfigurations'] == len(ids)
            assert agg_info['nsites'] == sum(len(c) for c in images)
            assert agg_info['nelements'] == 1
            assert agg_info['elements'] == ['H']
            assert agg_info['individual_elements_ratios'] == [[1.0]]
            assert agg_info['total_elements_ratios'] == [1.0]
            assert agg_info['labels'] == ['a_label']
            assert agg_info['labels_counts'] == [len(ids)]
            assert agg_info['chemical_formula_reduced'] == ['H']
            assert agg_info['chemical_formula_anonymous'] == ['A']
            assert set(agg_info['chemical_formula_hill']) == {
                f'H{i+1}' if i > 0 else 'H' for i in range(len(ids))
            }
            assert agg_info['nperiodic_dimensions'] == [0]
            assert agg_info['dimension_types'] == [[0,0,0]]