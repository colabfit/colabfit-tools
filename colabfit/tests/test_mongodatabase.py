import pytest
import tempfile
import numpy as np
np.random.seed(42)
import random
random.seed(42)
from ase import Atoms

from colabfit import ATOMS_NAME_FIELD, ATOMS_LABELS_FIELD, ID_FORMAT_STRING
from colabfit.tools.configuration import AtomicConfiguration
from colabfit.tools.database import MongoDatabase
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

        images.append(AtomicConfiguration.from_ase(atoms))

    return (
        images,
        energies, stress, names, nd_same_shape, nd_diff_shape,
        forces, nd_same_shape_arr, nd_diff_shape_arr,
    )


class TestMongoDatabase:
    database_name = 'colabfit_test'

    # def test_from_html(self):
    #    with tempfile.TemporaryFile() as tmpfile:

    #         database = MongoDatabase(self.database_name, drop_database=True)

    #         # Just makes sure from_markdown doesn't throw an error
    #         dataset = database.dataset_from_markdown(
    #             'colabfit/tests/files/test.md',
    #             generator=True
    #         )

    #         assert 3 == dataset.aggregated_info['nconfigurations']
    #         assert 3 == sum(dataset.aggregated_info['property_types_counts'])


    def test_add_then_update_nochange_config(self):
        with tempfile.TemporaryFile() as tmpfile:

            database = MongoDatabase(self.database_name, drop_database=True,configuration_type=AtomicConfiguration)

            returns = build_n(10)

            images              = returns[0]
            energies            = returns[1]
            stress              = returns[2]
            names               = returns[3]
            nd_same_shape       = returns[4]
            nd_diff_shape       = returns[5]
            forces              = returns[6]
            nd_same_shape_arr   = returns[7]
            nd_diff_shape_arr   = returns[8]

            ids = list(database.insert_data(images))

            for co_data in database.get_data('configurations', 'names'):
                assert len(co_data) == 0

            for co_data in database.get_data('configurations', 'labels'):
                assert len(co_data) == 0

            for co_data in database.get_data('configurations', 'last_modified'):
                assert len(co_data)

            np.testing.assert_allclose(
                database.get_data('configurations', 'atomic_numbers', concatenate=True),
                np.concatenate([_.get_atomic_numbers() for _ in images])
            )
            np.testing.assert_allclose(
                database.get_data('configurations', 'positions', concatenate=True),
                np.concatenate([_.get_positions() for _ in images])
            )
            np.testing.assert_allclose(
                database.get_data('configurations', 'cell', concatenate=True),
                np.concatenate([_.get_cell() for _ in images])
            )
            np.testing.assert_allclose(
                database.get_data('configurations', 'pbc', vstack=True),
                np.vstack([_.get_pbc() for _ in images]).astype(int)
            )


    def test_add_then_update_with_changes_config(self):
        with tempfile.TemporaryFile() as tmpfile:

            database = MongoDatabase(self.database_name, drop_database=True,configuration_type=AtomicConfiguration)

            returns = build_n(10)

            images              = returns[0]
            energies            = returns[1]
            stress              = returns[2]
            names               = returns[3]
            nd_same_shape       = returns[4]
            nd_diff_shape       = returns[5]
            forces              = returns[6]
            nd_same_shape_arr   = returns[7]
            nd_diff_shape_arr   = returns[8]

            list(database.insert_data(images))

            for img in images:
                img.info[ATOMS_NAME_FIELD].add('change')
                img.info[ATOMS_LABELS_FIELD].add('another_label')

            list(database.insert_data(images))

            for n in database.get_data('configurations', 'names', concatenate=True):
                assert n == 'change'

            for n in database.get_data('configurations', 'labels', concatenate=True):
                assert n == 'another_label'

            for img in images:
                img.info[ATOMS_NAME_FIELD].add('change2')
                img.info[ATOMS_LABELS_FIELD] = {'another_label2'}

            ids = list(database.insert_data(images))

            for n in database.get_data('configurations', 'names'):
                assert n[0] == 'change'
                assert n[1] == 'change2'

            for n in database.get_data('configurations', 'labels'):
                assert n[0] == 'another_label'
                assert n[1] == 'another_label2'

            database.get_configuration(ids[0][0])

            np.testing.assert_allclose(
                database.get_data('configurations', 'atomic_numbers', concatenate=True),
                np.concatenate([_.get_atomic_numbers() for _ in images])
            )
            np.testing.assert_allclose(
                database.get_data('configurations', 'positions', concatenate=True),
                np.concatenate([_.get_positions() for _ in images])
            )
            np.testing.assert_allclose(
                database.get_data('configurations', 'cell', concatenate=True),
                np.concatenate([_.get_cell() for _ in images])
            )
            np.testing.assert_allclose(
                database.get_data('configurations', 'pbc', vstack=True),
                np.vstack([_.get_pbc() for _ in images]).astype(int)
            )



    def test_add_configs_props_diff_def(self):
        with tempfile.TemporaryFile() as tmpfile:

            database = MongoDatabase(self.database_name, drop_database=True,configuration_type=AtomicConfiguration)

            returns = build_n(10)

            images              = returns[0]
            energies            = returns[1]
            stress              = returns[2]
            names               = returns[3]
            nd_same_shape       = returns[4]
            nd_diff_shape       = returns[5]
            forces              = returns[6]
            nd_same_shape_arr   = returns[7]
            nd_diff_shape_arr   = returns[8]

            database.insert_property_definition(
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
                'default': [{
                    'energy': {'field': 'dft-energy', 'units': 'eV'},
                    'stress': {'field': 'dft-stress', 'units': 'GPa'},
                    'name': {'field': 'name', 'units': None},
                    'nd-same-shape': {'field': 'nd-same-shape', 'units': 'eV'},
                    'nd-diff-shapes': {'field': 'nd-diff-shapes', 'units': 'eV'},
                    'forces': {'field': 'dft-forces', 'units': 'eV/Ang'},
                    'nd-same-shape-arr': {'field': 'nd-same-shape-arr', 'units': 'eV/Ang'},
                    'nd-diff-shapes-arr': {'field': 'nd-diff-shapes-arr', 'units': 'eV/Ang'},
                }]
            }

            for img in images:
                img.info['dft-energy'] = img.info['energy']
                img.info['dft-stress'] = img.info['stress']
                img.arrays['dft-forces'] = img.arrays['forces']

                del img.info['energy']
                del img.info['stress']
                del img.arrays['forces']

            list(zip(*database.insert_data(
                images, property_map=property_map
            )))

            np.testing.assert_allclose(
                database.get_data('properties', 'default.energy', ravel=True),
                np.hstack(energies)
            )
            np.testing.assert_allclose(
                database.get_data('properties', 'default.stress', ravel=True),
                np.hstack(stress)
            )
            decoded_names = database.get_data('properties', 'default.name', ravel=True)
            decoded_names = decoded_names.tolist()
            assert decoded_names == names
            np.testing.assert_allclose(
                database.get_data('properties', 'default.nd-same-shape', ravel=True),
                np.concatenate(nd_same_shape).ravel()
            )
            data = database.get_data('properties', 'default.nd-diff-shapes')
            for a1, a2 in zip(data, nd_diff_shape):
                np.testing.assert_allclose(a1, a2)

            np.testing.assert_allclose(
                database.get_data('properties', 'default.forces', concatenate=True),
                np.concatenate(forces)
            )
            np.testing.assert_allclose(
                database.get_data('properties', 'default.nd-same-shape-arr', concatenate=True),
                np.concatenate(nd_same_shape_arr)
            )
            data = database.get_data('properties', 'default.nd-diff-shapes-arr')
            for a1, a2 in zip(data, nd_diff_shape_arr):
                np.testing.assert_allclose(a1, a2)


    def test_add_then_update_with_properties_with_change(self):
        with tempfile.TemporaryFile() as tmpfile:

            database = MongoDatabase(self.database_name, drop_database=True,configuration_type=AtomicConfiguration)

            returns = build_n(10)

            images              = returns[0]
            energies            = returns[1]
            stress              = returns[2]
            names               = returns[3]
            nd_same_shape       = returns[4]
            nd_diff_shape       = returns[5]
            forces              = returns[6]
            nd_same_shape_arr   = returns[7]
            nd_diff_shape_arr   = returns[8]

            database.insert_property_definition(
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
                'default': [{
                    'energy': {'field': 'energy', 'units': 'eV'},
                    'stress': {'field': 'stress', 'units': 'GPa'},
                    'name': {'field': 'name', 'units': None},
                    'nd-same-shape': {'field': 'nd-same-shape', 'units': 'eV'},
                    'nd-diff-shapes': {'field': 'nd-diff-shapes', 'units': 'eV'},
                    'forces': {'field': 'forces', 'units': 'eV/Ang'},
                    'nd-same-shape-arr': {'field': 'nd-same-shape-arr', 'units': 'eV/Ang'},
                    'nd-diff-shapes-arr': {'field': 'nd-diff-shapes-arr', 'units': 'eV/Ang'},

                    '_settings': {
                        '_method': 'VASP',
                        '_description': 'A basic test calculation',
                        '_files': [('dummy_name', 'dummy file contents')],
                    }
                }]
            }

            database.insert_data(
                images,
                property_map=property_map,
            )

            for img in images:
                img.info['energy'] += 100000

            database.insert_data(
                images,
                property_map=property_map,
            )

            for img in images:
                img.info['energy'] += 100000

            database.insert_data(
                images,
                property_map=property_map,
            )

            np.testing.assert_allclose(
                database.get_data('configurations', 'atomic_numbers', concatenate=True),
                np.concatenate([_.get_atomic_numbers() for _ in images])
            )
            np.testing.assert_allclose(
                database.get_data('configurations', 'positions', concatenate=True),
                np.concatenate([_.get_positions() for _ in images])
            )
            np.testing.assert_allclose(
                database.get_data('configurations', 'cell', concatenate=True),
                np.concatenate([_.get_cell() for _ in images])
            )
            np.testing.assert_allclose(
                database.get_data('configurations', 'pbc', concatenate=True),
                np.concatenate([_.get_pbc() for _ in images]).astype(int)
            )

            np.testing.assert_allclose(
                database.get_data('properties', 'default.energy', ravel=True),
                np.concatenate([
                    np.hstack(energies),
                    np.hstack(energies)+100000,
                    np.hstack(energies)+200000
                ]).ravel()
            )
            np.testing.assert_allclose(
                database.get_data('properties', 'default.stress', concatenate=True),
                np.concatenate([
                    np.hstack(stress),
                    np.hstack(stress),
                    np.hstack(stress),
                ])
            )
            decoded_names = database.get_data('properties', 'default.name')
            #decoded_names = decoded_names.tolist()
            assert decoded_names == names*3
            np.testing.assert_allclose(
                database.get_data('properties', 'default.nd-same-shape', concatenate=True),
                np.concatenate([
                    np.concatenate(nd_same_shape),
                    np.concatenate(nd_same_shape),
                    np.concatenate(nd_same_shape),
                ])
            )
            data = database.get_data('properties', 'default.nd-diff-shapes')
            for a1, a2 in zip(data, nd_diff_shape*3):
                np.testing.assert_allclose(a1, a2)

            np.testing.assert_allclose(
                database.get_data('properties', 'default.forces', concatenate=True),
                np.concatenate([
                    np.concatenate(forces),
                    np.concatenate(forces),
                    np.concatenate(forces),
                ])
            )
            np.testing.assert_allclose(
                database.get_data('properties', 'default.nd-same-shape-arr', concatenate=True),
                np.concatenate([
                    np.concatenate(nd_same_shape_arr),
                    np.concatenate(nd_same_shape_arr),
                    np.concatenate(nd_same_shape_arr),
                ])
            )
            data = database.get_data('properties', 'default.nd-diff-shapes-arr')
            for a1, a2 in zip(data, nd_diff_shape_arr*3):
                np.testing.assert_allclose(a1, a2)


    def test_add_then_update_with_properties_nochange(self):
        with tempfile.TemporaryFile() as tmpfile:

            database = MongoDatabase(self.database_name, drop_database=True,configuration_type=AtomicConfiguration)

            returns = build_n(10)

            images              = returns[0]
            energies            = returns[1]
            stress              = returns[2]
            names               = returns[3]
            nd_same_shape       = returns[4]
            nd_diff_shape       = returns[5]
            forces              = returns[6]
            nd_same_shape_arr   = returns[7]
            nd_diff_shape_arr   = returns[8]

            database.insert_property_definition(
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
                'default': [{
                    'energy': {'field': 'energy', 'units': 'eV'},
                    'stress': {'field': 'stress', 'units': 'GPa'},
                    'name': {'field': 'name', 'units': None},
                    'nd-same-shape': {'field': 'nd-same-shape', 'units': 'eV'},
                    'nd-diff-shapes': {'field': 'nd-diff-shapes', 'units': 'eV'},
                    'forces': {'field': 'forces', 'units': 'eV/Ang'},
                    'nd-same-shape-arr': {'field': 'nd-same-shape-arr', 'units': 'eV/Ang'},
                    'nd-diff-shapes-arr': {'field': 'nd-diff-shapes-arr', 'units': 'eV/Ang'},

                    '_settings': {
                        '_method': 'VASP',
                        '_description': 'A basic test calculation',
                        '_files': [('dummy_name', 'dummy file contents')],
                    }
                }]
            }

            database.insert_data(
                images,
                property_map=property_map,
            )

            database.insert_data(
                images,
                property_map=property_map,
            )

            np.testing.assert_allclose(
                database.get_data('properties', 'default.energy'),
                np.hstack(energies)
            )
            np.testing.assert_allclose(
                database.get_data('properties', 'default.stress', concatenate=True),
                np.hstack(stress)
            )
            decoded_names =  database.get_data('properties', 'default.name')
            #decoded_names = decoded_names.tolist()
            assert decoded_names == names
            np.testing.assert_allclose(
                database.get_data('properties', 'default.nd-same-shape', concatenate=True),
                np.concatenate(nd_same_shape)
            )
            data = database.get_data('properties', 'default.nd-diff-shapes')
            for a1, a2 in zip(data, nd_diff_shape):
                np.testing.assert_allclose(a1, a2)

            np.testing.assert_allclose(
                database.get_data('properties', 'default.forces', concatenate=True),
                np.concatenate(forces)
            )
            np.testing.assert_allclose(
                database.get_data('properties', 'default.nd-same-shape-arr', concatenate=True),
                np.concatenate(nd_same_shape_arr)
            )
            data = database.get_data('properties', 'default.nd-diff-shapes-arr')
            for a1, a2 in zip(data, nd_diff_shape_arr):
                np.testing.assert_allclose(a1, a2)


    def test_get_configurations(self):

        with tempfile.TemporaryFile() as tmpfile:
            database = MongoDatabase(self.database_name, drop_database=True,configuration_type=AtomicConfiguration)

            images = build_n(10)[0]

            database.insert_data(images)

            count = 0
            for atoms, img in zip(database.get_configurations('all'), images):
                assert atoms == img
                count += 1
            assert count == 10

    # Not implemented yet
    # def test_get_configurations_attach_settings(self):
    #
    #     database = MongoDatabase(self.database_name, drop_database=True,configuration_type=AtomicConfiguration)
    #
    #     images = build_n(10)[0]
    #
    #     database.insert_property_definition(
    #         {
    #             'property-id': 'default',
    #             'property-title': 'A default property used for testing',
    #             'property-description': 'A description of the property',
    #
    #             'energy': {'type': 'float', 'has-unit': True, 'extent': [], 'required': True, 'description': 'empty'},
    #             'forces': {'type': 'float', 'has-unit': True, 'extent': [":", 3], 'required': True, 'description': 'empty'},
    #             'stress': {'type': 'float', 'has-unit': True, 'extent': [6], 'required': True, 'description': 'empty'},
    #
    #             # 'name': {'type': 'string', 'has-unit': False, 'extent': [], 'required': True, 'description': 'empty'},
    #             # 'nd-same-shape': {'type': 'float', 'has-unit': True, 'extent': [2,3,5], 'required': True, 'description': 'empty'},
    #             # 'nd-diff-shapes': {'type': 'float', 'has-unit': True, 'extent': [":", ":", ":"], 'required': True, 'description': 'empty'},
    #             # 'forces': {'type': 'float', 'has-unit': True, 'extent': [":", 3], 'required': True, 'description': 'empty'},
    #             # 'nd-same-shape-arr': {'type': 'float', 'has-unit': True, 'extent': [':', 2, 3], 'required': True, 'description': 'empty'},
    #             # 'nd-diff-shapes-arr': {'type': 'float', 'has-unit': True, 'extent': [':', ':', ':'], 'required': True, 'description': 'empty'},
    #         }
    #     )
    #
    #     property_map = {
    #         'default': [{
    #             'energy': {'field': 'energy', 'units': 'eV'},
    #             'forces': {'field': 'forces', 'units': 'eV/Ang'},
    #             'stress': {'field': 'stress', 'units': 'GPa'},
    #
    #             '_settings': {
    #
    #                 '_method': 'VASP',
    #                 '_description': 'A basic test calculation',
    #                 '_files': [('dummy_name', 'dummy file contents')],
    #                 '_labels': ['ps_label1', 'ps_label2'],
    #
    #                 'name':             {'required': False, 'field': 'name', 'units': None},
    #                 'nd-same-shape':    {'required': False, 'field': 'nd-same-shape', 'units': 'eV'},
    #                 'nd-diff-shapes':   {'required': False, 'field': 'nd-diff-shapes', 'units': 'eV'},
    #                 'nd-same-shape-arr':    {'required': False, 'field': 'nd-same-shape-arr', 'units': 'eV/Ang'},
    #                 'nd-diff-shapes-arr':   {'required': False, 'field': 'nd-diff-shapes-arr', 'units': 'eV/Ang'},
    #             }
    #         }]
    #     }
    #
    #     for i, img in enumerate(images):
    #         img.info[ATOMS_NAME_FIELD].add(f'config_{i}')
    #         img.info[ATOMS_LABELS_FIELD].add('a_label')
    #
    #     ids = database.insert_data(
    #         images,
    #         property_map=property_map,
    #     )
    #
    #     rebuilt_configs = database.get_configurations(
    #         [_[0] for _ in ids],
    #         attach_properties=True,
    #         attach_settings=True,
    #     )
    #
    #     for i, ((cid, pid), config) in enumerate(zip(ids, images)):
    #         config_doc = next(database.configurations.find({'_id': cid}))
    #         prop_doc   = next(database.properties.find({'_id': pid}))
    #
    #         pn = database.get_data(
    #             'properties', 'default.forces', ids=[pid], concatenate=True
    #         ).shape[0]
    #
    #         na = len(config)
    #         assert config_doc['nsites'] == na
    #         assert pn == na
    #
    #         assert config_doc['chemical_formula_anonymous'] == 'A'
    #         assert config_doc['chemical_formula_hill'] == config.get_chemical_formula()
    #         assert config_doc['chemical_formula_reduced'] == 'H'
    #         assert config_doc['dimension_types'] == [0, 0, 0]
    #         assert config_doc['elements'] == ['H']
    #         assert config_doc['elements_ratios'] == [1.0]
    #         assert {'a_label'}.issubset(config_doc['labels'])
    #         np.testing.assert_allclose(
    #             config_doc['cell],
    #             np.array(config.get_cell())
    #         )
    #         assert config_doc['names'] == [f'config_{i}']
    #         assert config_doc['nsites'] == len(config)
    #         assert config_doc['nelements'] == 1
    #         assert config_doc['nperiodic_dimensions'] == 0
    #         assert {pid}.issubset(config_doc['relationships']['properties'])
    #
    #         assert {cid}.issubset(prop_doc['relationships']['configurations'])
    #
    #         assert database.property_settings.count_documents({
    #             'relationships.properties': pid
    #         })
    #
    #
    #         assert rebuilt_configs[i].info['_settings._method'] == 'VASP'
    #         assert rebuilt_configs[i].info['_settings._description'] == 'A basic test calculation'
    #         assert set(rebuilt_configs[i].info['_settings._labels']) == {'ps_label1', 'ps_label2'}
    #
    #     database.drop_database(database.database_name)


    def test_insert_pso_definition_data(self):

        with tempfile.NamedTemporaryFile() as tmpfile:
            database = MongoDatabase(self.database_name, drop_database=True,configuration_type=AtomicConfiguration)

            images = build_n(10)[0]

            database.insert_property_definition(
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
                'default': [{
                    'energy': {'field': 'energy', 'units': 'eV'},
                    'stress': {'field': 'stress', 'units': 'GPa'},
                    'name': {'field': 'name', 'units': None},
                    'nd-same-shape': {'field': 'nd-same-shape', 'units': 'eV'},
                    'nd-diff-shapes': {'field': 'nd-diff-shapes', 'units': 'eV'},
                    'forces': {'field': 'forces', 'units': 'eV/Ang'},
                    'nd-same-shape-arr': {'field': 'nd-same-shape-arr', 'units': 'eV/Ang'},
                    'nd-diff-shapes-arr': {'field': 'nd-diff-shapes-arr', 'units': 'eV/Ang'},

                    '_settings': {
                        '_method': 'VASP',
                        '_description': 'A basic test calculation',
                        '_files': [('dummy_name', 'dummy file contents')],
                    }
                }]
            }

            for i, img in enumerate(images):
                img.info[ATOMS_NAME_FIELD].add(f'config_{i}')
                img.info[ATOMS_LABELS_FIELD].add('a_label')

            ids = database.insert_data(
                images,
                property_map=property_map,
            )

            for i, ((cid, pid), config) in enumerate(zip(ids, images)):
                config_doc = next(database.configurations.find({'_id': cid}))
                prop_doc   = next(database.properties.find({'_id': pid}))

                pn = database.get_data(
                    'properties', 'default.forces', ids=[pid], concatenate=True
                ).shape[0]

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
                    config_doc['cell'],
                    np.array(config.get_cell())
                )
                assert config_doc['names'] == [f'config_{i}']
                assert config_doc['nsites'] == len(config)
                assert config_doc['nelements'] == 1
                assert config_doc['nperiodic_dimensions'] == 0
                assert {pid}.issubset(config_doc['relationships']['properties'])

                assert {cid}.issubset(prop_doc['relationships']['configurations'])

                assert database.property_settings.count_documents({
                    'relationships.properties': pid
                })

            database.drop_database(database.database_name)


    def test_insert_cs(self):

        with tempfile.NamedTemporaryFile() as tmpfile:
            database = MongoDatabase(self.database_name, drop_database=True,configuration_type=AtomicConfiguration)

            images = build_n(10)[0]

            for i, img in enumerate(images):
                img.info[ATOMS_NAME_FIELD].add(f'config_{i}')
                img.info[ATOMS_LABELS_FIELD].add('a_label')

            ids = database.insert_data(images)

            co_ids = list(zip(*ids))[0]

            cs_id = database.insert_configuration_set(co_ids, 'a description')

            cs_doc = next(database.configuration_sets.find({'_id': cs_id}))

            agg_info = cs_doc['aggregated_info']

            assert cs_doc['description'] == 'a description'

            assert agg_info['nconfigurations'] == len(ids)
            assert agg_info['nsites'] == sum(len(c) for c in images)
            assert agg_info['nelements'] == 1
            assert agg_info['elements'] == ['H']
            assert agg_info['individual_elements_ratios'] == {'H': [1.0]}
            assert agg_info['total_elements_ratios'] == {'H': 1.0}
            assert agg_info['labels'] == ['a_label']
            assert agg_info['labels_counts'] == [len(ids)]
            assert agg_info['chemical_formula_reduced'] == ['H']
            assert agg_info['chemical_formula_anonymous'] == ['A']
            assert set(agg_info['chemical_formula_hill']) == {
                f'H{i+1}' if i > 0 else 'H' for i in range(len(ids))
            }
            assert agg_info['nperiodic_dimensions'] == [0]
            assert agg_info['dimension_types'] == [[0,0,0]]

            database.drop_database(database.database_name)

    def test_insert_ds_diff_cs(self):

        with tempfile.NamedTemporaryFile() as tmpfile:
            database = MongoDatabase(self.database_name, drop_database=True,configuration_type=AtomicConfiguration)

            database.insert_property_definition(
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
                'default': [{
                    'energy': {'field': 'energy', 'units': 'eV'},
                    'stress': {'field': 'stress', 'units': 'GPa'},
                    'name': {'field': 'name', 'units': None},
                    'nd-same-shape': {'field': 'nd-same-shape', 'units': 'eV'},
                    'nd-diff-shapes': {'field': 'nd-diff-shapes', 'units': 'eV'},
                    'forces': {'field': 'forces', 'units': 'eV/Ang'},
                    'nd-same-shape-arr': {'field': 'nd-same-shape-arr', 'units': 'eV/Ang'},
                    'nd-diff-shapes-arr': {'field': 'nd-diff-shapes-arr', 'units': 'eV/Ang'},

                    '_settings': {
                        '_method': 'VASP',
                        '_description': 'A basic test calculation',
                        '_files': [('dummy_name', 'dummy file contents')],
                        '_labels': ['pso_label1', 'pso_label2']
                    }
                }]
            }

            # pso = PropertySettings(
            #     method='VASP',
            #     description='A basic test calculation',
            #     files=[('dummy_name', 'dummy file contents')],
            #     labels=['pso_label1', 'pso_label2']
            # )

            # pso_id = database.insert_property_settings(pso)

            images = build_n(10)[0]

            for i, img in enumerate(images):
                img.info[ATOMS_NAME_FIELD].add(f'config_{i}')
                img.info[ATOMS_LABELS_FIELD].add('a_label')

            ids = database.insert_data(
                images,
                property_map=property_map,
            )

            co_ids1, pr_ids1 = list(zip(*ids))

            cs_id1 = database.insert_configuration_set(co_ids1, 'a description1')

            images = build_n(10)[0]

            for i, img in enumerate(images):
                img.info[ATOMS_NAME_FIELD].add(f'second_config_{i}')
                img.info[ATOMS_LABELS_FIELD].add('a_second_label')

                img.info['energy'] += 100000

            ids = database.insert_data(
                images,
                property_map=property_map,
            )

            co_ids2, pr_ids2 = list(zip(*ids))

            cs_id2 = database.insert_configuration_set(co_ids2, 'a description2')

            ds_id = database.insert_dataset(
                cs_ids=[cs_id1, cs_id2],
                pr_ids=pr_ids1+pr_ids2,
                name='example_dataset',
                authors=['colabfit'],
                links=['https://colabfit.org'],
                description='an example dataset',
                resync=True
            )

            ds_doc = next(database.datasets.find({'_id': ds_id}))

            assert ds_doc['authors'] == ['colabfit']
            assert ds_doc['links'] == ['https://colabfit.org']
            assert ds_doc['description'] == 'an example dataset'
            assert len(ds_doc['relationships']['configuration_sets']) == 2
            assert len(ds_doc['relationships']['properties']) == 20

            agg = ds_doc['aggregated_info']

            assert agg['nconfigurations'] == 20
            assert agg['nsites'] == 110
            assert agg['nelements'] == 1
            assert agg['elements'] == ['H']
            assert agg['individual_elements_ratios'] == {'H': [1.0]}
            assert agg['total_elements_ratios'] == {'H': 1.0}
            assert {'a_label', 'a_second_label'}.issubset(agg['configuration_labels'])
            assert agg['chemical_formula_reduced'] == ['H']
            assert agg['chemical_formula_anonymous'] == ['A']
            assert set(agg['chemical_formula_hill']) == {
                f'H{i+1}' if i > 0 else 'H' for i in range(len(ids))
            }
            assert agg['nperiodic_dimensions'] == [0]
            assert agg['dimension_types'] == [[0,0,0]]

            assert agg['property_types'] == ['default']
            assert set(agg['property_labels']) == {'pso_label1', 'pso_label2'}

            database.drop_database(database.database_name)


class TestPropertyDefinitionsAndSettings:
    database_name = 'colabfit_test'

    def test_invalid_definition(self):
        with tempfile.TemporaryFile() as tmpfile:
            database = MongoDatabase(self.database_name, drop_database=True,configuration_type=AtomicConfiguration)

            property_definition = {
                'property-id': 'this should throw an error',
            }

            with pytest.raises(Exception):
                database.insert_property_definition(property_definition)


    def test_definition_setter_getter(self):
        with tempfile.TemporaryFile() as tmpfile:
            database = MongoDatabase(self.database_name, drop_database=True,configuration_type=AtomicConfiguration)

            property_definition = {
                    'property-id': 'default',
                    'property-title': 'A default property used for testing',
                    'property-description': 'A description of the property',
                    'energy': {'type': 'float', 'has-unit': True, 'extent': [], 'required': True, 'description': 'empty'},
                    'stress': {'type': 'float', 'has-unit': True, 'extent': [6], 'required': True, 'description': 'empty'},
                    'name': {'type': 'string', 'has-unit': False, 'extent': [], 'required': True, 'description': 'empty'},
                    'nd-same-shape': {'type': 'float', 'has-unit': True, 'extent': [2,3,5], 'required': True, 'description': 'empty'},
                    'nd-diff-shape': {'type': 'float', 'has-unit': True, 'extent': [":", ":", ":"], 'required': True, 'description': 'empty'},
                    'forces': {'type': 'float', 'has-unit': True, 'extent': [":", 3], 'required': True, 'description': 'empty'},
                    'nd-same-shape-arr': {'type': 'float', 'has-unit': True, 'extent': [':', 2, 3], 'required': True, 'description': 'empty'},
                    'nd-diff-shape-arr': {'type': 'float', 'has-unit': True, 'extent': [':', ':', ':'], 'required': True, 'description': 'empty'},
                }

            database.insert_property_definition(property_definition)
            
            get_def = database.get_property_definition('default')['definition']
            get_def['property-id'] = 'default'

            assert  get_def == property_definition


    def test_settings_setter_getter(self):
        with tempfile.TemporaryFile() as tmpfile:
            database = MongoDatabase(self.database_name, drop_database=True,configuration_type=AtomicConfiguration)

            dummy_file_contents = 'this is a dummy file\nwith nonsense contents'

            pso = PropertySettings(
                method='VASP',
                description='A basic test calculation',
                files=[('dummy_name', dummy_file_contents)],
            )

            pso_id = database.insert_property_settings(pso)

            rebuilt_pso = database.get_property_settings(pso_id)

            assert pso == rebuilt_pso


    def test_settings_duplicate(self):
        with tempfile.TemporaryFile() as tmpfile:
            database = MongoDatabase(self.database_name, drop_database=True,configuration_type=AtomicConfiguration)

            dummy_file_contents = 'this is a dummy file\nwith nonsense contents'

            pso = PropertySettings(
                method='VASP',
                description='A basic test calculation',
                files=[('dummy_name', dummy_file_contents)],
            )

            pso_id = database.insert_property_settings(pso)
            pso_id = database.insert_property_settings(pso)

            rebuilt_pso = database.get_property_settings(pso_id)

            assert pso == rebuilt_pso

class TestConfigurationSets:

    database_name = 'colabfit_test'

    def test_insert_cs(self):
        with tempfile.TemporaryFile() as tmpfile:
            database = MongoDatabase(self.database_name, drop_database=True,configuration_type=AtomicConfiguration)

            images = build_n(10)[0]

            ids = [_[0] for _ in database.insert_data(images)]

            cs_id = database.insert_configuration_set(
                ids, description='A basic configuration set'
            )

            desc = next(database.configuration_sets.find({'_id': cs_id}))['description']

            assert desc == 'A basic configuration set'

            rebuilt_ids = next(database.configuration_sets.find({'_id': cs_id}))['relationships']['configurations']

            # rebuilt_ids = database.get_data(
            #     f'configuration_sets/{cs_id}/ids',
            #     ravel=True, in_memory=True, as_str=True
            # ).tolist()
            assert rebuilt_ids.sort() == ids.sort()

            for img in images:
                img2 = database.get_configuration(
                    ID_FORMAT_STRING.format('CO', hash(img), 0)
                )

                assert img == img2

class TestDatasets:
    database_name = 'colabfit_test'

    def test_insert_ds(self):
        with tempfile.TemporaryFile() as tmpfile:
            database = MongoDatabase(self.database_name, drop_database=True, configuration_type=AtomicConfiguration)

            images = build_n(10)[0]

            database.insert_property_definition(
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
                'default': [{
                    'energy': {'field': 'energy', 'units': 'eV'},
                    'stress': {'field': 'stress', 'units': 'GPa'},
                    'name': {'field': 'name', 'units': None},
                    'nd-same-shape': {'field': 'nd-same-shape', 'units': 'eV'},
                    'nd-diff-shapes': {'field': 'nd-diff-shapes', 'units': 'eV'},
                    'forces': {'field': 'forces', 'units': 'eV/Ang'},
                    'nd-same-shape-arr': {'field': 'nd-same-shape-arr', 'units': 'eV/Ang'},
                    'nd-diff-shapes-arr': {'field': 'nd-diff-shapes-arr', 'units': 'eV/Ang'},
                }]
            }

            ids = database.insert_data(
                images, property_map=property_map
            )

            co_ids1, pr_ids1 = list(zip(*ids))

            cs_id1 = database.insert_configuration_set(
                co_ids1, description='A basic configuration set'
            )

            images = build_n(10)[0]

            for img in images:
                img.info['energy'] += 100000

            ids = database.insert_data(
                images, property_map=property_map
            )

            co_ids2, pr_ids2 = list(zip(*ids))

            cs_id2 = database.insert_configuration_set(
                co_ids2, description='A basic configuration set'
            )

            ds_id = database.insert_dataset(
                cs_ids=[cs_id1, cs_id2],
                pr_ids=pr_ids1+pr_ids2,
                name='example_dataset',
                authors='colabfit',
                links='https://colabfit.openkim.org/',
                description='An example dataset',
                resync=True
            )

            ds_doc = next(database.datasets.find({'_id': ds_id}))
            assert ds_doc['authors'] == ['colabfit']