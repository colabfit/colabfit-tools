import os
import pytest
import tempfile
import numpy as np

from ase import Atoms

from colabfit import (
    ATOMS_NAME_FIELD, ATOMS_LABELS_FIELD
)
from colabfit.tools.database import ConcatenationException, Database
from colabfit.tools.configuration import Configuration
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


class TestAddingConfigurations:

    def test_adding_configurations_no_properties(self):
        with tempfile.TemporaryFile() as tmpfile:

            database = Database(tmpfile, mode='w')

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

            database.insert_data(images)

            database.concatenate_group('configurations/atomic_numbers')
            database.concatenate_group('configurations/positions')
            database.concatenate_group('configurations/cells')
            database.concatenate_group('configurations/pbcs')

            np.testing.assert_allclose(
                database.get_data('configurations/atomic_numbers'),
                np.concatenate([_.get_atomic_numbers() for _ in images])
            )
            np.testing.assert_allclose(
                database.get_data('configurations/positions'),
                np.concatenate([_.get_positions() for _ in images])
            )
            np.testing.assert_allclose(
                database.get_data('configurations/cells'),
                np.concatenate([_.get_cell() for _ in images])
            )
            np.testing.assert_allclose(
                database.get_data('configurations/pbcs'),
                np.concatenate([_.get_pbc() for _ in images]).astype(int)
            )


    def test_add_then_update_nochange_config(self):
        with tempfile.TemporaryFile() as tmpfile:

            database = Database(tmpfile, mode='w')

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

            database.insert_data(images)
            database.insert_data(images)

            for co_data in database.get_data('configurations/names').values():
                assert len(co_data) == 0

            for co_data in database.get_data('configurations/labels').values():
                assert len(co_data) == 0

            for co_data in database.get_data('configurations/last_modified').values():
                assert len(co_data) == 1

            database.concatenate_group('configurations/atomic_numbers')
            database.concatenate_group('configurations/positions')
            database.concatenate_group('configurations/cells')
            database.concatenate_group('configurations/pbcs')

            np.testing.assert_allclose(
                database.get_data('configurations/atomic_numbers'),
                np.concatenate([_.get_atomic_numbers() for _ in images])
            )
            np.testing.assert_allclose(
                database.get_data('configurations/positions'),
                np.concatenate([_.get_positions() for _ in images])
            )
            np.testing.assert_allclose(
                database.get_data('configurations/cells'),
                np.concatenate([_.get_cell() for _ in images])
            )
            np.testing.assert_allclose(
                database.get_data('configurations/pbcs'),
                np.concatenate([_.get_pbc() for _ in images]).astype(int)
            )

    def test_add_then_update_with_changes_config(self):
        with tempfile.TemporaryFile() as tmpfile:

            database = Database(tmpfile, mode='w')

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

            database.insert_data(images)

            for img in images:
                img.info[ATOMS_NAME_FIELD] += 'change'
                img.info[ATOMS_LABELS_FIELD].add('another_label')

            database.insert_data(images)

            for n in database.get_data('configurations/names').values():
                assert n[()] == b'change'

            for n in database.get_data('configurations/labels').values():
                assert n[()] == b'another_label'

            for img in images:
                img.info[ATOMS_NAME_FIELD] = 'change2'
                img.info[ATOMS_LABELS_FIELD] = {'another_label2'}

            database.insert_data(images)

            for n in database.get_data('configurations/names').values():
                assert n[0] == b'change'
                assert n[1] == b'change2'

            for n in database.get_data('configurations/labels').values():
                assert n[0] == b'another_label'
                assert n[1] == b'another_label2'

            database.concatenate_group('configurations/atomic_numbers')
            database.concatenate_group('configurations/positions')
            database.concatenate_group('configurations/cells')
            database.concatenate_group('configurations/pbcs')

            np.testing.assert_allclose(
                database.get_data('configurations/atomic_numbers'),
                np.concatenate([_.get_atomic_numbers() for _ in images])
            )
            np.testing.assert_allclose(
                database.get_data('configurations/positions'),
                np.concatenate([_.get_positions() for _ in images])
            )
            np.testing.assert_allclose(
                database.get_data('configurations/cells'),
                np.concatenate([_.get_cell() for _ in images])
            )
            np.testing.assert_allclose(
                database.get_data('configurations/pbcs'),
                np.concatenate([_.get_pbc() for _ in images]).astype(int)
            )


    def test_adding_configurations_no_properties_gen(self):
        with tempfile.TemporaryFile() as tmpfile:

            database = Database(tmpfile, mode='w')

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

            ids = database.insert_data(images, generator=True)

            list(ids)  # consume generator

            database.concatenate_group('configurations/atomic_numbers')
            database.concatenate_group('configurations/positions')
            database.concatenate_group('configurations/cells')
            database.concatenate_group('configurations/pbcs')

            np.testing.assert_allclose(
                database.get_data('configurations/atomic_numbers'),
                np.concatenate([_.get_atomic_numbers() for _ in images])
            )
            np.testing.assert_allclose(
                database.get_data('configurations/positions'),
                np.concatenate([_.get_positions() for _ in images])
            )
            np.testing.assert_allclose(
                database.get_data('configurations/cells'),
                np.concatenate([_.get_cell() for _ in images])
            )
            np.testing.assert_allclose(
                database.get_data('configurations/pbcs'),
                np.concatenate([_.get_pbc() for _ in images]).astype(int)
            )


    def test_adding_configurations_with_properties_gen(self):
        with tempfile.TemporaryFile() as tmpfile:

            database = Database(tmpfile, mode='w')

            returns = build_n(2)

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

            ids = database.insert_data(
                images, property_map=property_map, generator=True
            )

            list(ids)  # consume generator

            database.concatenate_group('properties/default/energy')
            database.concatenate_group('properties/default/stress')
            database.concatenate_group('properties/default/name')
            database.concatenate_group('properties/default/nd-same-shape')

            database.concatenate_group('properties/default/forces')
            database.concatenate_group('properties/default/nd-same-shape-arr')

            with pytest.raises(ConcatenationException):
                database.concatenate_group('properties/default/nd-diff-shapes')

            with pytest.raises(ConcatenationException):
                database.concatenate_group('properties/default/nd-diff-shapes-arr')

            np.testing.assert_allclose(
                database.get_data('properties/default/energy'),
                np.hstack(energies)
            )
            np.testing.assert_allclose(
                database.get_data('properties/default/stress'),
                np.hstack(stress)
            )
            decoded_names = [
                _.decode('utf-8')
                for _ in database.get_data('properties/default/name')
            ]
            assert decoded_names == names
            np.testing.assert_allclose(
                database.get_data('properties/default/nd-same-shape'),
                np.concatenate(nd_same_shape)
            )
            data = database.get_data('properties/default/nd-diff-shapes')
            for a1, a2 in zip(data.values(), nd_diff_shape):
                np.testing.assert_allclose(a1, a2)

            np.testing.assert_allclose(
                database.get_data('properties/default/forces'),
                np.concatenate(forces)
            )
            np.testing.assert_allclose(
                database.get_data('properties/default/nd-same-shape-arr'),
                np.concatenate(nd_same_shape_arr)
            )
            data = database.get_data('properties/default/nd-diff-shapes-arr')
            for a1, a2 in zip(data.values(), nd_diff_shape_arr):
                np.testing.assert_allclose(a1, a2)


    def test_add_then_update_with_properties_nochange(self):
        with tempfile.TemporaryFile() as tmpfile:

            database = Database(tmpfile, mode='w')

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

            pso_id = database.insert_property_settings(pso)

            database.insert_data(
                images,
                property_map=property_map,
                property_settings={'default': pso_id}
            )

            database.insert_data(
                images,
                property_map=property_map,
                property_settings={'default': pso_id}
            )

            database.concatenate_group('properties/default/energy')
            database.concatenate_group('properties/default/stress')
            database.concatenate_group('properties/default/name')
            database.concatenate_group('properties/default/nd-same-shape')

            database.concatenate_group('properties/default/forces')
            database.concatenate_group('properties/default/nd-same-shape-arr')

            with pytest.raises(ConcatenationException):
                database.concatenate_group('properties/default/nd-diff-shapes')

            with pytest.raises(ConcatenationException):
                database.concatenate_group('properties/default/nd-diff-shapes-arr')

            np.testing.assert_allclose(
                database.get_data('properties/default/energy'),
                np.hstack(energies)
            )
            np.testing.assert_allclose(
                database.get_data('properties/default/stress'),
                np.hstack(stress)
            )
            decoded_names = [
                _.decode('utf-8')
                for _ in database.get_data('properties/default/name')
            ]
            assert decoded_names == names
            np.testing.assert_allclose(
                database.get_data('properties/default/nd-same-shape'),
                np.concatenate(nd_same_shape)
            )
            data = database.get_data('properties/default/nd-diff-shapes')
            for a1, a2 in zip(data.values(), nd_diff_shape):
                np.testing.assert_allclose(a1, a2)

            np.testing.assert_allclose(
                database.get_data('properties/default/forces'),
                np.concatenate(forces)
            )
            np.testing.assert_allclose(
                database.get_data('properties/default/nd-same-shape-arr'),
                np.concatenate(nd_same_shape_arr)
            )
            data = database.get_data('properties/default/nd-diff-shapes-arr')
            for a1, a2 in zip(data.values(), nd_diff_shape_arr):
                np.testing.assert_allclose(a1, a2)


    def test_add_then_update_with_properties_with_change(self):
        with tempfile.TemporaryFile() as tmpfile:

            database = Database(tmpfile, mode='w')

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

            pso_id = database.insert_property_settings(pso)

            database.insert_data(
                images,
                property_map=property_map,
                property_settings={'default': pso_id}
            )

            for img in images:
                img.info['energy'] += 100000

            database.insert_data(
                images,
                property_map=property_map,
                property_settings={'default': pso_id}
            )

            for img in images:
                img.info['energy'] += 100000

            database.insert_data(
                images,
                property_map=property_map,
                property_settings={'default': pso_id}
            )

            database.concatenate_group('properties/default/energy')
            database.concatenate_group('properties/default/stress')
            database.concatenate_group('properties/default/name')
            database.concatenate_group('properties/default/nd-same-shape')

            database.concatenate_group('properties/default/forces')
            database.concatenate_group('properties/default/nd-same-shape-arr')

            with pytest.raises(ConcatenationException):
                database.concatenate_group('properties/default/nd-diff-shapes')

            with pytest.raises(ConcatenationException):
                database.concatenate_group('properties/default/nd-diff-shapes-arr')

            np.testing.assert_allclose(
                database.get_data('properties/default/energy'),
                np.concatenate([
                    np.hstack(energies),
                    np.hstack(energies)+100000,
                    np.hstack(energies)+200000
                ])
            )
            np.testing.assert_allclose(
                database.get_data('properties/default/stress'),
                np.concatenate([
                    np.hstack(stress),
                    np.hstack(stress),
                    np.hstack(stress),
                ])
            )
            decoded_names = [
                _.decode('utf-8')
                for _ in database.get_data('properties/default/name')
            ]
            assert decoded_names == names*3
            np.testing.assert_allclose(
                database.get_data('properties/default/nd-same-shape'),
                np.concatenate([
                    np.concatenate(nd_same_shape),
                    np.concatenate(nd_same_shape),
                    np.concatenate(nd_same_shape),
                ])
            )
            data = database.get_data('properties/default/nd-diff-shapes')
            for a1, a2 in zip(data.values(), nd_diff_shape*3):
                np.testing.assert_allclose(a1, a2)

            np.testing.assert_allclose(
                database.get_data('properties/default/forces'),
                np.concatenate([
                    np.concatenate(forces),
                    np.concatenate(forces),
                    np.concatenate(forces),
                ])
            )
            np.testing.assert_allclose(
                database.get_data('properties/default/nd-same-shape-arr'),
                np.concatenate([
                    np.concatenate(nd_same_shape_arr),
                    np.concatenate(nd_same_shape_arr),
                    np.concatenate(nd_same_shape_arr),
                ])
            )
            data = database.get_data('properties/default/nd-diff-shapes-arr')
            for a1, a2 in zip(data.values(), nd_diff_shape_arr*3):
                np.testing.assert_allclose(a1, a2)

    def test_adding_configurations_with_properties(self):
        with tempfile.TemporaryFile() as tmpfile:

            database = Database(tmpfile, mode='w')

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

            pso_id = database.insert_property_settings(pso)

            database.insert_data(
                images,
                property_map=property_map,
                property_settings={'default': pso_id}
            )

            database.concatenate_group('properties/default/energy')
            database.concatenate_group('properties/default/stress')
            database.concatenate_group('properties/default/name')
            database.concatenate_group('properties/default/nd-same-shape')

            database.concatenate_group('properties/default/forces')
            database.concatenate_group('properties/default/nd-same-shape-arr')

            with pytest.raises(ConcatenationException):
                database.concatenate_group('properties/default/nd-diff-shapes')

            with pytest.raises(ConcatenationException):
                database.concatenate_group('properties/default/nd-diff-shapes-arr')

            np.testing.assert_allclose(
                database.get_data('properties/default/energy'),
                np.hstack(energies)
            )
            np.testing.assert_allclose(
                database.get_data('properties/default/stress'),
                np.hstack(stress)
            )
            decoded_names = [
                _.decode('utf-8')
                for _ in database.get_data('properties/default/name')
            ]
            assert decoded_names == names
            np.testing.assert_allclose(
                database.get_data('properties/default/nd-same-shape'),
                np.concatenate(nd_same_shape)
            )
            data = database.get_data('properties/default/nd-diff-shapes')
            for a1, a2 in zip(data.values(), nd_diff_shape):
                np.testing.assert_allclose(a1, a2)

            np.testing.assert_allclose(
                database.get_data('properties/default/forces'),
                np.concatenate(forces)
            )
            np.testing.assert_allclose(
                database.get_data('properties/default/nd-same-shape-arr'),
                np.concatenate(nd_same_shape_arr)
            )
            data = database.get_data('properties/default/nd-diff-shapes-arr')
            for a1, a2 in zip(data.values(), nd_diff_shape_arr):
                np.testing.assert_allclose(a1, a2)


    def test_config_add_concat_add_concat(self):

        with tempfile.TemporaryFile() as tmpfile:

            database = Database(tmpfile, mode='w')

            images1 = build_n(10)[0]
            atomic_numbers = [atoms.get_atomic_numbers() for atoms in images1]
            positions = [atoms.get_positions() for atoms in images1]
            cells = [atoms.get_cell() for atoms in images1]
            pbcs = [atoms.get_pbc() for atoms in images1]
            database.insert_data(images1)

            database.concatenate_group('configurations/atomic_numbers')

            images2 = build_n(10)[0]
            atomic_numbers += [atoms.get_atomic_numbers() for atoms in images2]
            positions += [atoms.get_positions() for atoms in images2]
            cells += [atoms.get_cell() for atoms in images2]
            pbcs += [atoms.get_pbc() for atoms in images2]
            database.insert_data(images2)

            database.concatenate_group('configurations/atomic_numbers')

            np.testing.assert_allclose(
                database.get_data('configurations/atomic_numbers'),
                np.concatenate(atomic_numbers)
            )
            np.testing.assert_allclose(
                database.get_data(
                    'configurations/positions', concatenate=True, in_memory=True
                ),
                np.concatenate(positions)
            )
            np.testing.assert_allclose(
                database.get_data(
                    'configurations/cells', concatenate=True, in_memory=True
                ),
                np.concatenate(cells)
            )
            np.testing.assert_allclose(
                database.get_data(
                    'configurations/pbcs', concatenate=True, in_memory=True
                ),
                np.concatenate(pbcs)
            )


    def test_get_configurations(self):

        with tempfile.TemporaryFile() as tmpfile:
            database = Database(tmpfile, mode='w')

            images = build_n(10)[0]

            database.insert_data(images)

            # database.concatenate_configurations()

            count = 0
            for atoms, img in zip(database.get_configurations('all'), images):
                assert atoms == img
                count += 1
            assert count == 10


    def test_get_configurations_after_concat(self):

        with tempfile.TemporaryFile() as tmpfile:
            database = Database(tmpfile, mode='w')

            images = build_n(10)[0]

            database.insert_data(images)

            database.concatenate_configurations()

            count = 0
            for atoms, img in zip(database.get_configurations('all'), images):
                assert atoms == img
                count += 1
            assert count == 10


    def test_get_configurations_after_concat_gen(self):

        with tempfile.TemporaryFile() as tmpfile:
            database = Database(tmpfile, mode='w')

            images = build_n(10)[0]

            ids = database.insert_data(images, generator=True)
            list(ids)  # consume generator

            database.concatenate_configurations()

            count = 0
            for atoms, img in zip(
                database.get_configurations('all', generator=True), images
                ):
                assert atoms == img
                count += 1
            assert count == 10


    def test_get_using_returns(self):
        with tempfile.TemporaryFile() as tmpfile:
            database = Database(tmpfile, mode='w')

            returns = build_n(10)

            images = returns[0]
            ids    = [_[1] for _ in database.insert_data(images)]

            database.concatenate_configurations()

            for atoms, img in zip(database.get_configurations(ids), images):
                assert atoms == img


    def test_get_using_returns_gen(self):
        with tempfile.TemporaryFile() as tmpfile:
            database = Database(tmpfile, mode='w')

            returns = build_n(10)

            images = returns[0]
            ids    = [
                _[1] for _ in database.insert_data(images, generator=True)
            ]

            database.concatenate_configurations()

            for atoms, img in zip(
                database.get_configurations(ids, generator=True), images
                ):
                assert atoms == img


class TestPropertyDefinitions:

    def test_invalid_definition(self):
        with tempfile.TemporaryFile() as tmpfile:
            database = Database(tmpfile, mode='w')

            property_definition = {
                'property-id': 'this should throw an error',
            }

            with pytest.raises(Exception):
                database.insert_property_definition(property_definition)


    def test_definition_setter_getter(self):
        with tempfile.TemporaryFile() as tmpfile:
            database = Database(tmpfile, mode='w')

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

            assert database.get_property_definition('default') == property_definition


    def test_settings_setter_getter(self):
        with tempfile.TemporaryFile() as tmpfile:
            database = Database(tmpfile, mode='w')

            dummy_file_contents = 'this is a dummy file\nwith nonsense contents'

            pso = PropertySettings(
                method='VASP',
                description='A basic test calculation',
                files=[('dummy_name', dummy_file_contents)],
            )

            pso_id = database.insert_property_settings(pso)

            rebuilt_pso = database.get_property_settings(pso_id)

            assert pso == rebuilt_pso