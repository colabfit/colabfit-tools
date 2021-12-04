import os
import pytest
import numpy as np

from ase import Atoms

from colabfit.tools.database import ConcatenationException, Database
from colabfit.tools.configuration import Configuration


class TestDatabase:
    @staticmethod
    def add_n(database, n):
        energies            = []
        stress              = []
        names               = []
        nd_same_shape       = []
        nd_diff_shape       = []
        forces              = []
        nd_same_shape_arr   = []
        nd_diff_shape_arr   = []

        images = []
        for i in range(1, n+1):
            atoms = Atoms(f'H{i}', positions=np.random.random((i, 3)))

            atoms.info['energy'] = np.random.random()
            atoms.info['stress'] = np.random.random(6)
            atoms.info['name'] = f'configuration_{i}'
            atoms.info['nd-same-shape'] = np.random.random((2, 3, 5))
            atoms.info['nd-diff-shapes'] = np.random.random((
                i+np.random.randint(1, 4),
                i+np.random.randint(1, 4),
                i+np.random.randint(1, 4),
            ))

            energies.append(atoms.info['energy'])
            stress.append(atoms.info['stress'])
            names.append(atoms.info['name'])
            nd_same_shape.append(atoms.info['nd-same-shape'])
            nd_diff_shape.append(atoms.info['nd-diff-shapes'])

            atoms.arrays['forces'] = np.random.random((i, 3))
            atoms.arrays['nd-same-shape'] = np.random.random((i, 2, 3))
            atoms.arrays['nd-diff-shapes'] = np.random.random((
                i,
                np.random.randint(1, 4),
                np.random.randint(1, 4),
            ))

            forces.append(atoms.arrays['forces'])
            nd_same_shape_arr.append(atoms.arrays['nd-same-shape'])
            nd_diff_shape_arr.append(atoms.arrays['nd-diff-shapes'])

            images.append(Configuration.from_ase(atoms))

        database.add_configurations(images)

        return (
            images,
            energies, stress, names, nd_same_shape, nd_diff_shape,
            forces, nd_same_shape_arr, nd_diff_shape_arr
        )


    def test_adding_configurations(self):
        database = Database('/tmp/test.hdf5', mode='w')

        returns = self.add_n(database, 2)

        energies            = returns[1]
        stress              = returns[2]
        names               = returns[3]
        nd_same_shape       = returns[4]
        nd_diff_shape       = returns[5]
        forces              = returns[6]
        nd_same_shape_arr   = returns[7]
        nd_diff_shape_arr   = returns[8]

        database.concatenate_group('configurations/info/energy')
        database.concatenate_group('configurations/info/stress')
        database.concatenate_group('configurations/info/name')
        database.concatenate_group('configurations/info/nd-same-shape')

        database.concatenate_group('configurations/arrays/forces')
        database.concatenate_group('configurations/arrays/nd-same-shape')

        with pytest.raises(ConcatenationException):
            database.concatenate_group('configurations/info/nd-diff-shapes')

        with pytest.raises(ConcatenationException):
            database.concatenate_group('configurations/arrays/nd-diff-shapes')

        np.testing.assert_allclose(
            database['configurations/info/energy/data'],
            np.hstack(energies)
        )
        np.testing.assert_allclose(
            database['configurations/info/stress/data'],
            np.hstack(stress)
        )
        decoded_names = [
            _.decode('utf-8') for _ in database['configurations/info/name/data']
        ]
        assert decoded_names == names
        np.testing.assert_allclose(
            database['configurations/info/nd-same-shape/data'],
            np.concatenate(nd_same_shape)
        )
        data = [
                _[()] for _ in
                database['configurations/info/nd-diff-shapes/data'].values()
            ]
        
        for a1, a2 in zip(data, nd_diff_shape):
            np.testing.assert_allclose(a1, a2)

        np.testing.assert_allclose(
            database['configurations/arrays/forces/data'],
            np.concatenate(forces)
        )
        np.testing.assert_allclose(
            database['configurations/arrays/nd-same-shape/data'],
            np.concatenate(nd_same_shape_arr)
        )
        data = [
                _[()] for _ in
                database['configurations/arrays/nd-diff-shapes/data'].values()
            ]
        
        for a1, a2 in zip(data, nd_diff_shape_arr):
            np.testing.assert_allclose(a1, a2)

        os.remove('/tmp/test.hdf5')

    
    def test_get_one_configuration(self):
        database = Database('/tmp/test.hdf5', mode='w')

        images = self.add_n(database, 2)[0]

        atoms = database.get_configuration(0)



    def test_get_one_configuration_after_concat(self):
        pass