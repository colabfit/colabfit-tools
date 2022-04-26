import unittest
import numpy as np
from ase import Atoms

from colabfit import ATOMS_CONSTRAINTS_FIELD, ATOMS_NAME_FIELD, ATOMS_LABELS_FIELD
from colabfit.tools.configuration import AtomicConfiguration


class TestConfigurations(unittest.TestCase):
    def test_empty_constructor(self):
        conf = AtomicConfiguration()

        self.assertEqual(conf.info[ATOMS_NAME_FIELD], set())
        self.assertSetEqual(conf.info[ATOMS_LABELS_FIELD], set())


    def test_from_ase(self):
        atoms = Atoms('H4', pbc=True)
        atoms.info[ATOMS_NAME_FIELD] = 'test'
        labels = {'label1', 'label2'}
        atoms.info[ATOMS_LABELS_FIELD] = labels

        natoms = len(atoms)

        cell = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        pos = np.random.random((natoms, 3))

        atoms.cell = cell
        atoms.positions = pos

        conf = AtomicConfiguration.from_ase(atoms)

        # self.assertEqual(conf, atoms)

        self.assertEqual(conf.info[ATOMS_NAME_FIELD], {'test'})
        self.assertSetEqual(conf.info[ATOMS_LABELS_FIELD], labels)

        np.testing.assert_allclose(conf.arrays['positions'], pos)
        np.testing.assert_allclose(conf.arrays['numbers'], np.ones(natoms))
        np.testing.assert_allclose(np.array(conf.cell), cell)
        np.testing.assert_allclose(np.array(conf.pbc), np.array([True]*3))


    def test_hashing_identical(self):
        atoms = Atoms('H4', pbc=True)
        atoms.info[ATOMS_NAME_FIELD] = 'test'
        labels = {'label1', 'label2'}
        atoms.info[ATOMS_LABELS_FIELD] = labels

        natoms = len(atoms)

        cell = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        pos = np.random.random((natoms, 3))

        atoms.cell = cell
        atoms.positions = pos

        conf1 = AtomicConfiguration.from_ase(atoms)
        conf2 = AtomicConfiguration.from_ase(atoms)

        self.assertEqual(hash(conf1), hash(conf2))


    def test_hashing_different(self):
        atoms = Atoms('H4', pbc=True)
        atoms.info[ATOMS_NAME_FIELD] = 'test'
        labels = {'label1', 'label2'}
        atoms.info[ATOMS_LABELS_FIELD] = labels

        natoms = len(atoms)

        cell = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        pos = np.random.random((natoms, 3))

        atoms.cell = cell
        atoms.positions = pos

        atoms2 = Atoms(atoms)
        atoms2.positions += 1

        atoms3 = Atoms(atoms2)
        atoms3.positions = atoms.positions

        conf1 = AtomicConfiguration.from_ase(atoms)
        conf2 = AtomicConfiguration.from_ase(atoms2)
        conf3 = AtomicConfiguration.from_ase(atoms3)

        self.assertNotEqual(hash(conf1), hash(conf2))
        self.assertEqual(hash(conf1), hash(conf3))


    # def test_constrained_hashing_same(self):
    #     atoms = Atoms('H4', pbc=True)
    #     atoms.info[ATOMS_NAME_FIELD] = 'test'

    #     labels = {'label1', 'label2'}
    #     atoms.info[ATOMS_LABELS_FIELD] = labels

    #     atoms.info[ATOMS_CONSTRAINTS_FIELD] = {ATOMS_LABELS_FIELD}

    #     natoms = len(atoms)

    #     cell = np.array([
    #         [1, 0, 0],
    #         [0, 1, 0],
    #         [0, 0, 1]
    #     ])
    #     pos = np.random.random((natoms, 3))

    #     atoms.cell = cell
    #     atoms.positions = pos

    #     conf1 = Configuration.from_ase(atoms)
    #     conf2 = Configuration.from_ase(atoms)

    #     self.assertEqual(hash(conf1), hash(conf2))


    # def test_constrained_hashing_diff(self):
    #     atoms = Atoms('H4', pbc=True)
    #     atoms.info[ATOMS_NAME_FIELD] = 'test'
    #     labels = {'label1', 'label2'}
    #     atoms.info[ATOMS_LABELS_FIELD] = labels

    #     atoms.info[ATOMS_CONSTRAINTS_FIELD] = {ATOMS_LABELS_FIELD}

    #     natoms = len(atoms)

    #     cell = np.array([
    #         [1, 0, 0],
    #         [0, 1, 0],
    #         [0, 0, 1]
    #     ])
    #     pos = np.random.random((natoms, 3))

    #     atoms.cell = cell
    #     atoms.positions = pos

    #     atoms2 = Atoms(atoms)

    #     atoms2.info[ATOMS_LABELS_FIELD] = {'a_different_label'}

    #     conf1 = Configuration.from_ase(atoms)
    #     conf2 = Configuration.from_ase(atoms2)

    #     self.assertNotEqual(hash(conf1), hash(conf2))