import re
import unittest
import numpy as np
from ase import Atoms

from core import ATOMS_NAME_FIELD, ATOMS_LABELS_FIELD
from core.configuration import Configuration
from core.dataset import Dataset

class TestDatasetConstruction(unittest.TestCase):
    def test_from_html(self):
        # Just makes sure from_markdown doesn't throw an error
        dataset = Dataset.from_markdown('tests/files/test.md')

    def test_config_setter(self):
        atoms = []
        for i in range(3):
            atoms.append(Atoms())
            atoms[-1].info[ATOMS_NAME_FIELD] = i

        dataset = Dataset(name='test')
        dataset.configurations = [Configuration(at) for at in atoms]

        self.assertEqual(
            ['0', '1', '2'],
            [c.atoms.info[ATOMS_NAME_FIELD] for c in dataset.configurations]
        )

        del dataset.configurations[1]

        self.assertEqual(
            ['0', '2'],
            [c.atoms.info[ATOMS_NAME_FIELD] for c in dataset.configurations]
        )

        new_atoms = Atoms()
        new_atoms.info[ATOMS_NAME_FIELD] = 4
        dataset.configurations += [Configuration(new_atoms)]

        self.assertEqual(
            ['0', '2', '4'],
            [c.atoms.info[ATOMS_NAME_FIELD] for c in dataset.configurations]
        )

    def test_co_label_refresher(self):

        dataset = Dataset(name='test')
        atoms = []
        for ii in range(10):
            atoms.append(Atoms())
            atoms[-1].info[ATOMS_NAME_FIELD] = ii
        dataset.configurations = [Configuration(at) for at in atoms]

        dataset.co_label_regexes = {
            '[0-4]': {'0_to_4'},
            '[5-9]': {'5_to_9'},
        }

        # Make sure label refresh is working
        for ai, conf in enumerate(dataset.configurations):
            self.assertEqual(conf.atoms.info[ATOMS_NAME_FIELD], f"{ai}")
            if ai < 5:
                self.assertSetEqual(conf.atoms.info[ATOMS_LABELS_FIELD], {'0_to_4'})
            elif ai >= 5:
                self.assertSetEqual(conf.atoms.info[ATOMS_LABELS_FIELD], {'5_to_9'})


        # Make sure that a new CO gets its labels updated
        new_atoms = Atoms()
        new_atoms.info[ATOMS_NAME_FIELD] = '4'
        dataset.configurations += [Configuration(new_atoms)]

        self.assertSetEqual(
            dataset.configurations[-1].atoms.info[ATOMS_LABELS_FIELD], set()
        )

        dataset.refresh_config_labels()

        self.assertSetEqual(
            dataset.configurations[-1].atoms.info[ATOMS_LABELS_FIELD], {'0_to_4'}
        )


    def test_cs_refresher(self):

        dataset = Dataset(name='test')
        atoms = []
        for ii in range(10):
            atoms.append(Atoms())
            atoms[-1].info[ATOMS_NAME_FIELD] = ii
        dataset.configurations = [Configuration(at)for at in atoms]

        dataset.cs_regexes = {}

        for cs in dataset.configuration_sets:
            for conf in cs.configurations:
                if int(conf.atoms.info[ATOMS_NAME_FIELD][-1]) < 5:
                    self.assertEqual(cs.description, 'Default configuration set')
                else:
                    self.assertEqual(cs.description, 'Default configuration set')

        dataset.cs_regexes = {
            'default': 'default',
            '[0-4]': '0_to_4',
            '[5-9]': '5_to_9',
        }

        for cs in dataset.configuration_sets:
            for conf in cs.configurations:
                if int(conf.atoms.info[ATOMS_NAME_FIELD][-1]) < 5:
                    self.assertEqual(cs.description, '0_to_4')
                else:
                    self.assertEqual(cs.description, '5_to_9')

        dataset.cs_regexes['[3-7]'] = '3_to_7'

        self.assertEqual(len(dataset.configuration_sets), 2)

        dataset.refresh_config_sets()

        self.assertEqual(len(dataset.configuration_sets), 3)


    def test_labels_updated_everywhere(self):

        dataset = Dataset(name='test')
        atoms = []
        for ii in range(10):
            atoms.append(Atoms())
            atoms[-1].info[ATOMS_NAME_FIELD] = ii
        dataset.configurations = [Configuration(at)for at in atoms]

        dataset.cs_regexes = {
            'default': 'default',
            '[0-4]': '0_to_4',
            '[5-9]': '5_to_9',
        }

        # Ensure labels are initially empty
        for cs in dataset.configuration_sets:
            for conf in cs.configurations:
                self.assertSetEqual(set(), conf.atoms.info[ATOMS_LABELS_FIELD])

        # Make sure they're re-written correctly
        dataset.co_label_regexes = {
            '[0-4]': {'0_to_4'},
            '[5-9]': {'5_to_9'},
        }

        for cs in dataset.configuration_sets:
            if cs.description == '0_to_4':
                self.assertSetEqual({'0_to_4'}, set(cs.labels))
            elif cs.description == '5_to_9':
                self.assertSetEqual({'5_to_9'}, set(cs.labels))

            for conf in cs.configurations:
                if int(conf.atoms.info[ATOMS_NAME_FIELD][-1]) < 5:
                    self.assertSetEqual(
                        {'0_to_4'}, conf.atoms.info[ATOMS_LABELS_FIELD]
                    )
                else:
                    self.assertSetEqual(
                        {'5_to_9'}, conf.atoms.info[ATOMS_LABELS_FIELD]
                    )

        # Make sure they're added to correctly
        dataset.co_label_regexes['[0-9]'] = {'new_label'}
        dataset.resync()

        for cs in dataset.configuration_sets:
            for conf in cs.configurations:
                if int(conf.atoms.info[ATOMS_NAME_FIELD][-1]) < 5:
                    self.assertSetEqual(
                        {'0_to_4', 'new_label'}, conf.atoms.info[ATOMS_LABELS_FIELD]
                    )
                else:
                    self.assertSetEqual(
                        {'5_to_9', 'new_label'}, conf.atoms.info[ATOMS_LABELS_FIELD]
                    )

        # And also removed properly
        dataset.delete_config_label_regex('[0-4]')
        dataset.resync()

        for cs in dataset.configuration_sets:
            for conf in cs.configurations:
                if int(conf.atoms.info[ATOMS_NAME_FIELD][-1]) < 5:
                    self.assertSetEqual(
                        {'new_label'}, conf.atoms.info[ATOMS_LABELS_FIELD]
                    )
                else:
                    self.assertSetEqual(
                        {'5_to_9', 'new_label'}, conf.atoms.info[ATOMS_LABELS_FIELD]
                    )

class TestSetOperations(unittest.TestCase):

    def test_subset_is_subset(self):
        dataset1 = Dataset('test1')

        images1 = []
        for i in range(5):
            images1.append(Atoms('H2', positions=np.random.random((2, 3))))
            images1[-1].info[ATOMS_NAME_FIELD] = dataset1.name + str(i)
            images1[-1].info[ATOMS_LABELS_FIELD] = dataset1.name + '_label_'+ str(i)

            images1[-1].info['energy'] = float(i)

        dataset1.configurations = [Configuration(at) for at in images1]
        dataset1.property_map = {
            'energy': {'field': 'energy', 'units': 'eV'}
        }
        dataset1.load_data()

        dataset2 = Dataset('test1')
        dataset2.configurations = [Configuration(at) for at in images1[:3]]
        dataset2.property_map = {
            'energy': {'field': 'energy', 'units': 'eV'}
        }
        dataset2.load_data()

        self.assertFalse(dataset1.issubset(dataset2))
        self.assertTrue(dataset2.issubset(dataset1))


    def test_superset_is_subset(self):
        dataset1 = Dataset('test1')

        images1 = []
        for i in range(5):
            images1.append(Atoms('H2', positions=np.random.random((2, 3))))
            images1[-1].info[ATOMS_NAME_FIELD] = dataset1.name + str(i)
            images1[-1].info[ATOMS_LABELS_FIELD] = dataset1.name + '_label_'+ str(i)

            images1[-1].info['energy'] = float(i)

        dataset1.configurations = [Configuration(at) for at in images1]
        dataset1.property_map = {
            'energy': {'field': 'energy', 'units': 'eV'}
        }
        dataset1.load_data()

        dataset2 = Dataset('test1')
        dataset2.configurations = [Configuration(at) for at in images1[:3]]
        dataset2.property_map = {
            'energy': {'field': 'energy', 'units': 'eV'}
        }
        dataset2.load_data()

        self.assertFalse(dataset2.issuperset(dataset1))
        self.assertTrue(dataset1.issuperset(dataset2))


    def test_disjoint(self):
        dataset1 = Dataset('test1')

        images1 = []
        for i in range(5):
            images1.append(Atoms('H2', positions=np.random.random((2, 3))))
            images1[-1].info[ATOMS_NAME_FIELD] = dataset1.name + str(i)
            images1[-1].info[ATOMS_LABELS_FIELD] = dataset1.name + '_label_'+ str(i)

            images1[-1].info['energy'] = float(i)

        dataset1.configurations = [Configuration(at) for at in images1]
        dataset1.property_map = {
            'energy': {'field': 'energy', 'units': 'eV'}
        }
        dataset1.load_data()

        dataset2 = Dataset('test1')

        images2 = []
        for i in range(5):
            images2.append(Atoms('H2', positions=np.random.random((2, 3))))
            images2[-1].info[ATOMS_NAME_FIELD] = dataset2.name + str(i)
            images2[-1].info[ATOMS_LABELS_FIELD] = dataset2.name + '_label_'+ str(i)

            images2[-1].info['energy'] = float(i) + 20

        dataset2.configurations = [Configuration(at) for at in images2]
        dataset2.property_map = {
            'energy': {'field': 'energy', 'units': 'eV'}
        }
        dataset2.load_data()

        self.assertFalse(dataset1.issubset(dataset2))
        self.assertFalse(dataset2.issubset(dataset1))


    def test_parent_child(self):
        dataset1 = Dataset('test1')

        images1 = []
        for i in range(5):
            images1.append(Atoms('H2', positions=np.random.random((2, 3))))
            images1[-1].info[ATOMS_NAME_FIELD] = dataset1.name + str(i)
            images1[-1].info[ATOMS_LABELS_FIELD] = dataset1.name + '_label_'+ str(i)

            images1[-1].info['energy'] = float(i)

        dataset1.configurations = [Configuration(at) for at in images1]
        dataset1.property_map = {
            'energy': {'field': 'energy', 'units': 'eV'}
        }
        dataset1.load_data()

        dataset2 = Dataset('test1')

        images2 = []
        for i in range(5):
            images2.append(Atoms('H2', positions=np.random.random((2, 3))))
            images2[-1].info[ATOMS_NAME_FIELD] = dataset2.name + str(i)
            images2[-1].info[ATOMS_LABELS_FIELD] = dataset2.name + '_label_'+ str(i)

            images2[-1].info['energy'] = float(i) + 20

        dataset2.configurations = [Configuration(at) for at in images2]
        dataset2.property_map = {
            'energy': {'field': 'energy', 'units': 'eV'}
        }
        dataset2.load_data()

        parent = Dataset('parent')
        parent.attach_dataset(dataset1)

        self.assertTrue(dataset1.issubset(parent))
        self.assertTrue(parent.issubset(dataset1))

        self.assertFalse(dataset2.issubset(parent))
        self.assertFalse(parent.issubset(dataset2))

        parent.attach_dataset(dataset2)

        self.assertTrue(dataset1.issubset(parent))
        self.assertTrue(dataset2.issubset(parent))
        self.assertFalse(parent.issubset(dataset1))
        self.assertFalse(parent.issubset(dataset2))


    def test_parent_parent(self):
        dataset1 = Dataset('test1')

        images1 = []
        for i in range(5):
            images1.append(Atoms('H2', positions=np.random.random((2, 3))))
            images1[-1].info[ATOMS_NAME_FIELD] = dataset1.name + str(i)
            images1[-1].info[ATOMS_LABELS_FIELD] = dataset1.name + '_label_'+ str(i)

            images1[-1].info['energy'] = float(i)

        dataset1.configurations = [Configuration(at) for at in images1]
        dataset1.property_map = {
            'energy': {'field': 'energy', 'units': 'eV'}
        }
        dataset1.load_data()

        dataset2 = Dataset('test1')

        images2 = []
        for i in range(5):
            images2.append(Atoms('H2', positions=np.random.random((2, 3))))
            images2[-1].info[ATOMS_NAME_FIELD] = dataset2.name + str(i)
            images2[-1].info[ATOMS_LABELS_FIELD] = dataset2.name + '_label_'+ str(i)

            images2[-1].info['energy'] = float(i) + 20

        dataset2.configurations = [Configuration(at) for at in images2]
        dataset2.property_map = {
            'energy': {'field': 'energy', 'units': 'eV'}
        }
        dataset2.load_data()

        parent1 = Dataset('parent')
        parent1.attach_dataset(dataset1)

        parent2 = Dataset('parent')
        parent2.attach_dataset(dataset1)
        parent2.attach_dataset(dataset2)

        self.assertTrue(parent1.issubset(parent2))
        self.assertFalse(parent2.issubset(parent1))


    def test_equality(self):
        dataset1 = Dataset('test1')

        images1 = []
        for i in range(5):
            images1.append(Atoms('H2', positions=np.random.random((2, 3))))
            images1[-1].info[ATOMS_NAME_FIELD] = dataset1.name + str(i)
            images1[-1].info[ATOMS_LABELS_FIELD] = dataset1.name + '_label_'+ str(i)

            images1[-1].info['energy'] = float(i)

        dataset1.configurations = [Configuration(at) for at in images1]
        dataset1.property_map = {
            'energy': {'field': 'energy', 'units': 'eV'}
        }
        dataset1.load_data()

        dataset2 = Dataset('test1')

        images2 = []
        for i in range(5):
            images2.append(Atoms('H2', positions=np.random.random((2, 3))))
            images2[-1].info[ATOMS_NAME_FIELD] = dataset2.name + str(i)
            images2[-1].info[ATOMS_LABELS_FIELD] = dataset2.name + '_label_'+ str(i)

            images2[-1].info['energy'] = float(i) + 20

        dataset2.configurations = [Configuration(at) for at in images2]
        dataset2.property_map = {
            'energy': {'field': 'energy', 'units': 'eV'}
        }
        dataset2.load_data()

        self.assertEqual(dataset1, dataset1)
        self.assertNotEqual(dataset1, dataset2)


class TestFilter(unittest.TestCase):
    def test_filter_on_co_names(self):
        dataset = Dataset('test')

        images = []
        for i in range(5):
            images.append(Atoms('H2', positions=np.random.random((2, 3))))
            images[-1].info[ATOMS_NAME_FIELD] = dataset.name + str(i)
            images[-1].info[ATOMS_LABELS_FIELD] = dataset.name + '_label_'+ str(i)

            images[-1].info['energy'] = float(i)

        dataset.property_map = {
            'energy': {'field': 'energy', 'units': 'eV'}
        }

        dataset.configurations = [Configuration(at) for at in images]

        regex = re.compile('test[0-3]')
        filtered = dataset.filter(
            'configurations',
            lambda c: regex.search(c.atoms.info[ATOMS_NAME_FIELD])
        )

        self.assertEqual(len(filtered.configurations), 4)

        dataset.load_data()

        regex = re.compile('test[0-3]')
        filtered = dataset.filter(
            'configurations',
            lambda c: regex.search(c.atoms.info[ATOMS_NAME_FIELD])
        )

        self.assertEqual(len(filtered.configurations), 4)
        self.assertEqual(len(filtered.data), 4)

        regex = re.compile('.*')
        filtered = dataset.filter(
            'configurations',
            lambda c: regex.search(c.atoms.info[ATOMS_NAME_FIELD])
        )

        self.assertEqual(dataset, filtered)