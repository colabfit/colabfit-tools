import re
import unittest
import numpy as np
from ase import Atoms

from colabfit import ATOMS_NAME_FIELD, ATOMS_LABELS_FIELD
from colabfit.tools.configuration import Configuration
from colabfit.tools.dataset import Dataset, load_data
from colabfit.tools.property import MissingPropertyFieldWarning


class TestDatasetConstruction(unittest.TestCase):
    def test_from_html(self):
        # Just makes sure from_markdown doesn't throw an error
        dataset = Dataset.from_markdown('colabfit/tests/files/test.md')

        self.assertEqual(6, len(dataset.data))
        self.assertEqual(3, len(dataset.get_data('a-custom-field-name')))
        self.assertEqual(3, len(dataset.get_data('energy')))


    def test_config_setter(self):
        atoms = []
        for i in range(3):
            atoms.append(Atoms())
            atoms[-1].info[ATOMS_NAME_FIELD] = i

        dataset = Dataset(name='test')
        dataset.configurations = [Configuration.from_ase(at) for at in atoms]

        self.assertEqual(
            ['0', '1', '2'],
            [c.info[ATOMS_NAME_FIELD] for c in dataset.configurations]
        )

        del dataset.configurations[1]

        self.assertEqual(
            ['0', '2'],
            [c.info[ATOMS_NAME_FIELD] for c in dataset.configurations]
        )

        new_atoms = Atoms()
        new_atoms.info[ATOMS_NAME_FIELD] = 4
        dataset.configurations += [Configuration.from_ase(new_atoms)]

        self.assertEqual(
            ['0', '2', '4'],
            [c.info[ATOMS_NAME_FIELD] for c in dataset.configurations]
        )

    def test_co_label_refresher(self):

        dataset = Dataset(name='test')
        atoms = []
        for ii in range(10):
            atoms.append(Atoms())
            atoms[-1].info[ATOMS_NAME_FIELD] = ii

        dataset.configurations = [Configuration.from_ase(at) for at in atoms]

        dataset.configuration_label_regexes = {
            '[0-4]': {'0_to_4'},
            '[5-9]': {'5_to_9'},
        }

        dataset.refresh_config_labels()

        # Make sure label refresh is working
        for ai, conf in enumerate(dataset.configurations):
            self.assertEqual(conf.info[ATOMS_NAME_FIELD], f"{ai}")
            if ai < 5:
                self.assertSetEqual(conf.info[ATOMS_LABELS_FIELD], {'0_to_4'})
            elif ai >= 5:
                self.assertSetEqual(conf.info[ATOMS_LABELS_FIELD], {'5_to_9'})


        # Make sure that a new CO gets its labels updated
        new_atoms = Atoms()
        new_atoms.info[ATOMS_NAME_FIELD] = '4'
        dataset.configurations += [Configuration.from_ase(new_atoms)]

        self.assertSetEqual(
            dataset.configurations[-1].info[ATOMS_LABELS_FIELD], set()
        )

        dataset.refresh_config_labels()

        self.assertSetEqual(
            dataset.configurations[-1].info[ATOMS_LABELS_FIELD], {'0_to_4'}
        )


    def test_cs_refresher(self):

        dataset = Dataset(name='test')
        atoms = []
        for ii in range(10):
            atoms.append(Atoms())
            atoms[-1].info[ATOMS_NAME_FIELD] = ii

        dataset.configurations = [Configuration.from_ase(at)for at in atoms]

        # dataset.cs_regexes = {}

        for cs in dataset.configuration_sets:
            for conf in cs.configurations:
                if int(conf.info[ATOMS_NAME_FIELD][-1]) < 5:
                    self.assertEqual(cs.description, 'Default configuration set')
                else:
                    self.assertEqual(cs.description, 'Default configuration set')

        dataset.configuration_set_regexes = {
            'default': 'default',
            '[0-4]': '0_to_4',
            '[5-9]': '5_to_9',
        }

        for cs in dataset.configuration_sets:
            for conf in cs.configurations:
                if int(conf.info[ATOMS_NAME_FIELD][-1]) < 5:
                    self.assertEqual(cs.description, '0_to_4')
                else:
                    self.assertEqual(cs.description, '5_to_9')

        dataset.resync()
        self.assertEqual(len(dataset.configuration_sets), 2)

        dataset.configuration_set_regexes['[3-7]'] = '3_to_7'

        dataset.resync()
        self.assertEqual(len(dataset.configuration_sets), 3)


    def test_labels_updated_everywhere(self):

        dataset = Dataset(name='test')
        atoms = []
        for ii in range(10):
            atoms.append(Atoms())
            atoms[-1].info[ATOMS_NAME_FIELD] = ii
        dataset.configurations = [Configuration.from_ase(at)for at in atoms]

        dataset.configuration_set_regexes = {
            'default': 'default',
            '[0-4]': '0_to_4',
            '[5-9]': '5_to_9',
        }

        # Ensure labels are initially empty
        for cs in dataset.configuration_sets:
            for conf in cs.configurations:
                self.assertSetEqual(set(), conf.info[ATOMS_LABELS_FIELD])

        # Make sure they're re-written correctly
        dataset.configuration_label_regexes = {
            '[0-4]': {'0_to_4'},
            '[5-9]': {'5_to_9'},
        }

        for cs in dataset.configuration_sets:
            if cs.description == '0_to_4':
                self.assertSetEqual({'0_to_4'}, set(cs.labels))
            elif cs.description == '5_to_9':
                self.assertSetEqual({'5_to_9'}, set(cs.labels))

            for conf in cs.configurations:
                if int(conf.info[ATOMS_NAME_FIELD][-1]) < 5:
                    self.assertSetEqual(
                        {'0_to_4'}, conf.info[ATOMS_LABELS_FIELD]
                    )
                else:
                    self.assertSetEqual(
                        {'5_to_9'}, conf.info[ATOMS_LABELS_FIELD]
                    )

        # Make sure they're added to correctly
        dataset.configuration_label_regexes['[0-9]'] = {'new_label'}
        dataset.resync()

        for cs in dataset.configuration_sets:
            for conf in cs.configurations:
                if int(conf.info[ATOMS_NAME_FIELD][-1]) < 5:
                    self.assertSetEqual(
                        {'0_to_4', 'new_label'}, conf.info[ATOMS_LABELS_FIELD]
                    )
                else:
                    self.assertSetEqual(
                        {'5_to_9', 'new_label'}, conf.info[ATOMS_LABELS_FIELD]
                    )

        # And also removed properly
        dataset.delete_config_label_regex('[0-4]')
        dataset.resync()

        for cs in dataset.configuration_sets:
            for conf in cs.configurations:
                if int(conf.info[ATOMS_NAME_FIELD][-1]) < 5:
                    self.assertSetEqual(
                        {'new_label'}, conf.info[ATOMS_LABELS_FIELD]
                    )
                else:
                    self.assertSetEqual(
                        {'5_to_9', 'new_label'}, conf.info[ATOMS_LABELS_FIELD]
                    )

    def test_default_name(self):
        dataset = Dataset('test')

        dataset.configurations = load_data(
            'colabfit/tests/files/test_file.extxyz',
            file_format='xyz',
            name_field=None,
            elements=['In', 'P'],
            default_name='test',
        )

        self.assertEqual(
            ['test_0', 'test_1', 'test_2'],
            [co.info[ATOMS_NAME_FIELD] for co in dataset.configurations]
        )


    def test_custom_properties(self):

        dataset = Dataset(name='test')
        atoms = []
        for ii in range(1, 11):
            atoms.append(Atoms(f'H{ii}', positions=np.random.random((ii, 3))))
            atoms[-1].info[ATOMS_NAME_FIELD] = ii

            atoms[-1].info['eng'] = ii
            atoms[-1].info['fcs'] = np.ones((ii, 3))*ii

            atoms[-1].info['string'] = f'string_{ii}'
            atoms[-1].info['1d-array'] = np.ones(5)*ii
            atoms[-1].arrays['per-atom-array'] = np.ones((ii, 3))*ii+1

        dataset.configurations = [Configuration.from_ase(at)for at in atoms]

        dataset.property_map = {
            'default': {
                'energy': {'field': 'eng', 'units': 'eV'},
                'forces': {'field': 'fcs', 'units': 'eV/Ang'},
            },
            'my-custom-property': {
                'a-custom-string':
                    {'field': 'string', 'units': None},
                'a-custom-1d-array':
                    {'field': '1d-array', 'units': 'eV'},
                'a-custom-per-atom-array':
                    {'field': 'per-atom-array', 'units': 'eV'},
            }
        }

        dataset.custom_definitions = {
            'my-custom-property': 'colabfit/tests/files/test_property.edn'
        }

        dataset.parse_data()

        self.assertEqual(20, len(dataset.data))
        self.assertEqual(10, len(dataset.get_data('energy')))
        self.assertEqual(10, len(dataset.get_data('forces')))
        self.assertEqual(10, len(dataset.get_data('a-custom-string')))
        self.assertEqual(10, len(dataset.get_data('a-custom-1d-array')))
        self.assertEqual(10, len(dataset.get_data('a-custom-per-atom-array')))



    def test_custom_properties_no_edn(self):

        dataset = Dataset(name='test')
        atoms = []
        for ii in range(1, 11):
            atoms.append(Atoms(f'H{ii}', positions=np.random.random((ii, 3))))
            atoms[-1].info[ATOMS_NAME_FIELD] = ii

            atoms[-1].info['eng'] = ii
            atoms[-1].info['fcs'] = np.ones((ii, 3))*ii

            atoms[-1].info['string'] = f'string_{ii}'
            atoms[-1].info['1d-array'] = np.ones(5)*ii
            atoms[-1].arrays['per-atom-array'] = np.ones((ii, 3))*ii+1

        dataset.configurations = [Configuration.from_ase(at)for at in atoms]

        dataset.property_map = {
            'default': {
                'energy': {'field': 'eng', 'units': 'eV'},
                'forces': {'field': 'fcs', 'units': 'eV/Ang'},
            },
            'my-custom-property': {
                'a-custom-string':
                    {'field': 'string', 'units': None},
                'a-custom-1d-array':
                    {'field': '1d-array', 'units': 'eV'},
                'a-custom-per-atom-array':
                    {'field': 'per-atom-array', 'units': 'eV'},
            }
        }

        dataset.custom_definitions = {
            'my-custom-property': {
                "property-id":
                    "my-custom-property",
                "property-title":
                    "A custom, user-provided Property Definition. See "\
                    "https://openkim.org/doc/schema/properties-framework/ for "\
                    "instructions on how to build these files",
                "property-description":
                    "Some human-readable description",
                "a-custom-field-name": {
                    "type":         "string",
                    "has-unit":     False,
                    "extent":       [],
                    "required":     False,
                    "description":  "The description of the custom field"
                },
                "a-custom-1d-array": {
                    "type":         "float",
                    "has-unit":     True,
                    "extent":       [":"],
                    "required":     True,
                    "description":  "This should be a 1D vector of floats"
                },
                "a-custom-per-atom-array": {
                    "type":         "float",
                    "has-unit":     True,
                    "extent":       [":", 3],
                    "required":     True,
                    "description":
                        "This is a 2D array of floats, where the second "\
                        "dimension has a length of 3"
                },
            }
        }

        dataset.parse_data()

        self.assertEqual(20, len(dataset.data))
        self.assertEqual(10, len(dataset.get_data('energy')))
        self.assertEqual(10, len(dataset.get_data('forces')))
        self.assertEqual(10, len(dataset.get_data('a-custom-string')))
        self.assertEqual(10, len(dataset.get_data('a-custom-1d-array')))
        self.assertEqual(10, len(dataset.get_data('a-custom-per-atom-array')))


    def test_custom_properties_missing_key(self):

        dataset = Dataset(name='test')
        atoms = []
        for ii in range(1, 11):
            atoms.append(Atoms(f'H{ii}', positions=np.random.random((ii, 3))))
            atoms[-1].info[ATOMS_NAME_FIELD] = ii

            atoms[-1].info['eng'] = ii
            atoms[-1].info['fcs'] = np.ones((ii, 3))*ii

            atoms[-1].info['string'] = f'string_{ii}'
            atoms[-1].info['1d-array'] = np.ones(5)*ii

            # Remove one of the important keys
            # atoms[-1].arrays['per-atom-array'] = np.ones((ii, 3))*ii+1

        dataset.configurations = [Configuration.from_ase(at)for at in atoms]

        dataset.property_map = {
            'default': {
                'energy': {'field': 'eng', 'units': 'eV'},
                'forces': {'field': 'fcs', 'units': 'eV/Ang'},
            },
            'my-custom-property': {
                'a-custom-string':
                    {'field': 'string', 'units': None},
                'a-custom-1d-array':
                    {'field': '1d-array', 'units': 'eV'},
                'a-custom-per-atom-array':
                    {'field': 'per-atom-array', 'units': 'eV'},
            }
        }

        dataset.custom_definitions = {
            'my-custom-property': 'colabfit/tests/files/test_property.edn'
        }

        self.assertWarns(MissingPropertyFieldWarning, dataset.parse_data)


class TestSetOperations(unittest.TestCase):

    def test_subset_is_subset(self):
        dataset1 = Dataset('test1')

        images1 = []
        for i in range(5):
            images1.append(Atoms('H2', positions=np.random.random((2, 3))))
            images1[-1].info[ATOMS_NAME_FIELD] = dataset1.name + str(i)
            images1[-1].info[ATOMS_LABELS_FIELD] = dataset1.name + '_label_'+ str(i)

            images1[-1].info['energy'] = float(i)

        dataset1.configurations = [Configuration.from_ase(at) for at in images1]
        dataset1.property_map = {
            'default': {
                'energy': {'field': 'energy', 'units': 'eV'},
            },
        }
        dataset1.parse_data()

        dataset2 = Dataset('test1')
        dataset2.configurations = [Configuration.from_ase(at) for at in images1[:3]]
        dataset2.property_map = {
            'default': {
                'energy': {'field': 'energy', 'units': 'eV'},
            },
        }
        dataset2.parse_data()

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

        dataset1.configurations = [Configuration.from_ase(at) for at in images1]
        dataset1.property_map = {
            'default': {
                'energy': {'field': 'energy', 'units': 'eV'},
            },
        }
        dataset1.parse_data()

        dataset2 = Dataset('test1')
        dataset2.configurations = [Configuration.from_ase(at) for at in images1[:3]]
        dataset2.property_map = {
            'default': {
                'energy': {'field': 'energy', 'units': 'eV'},
            },
        }
        dataset2.parse_data()

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

        dataset1.configurations = [Configuration.from_ase(at) for at in images1]
        dataset1.property_map = {
            'default': {
                'energy': {'field': 'energy', 'units': 'eV'},
            },
        }
        dataset1.parse_data()

        dataset2 = Dataset('test1')

        images2 = []
        for i in range(5):
            images2.append(Atoms('H2', positions=np.random.random((2, 3))))
            images2[-1].info[ATOMS_NAME_FIELD] = dataset2.name + str(i)
            images2[-1].info[ATOMS_LABELS_FIELD] = dataset2.name + '_label_'+ str(i)

            images2[-1].info['energy'] = float(i) + 20

        dataset2.configurations = [Configuration.from_ase(at) for at in images2]
        dataset2.property_map = {
            'default': {
                'energy': {'field': 'energy', 'units': 'eV'},
            },
        }
        dataset2.parse_data()

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

        dataset1.configurations = [Configuration.from_ase(at) for at in images1]
        dataset1.property_map = {
            'default': {
                'energy': {'field': 'energy', 'units': 'eV'},
            },
        }
        dataset1.parse_data()

        dataset2 = Dataset('test1')

        images2 = []
        for i in range(5):
            images2.append(Atoms('H2', positions=np.random.random((2, 3))))
            images2[-1].info[ATOMS_NAME_FIELD] = dataset2.name + str(i)
            images2[-1].info[ATOMS_LABELS_FIELD] = dataset2.name + '_label_'+ str(i)

            images2[-1].info['energy'] = float(i) + 20

        dataset2.configurations = [Configuration.from_ase(at) for at in images2]
        dataset2.property_map = {
            'default': {
                'energy': {'field': 'energy', 'units': 'eV'},
            },
        }
        dataset2.parse_data()

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

        dataset1.configurations = [Configuration.from_ase(at) for at in images1]
        dataset1.property_map = {
            'default': {
                'energy': {'field': 'energy', 'units': 'eV'},
            },
        }
        dataset1.parse_data()

        dataset2 = Dataset('test1')

        images2 = []
        for i in range(5):
            images2.append(Atoms('H2', positions=np.random.random((2, 3))))
            images2[-1].info[ATOMS_NAME_FIELD] = dataset2.name + str(i)
            images2[-1].info[ATOMS_LABELS_FIELD] = dataset2.name + '_label_'+ str(i)

            images2[-1].info['energy'] = float(i) + 20

        dataset2.configurations = [Configuration.from_ase(at) for at in images2]
        dataset2.property_map = {
            'default': {
                'energy': {'field': 'energy', 'units': 'eV'},
            },
        }
        dataset2.parse_data()

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

        dataset1.configurations = [Configuration.from_ase(at) for at in images1]
        dataset1.property_map = {
            'default': {
                'energy': {'field': 'energy', 'units': 'eV'},
            },
        }
        dataset1.parse_data()

        dataset2 = Dataset('test1')

        images2 = []
        for i in range(5):
            images2.append(Atoms('H2', positions=np.random.random((2, 3))))
            images2[-1].info[ATOMS_NAME_FIELD] = dataset2.name + str(i)
            images2[-1].info[ATOMS_LABELS_FIELD] = dataset2.name + '_label_'+ str(i)

            images2[-1].info['energy'] = float(i) + 20

        dataset2.configurations = [Configuration.from_ase(at) for at in images2]
        dataset2.property_map = {
            'default': {
                'energy': {'field': 'energy', 'units': 'eV'},
            },
        }
        dataset2.parse_data()

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
            'default': {
                'energy': {'field': 'energy', 'units': 'eV'},
            },
        }

        dataset.configurations = [Configuration.from_ase(at) for at in images]

        regex = re.compile('test[0-3]')
        filtered = dataset.filter(
            'configurations',
            lambda c: regex.search(c.info[ATOMS_NAME_FIELD])
        )

        self.assertEqual(len(filtered.configurations), 4)

        dataset.parse_data()

        regex = re.compile('test[0-3]')
        filtered = dataset.filter(
            'configurations',
            lambda c: regex.search(c.info[ATOMS_NAME_FIELD])
        )

        self.assertEqual(len(filtered.configurations), 4)
        self.assertEqual(len(filtered.data), 4)

        regex = re.compile('.*')
        filtered = dataset.filter(
            'configurations',
            lambda c: regex.search(c.info[ATOMS_NAME_FIELD])
        )

        self.assertEqual(dataset, filtered)


    def test_filter_parent_on_co_names(self):
        dataset = Dataset('test')

        images = []
        for i in range(5):
            images.append(Atoms('H2', positions=np.random.random((2, 3))))
            images[-1].info[ATOMS_NAME_FIELD] = dataset.name + str(i)
            images[-1].info[ATOMS_LABELS_FIELD] = dataset.name + '_label_'+ str(i)

            images[-1].info['energy'] = float(i)

        dataset.property_map = {
            'default': {
                'energy': {'field': 'energy', 'units': 'eV'},
            },
        }


        dataset.configurations = [Configuration.from_ase(at) for at in images]

        dataset.parse_data()

        regex = re.compile('test[0-2]')
        filtered1 = dataset.filter(
            'configurations',
            lambda c: regex.search(c.info[ATOMS_NAME_FIELD])
        )

        regex = re.compile('test[3-4]')
        filtered2 = dataset.filter(
            'configurations',
            lambda c: regex.search(c.info[ATOMS_NAME_FIELD])
        )

        parent = Dataset('parent')
        parent.attach_dataset(filtered1)
        parent.attach_dataset(filtered2)
        parent.resync()

        regex = re.compile('test[2-3]')
        parent2 = parent.filter(
            'configurations',
            lambda c: regex.search(c.info[ATOMS_NAME_FIELD])
        )

        self.assertEqual(parent2.data[0].configurations[0].info[ATOMS_NAME_FIELD], 'test2')
        self.assertEqual(parent2.data[1].configurations[0].info[ATOMS_NAME_FIELD], 'test3')


    def test_filter_on_data(self):
        dataset = Dataset('test')

        images = []
        for i in range(5):
            images.append(Atoms('H2', positions=np.random.random((2, 3))))
            images[-1].info[ATOMS_NAME_FIELD] = dataset.name + str(i)
            images[-1].info[ATOMS_LABELS_FIELD] = dataset.name + '_label_'+ str(i)

            images[-1].info['energy'] = float(i)

        dataset.property_map = {
            'default': {
                'energy': {'field': 'energy', 'units': 'eV'},
            },
        }


        dataset.configurations = [Configuration.from_ase(at) for at in images]

        dataset.parse_data()

        filtered = dataset.filter(
            'data',
            lambda d: d.edn['unrelaxed-potential-energy']['source-value'] < 3.0
        )

        self.assertEqual(len(filtered.configurations), 3)


    def test_filter_parent_on_data(self):
        dataset = Dataset('test')

        images = []
        for i in range(5):
            images.append(Atoms('H2', positions=np.random.random((2, 3))))
            images[-1].info[ATOMS_NAME_FIELD] = dataset.name + str(i)
            images[-1].info[ATOMS_LABELS_FIELD] = dataset.name + '_label_'+ str(i)

            images[-1].info['energy'] = float(i)

        dataset.property_map = {
            'default': {
                'energy': {'field': 'energy', 'units': 'eV'},
            },
        }


        dataset.configurations = [Configuration.from_ase(at) for at in images]

        dataset.parse_data()

        filtered1 = dataset.filter(
            'data',
            lambda d:
                0 <= d.edn['unrelaxed-potential-energy']['source-value'] <= 2
        )

        filtered2 = dataset.filter(
            'data',
            lambda d:
                3 <= d.edn['unrelaxed-potential-energy']['source-value'] <= 4
        )


        parent = Dataset('parent')
        parent.attach_dataset(filtered1)
        parent.attach_dataset(filtered2)
        parent.resync()

        parent2 = parent.filter(
            'data',
            lambda d:
                2 <= d.edn['unrelaxed-potential-energy']['source-value'] <= 3
        )


        self.assertEqual(parent2.data[0].configurations[0].info[ATOMS_NAME_FIELD], 'test2')
        self.assertEqual(parent2.data[1].configurations[0].info[ATOMS_NAME_FIELD], 'test3')


    def test_multi_layer_parent(self):
        dataset = Dataset('test')

        images = []
        for i in range(30):
            images.append(Atoms('H2', positions=np.random.random((2, 3))))
            images[-1].info[ATOMS_NAME_FIELD] = dataset.name + str(i)
            images[-1].info[ATOMS_LABELS_FIELD] = dataset.name + '_label_'+ str(i)

            images[-1].info['energy'] = float(i)

        dataset.property_map = {
            'default': {
                'energy': {'field': 'energy', 'units': 'eV'},
            },
        }


        dataset.configurations = [Configuration.from_ase(at) for at in images]

        dataset.parse_data()

        child1 = dataset.filter(
            'data',
            lambda d:
                0 <= d.edn['unrelaxed-potential-energy']['source-value'] < 10
        )

        child2 = dataset.filter(
            'data',
            lambda d:
                0 <= d.edn['unrelaxed-potential-energy']['source-value'] < 10
        )

        child3 = dataset.filter(
            'data',
            lambda d:
                0 <= d.edn['unrelaxed-potential-energy']['source-value'] < 10
        )

        subparent = Dataset('subparent')
        subparent.attach_dataset(child1)
        subparent.attach_dataset(child2)
        subparent.resync()

        parent = Dataset('parent')
        parent.attach_dataset(subparent)
        parent.attach_dataset(child3)
        parent.resync()

        self.assertEqual(len(parent.configurations), 30)

        subset = parent.filter(
            'data',
            lambda d:
                d.edn['unrelaxed-potential-energy']['source-value'] % 5 == 0
        )

        self.assertEquals(len(subset.data[0].configurations), 4)
        self.assertEquals(len(subset.data[1].configurations), 2)


class Test_ParentDatasets(unittest.TestCase):
    def setUp(self):
        dataset = Dataset('test')

        dataset.configurations = load_data(
            'colabfit/tests/files/test_file.extxyz',
            file_format='xyz',
            name_field='name',
            elements=['In', 'P'],
            default_name='test',
        )

        dataset.property_map = {
            'default': {
                'energy': {'field': 'energy', 'units': 'eV'},
                'forces': {'field': 'forces', 'units': 'eV/Ang'},
            }
        }

        dataset.parse_data()

        dataset.configuration_set_regexes = {
            'd0lda[7|5]': 'first two',
            'd0lda1': 'last one',
        }

        dataset.resync()
        self.dataset = dataset


    def test_basic(self):
        child0 = self.dataset.dataset_from_config_sets(0)
        child1 = self.dataset.dataset_from_config_sets(1)

        parent1 = Dataset('parent1')
        parent1.attach_dataset(child0)
        parent1.attach_dataset(child1)
        parent1.resync()

        parent2 = parent1.dataset_from_config_sets([(0, 0), (1, 0)])

        parent2.resync()

        self.assertEqual(parent1, parent2)

        eng1 = parent1.get_data('energy', ravel=True)
        eng2 = parent2.get_data('energy', ravel=True)

        np.testing.assert_allclose(eng1, eng2)


class Test_DatasetFunctions(unittest.TestCase):
    def setUp(self):
        dataset1 = Dataset('dataset1')

        images1 = []
        for i in range(5):
            images1.append(Atoms('H2', positions=np.random.random((2, 3))))
            images1[-1].info[ATOMS_NAME_FIELD] = dataset1.name + str(i)
            images1[-1].info[ATOMS_LABELS_FIELD] = dataset1.name + '_label_'+ str(i)

            images1[-1].info['energy'] = float(i)
            images1[-1].arrays['forces'] = np.ones_like(images1[-1].positions)*float(i)

        dataset1.configurations = [Configuration.from_ase(at) for at in images1]
        dataset1.property_map = {
            'default': {
                'energy': {'field': 'energy', 'units': 'eV'},
                'forces': {'field': 'forces', 'units': 'eV/Ang'},
            }
        }

        dataset1.parse_data()
        dataset1.resync()

        dataset2 = Dataset('dataset2')

        images2 = []
        for i in range(5):
            images2.append(Atoms('H2', positions=np.random.random((2, 3))))
            images2[-1].info[ATOMS_NAME_FIELD] = dataset2.name + str(i)
            images2[-1].info[ATOMS_LABELS_FIELD] = dataset2.name + '_label_'+ str(i)

            images2[-1].info['energy'] = float(i)
            images2[-1].arrays['forces'] = np.ones_like(images2[-1].positions)*float(i)

        dataset2.configurations = [Configuration.from_ase(at) for at in images2]
        dataset2.property_map = {
            'default': {
                'energy': {'field': 'energy', 'units': 'eV'},
                'forces': {'field': 'forces', 'units': 'eV/Ang'},
            }
        }

        dataset2.parse_data()
        dataset2.resync()

        parent = Dataset('parent')
        parent.attach_dataset(dataset1)
        parent.attach_dataset(dataset2)
        parent.resync()

        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.parent   = parent


    def test_rename_property(self):
        eng_org = self.dataset1.get_data('energy', ravel=True)

        self.dataset1.rename_property('energy', 'a_different_name')

        eng_new = self.dataset1.get_data('a_different_name', ravel=True)

        np.testing.assert_allclose(eng_org, eng_new)


    def test_basic_transform(self):
        self.dataset1.apply_transformation('energy', lambda x, c: 1.0)
        eng_org = self.dataset1.get_data('energy', ravel=True)