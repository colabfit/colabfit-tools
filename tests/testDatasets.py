import markdown
import unittest
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
        dataset.co_label_regexes.pop('[0-4]')
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

class TestDatasetManipulation(unittest.TestCase):
    pass