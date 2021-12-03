from ase import Atoms

from colabfit.tools.database import Database


class TestDatabaseBasic:

    def test_add_configurations(self):
        database = Database()

        database.add_configurations(
            (Atoms(f'H{i}') for i in range(1, 100))
        )