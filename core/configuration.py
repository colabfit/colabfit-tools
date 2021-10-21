import numpy as np
from ase import Atoms
from bson import ObjectId

from core import ATOMS_ID_FIELD, ATOMS_NAME_FIELD, ATOMS_LABELS_FIELD


class Configuration(Atoms):
    """
    A Configuration is used to store an `ase.Atoms` object and to propagate
    certain changes to any observers who are watching the Configuration.

    A Configuration will be observed by zero to many ConfigurationSet objects
    AND zero to many Property objects.
    """

    def __init__(self, atoms, labels=None):
        atoms.info[ATOMS_ID_FIELD] = ObjectId()
        atoms.info[ATOMS_NAME_FIELD] = str(atoms.info[ATOMS_NAME_FIELD])

        if labels is None:
            labels = set()

        atoms.info[ATOMS_LABELS_FIELD] = set(labels)

        self.atoms = atoms

    def colabfit_format(self):
        """
        Formats the attached Atoms object to be in proper ColabFit format:
            - shift to origin
            - rotate to LAMMPS-compliant orientation
            - sort atoms by X, Y, then Z positions
        """
        raise NotImplementedError()

    def __eq__(self, other):
        """
        Uses `ase.Atoms.__eq__` for comparison checking. This compares
        positions, atomic numbers, lattice vectors, and PBCs.
        """

        return self.atoms == other.atoms


    def __hash__(self):
        return hash((
            len(self.atoms),
            hash(self.atoms.arrays['positions'].data.tobytes()),
            hash(self.atoms.arrays['numbers'].data.tobytes()),
            hash(np.array(self.atoms.cell).data.tobytes()),
            hash(np.array(self.atoms.pbc).data.tobytes()),
        ))


    def __ne__(self, other):
        return not self.__eq__(other)


    def __str__(self):
        return "Configuration(name='{}', atoms={})".format(
            self.atoms.info[ATOMS_NAME_FIELD],
            self.atoms
        )


    def __repr__(self):
        return str(self)