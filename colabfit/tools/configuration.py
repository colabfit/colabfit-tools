import numpy as np
from ase import Atoms
from bson import ObjectId

from colabfit import (
    ATOMS_ID_FIELD, ATOMS_NAME_FIELD, ATOMS_LABELS_FIELD,
    ATOMS_CONSTRAINTS_FIELD
)


class Configuration(Atoms):
    """
    A Configuration is used to store an `ase.Atoms` object and to propagate
    certain changes to any observers who are watching the Configuration.

    A Configuration will be observed by zero to many ConfigurationSet objects
    AND zero to many Property objects.
    """

    def __init__(self, labels=None, constraints=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.info[ATOMS_ID_FIELD] = ObjectId()

        if ATOMS_NAME_FIELD in self.info:
            self.info[ATOMS_NAME_FIELD] = str(self.info[ATOMS_NAME_FIELD])
        else:
            self.info[ATOMS_NAME_FIELD] = ""

        if ATOMS_LABELS_FIELD not in self.info:
            if labels is None:
                labels = set()
        else:
            labels = set(self.info[ATOMS_LABELS_FIELD])

        self.info[ATOMS_LABELS_FIELD] = set(labels)

        if ATOMS_CONSTRAINTS_FIELD not in self.info:
            if constraints is None:
                constraints = set()
        else:
            constraints = set(self.info[ATOMS_CONSTRAINTS_FIELD])

        self.info[ATOMS_CONSTRAINTS_FIELD] = set(constraints)


    @classmethod
    def from_ase(cls, atoms):
        # Workaround for bug in todict() fromdict() with constraints.
        # Merge request: https://gitlab.com/ase/ase/-/merge_requests/2574
        if atoms.constraints is not None:
            atoms.constraints = [c.todict() for c in atoms.constraints]

        return cls.fromdict(atoms.todict())


    def colabfit_format(self):
        """
        Formats the attached Atoms object to be in proper ColabFit format:
            - shift to origin
            - rotate to LAMMPS-compliant orientation
            - sort atoms by X, Y, then Z positions
        """
        raise NotImplementedError()


    def __hash__(self):
        constraints_hash = hash(tuple(
            hash(tuple(self.info[c])) if c in self.info
            else hash(np.array(self.arrays[c]).data.tobytes())
            for c in self.info[ATOMS_CONSTRAINTS_FIELD]
        ))

        return hash((
            len(self),
            constraints_hash,
            hash(self.arrays['positions'].data.tobytes()),
            hash(self.arrays['numbers'].data.tobytes()),
            hash(np.array(self.cell).data.tobytes()),
            hash(np.array(self.pbc).data.tobytes()),
        ))


    # def __str__(self):
    #     ase_str = super().__str__()
    #     return "Configuration(name='{}', atoms={})".format(
    #         self.info[ATOMS_NAME_FIELD],
    #         ase_str
    #     )


    # def __repr__(self):
    #     return str(self)