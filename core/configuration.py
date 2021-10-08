from ase import Atoms
from bson import ObjectId

from core import ATOMS_ID_FIELD, ATOMS_NAME_FIELD, ATOMS_LABELS_FIELD
from core.observable import Observable


class Configuration(Atoms, Observable):
    """
    A Configuration is used to store an `ase.Atoms` object and to propagate
    certain changes to any observers who are watching the Configuration.

    A Configuraion will be observed by zero to many ConfigurationSet objects
    AND zero to many Property objects.
    """

    _observers = []

    def __init__(self, atoms, name, labels=None):
        atoms.info[ATOMS_ID_FIELD] = ObjectId()
        atoms.info[ATOMS_NAME_FIELD] = str(name)

        if labels is None:
            labels = set()

        atoms.info[ATOMS_LABELS_FIELD] = set(labels)

        self.atoms = atoms


    def attach(self, observer):
        self._observers.append(observer)


    def detach(self, observer):
        self._observers.remove(observer)


    def notify(self):
        for observer in self._observers:
            observer.update(self)


    def update_atoms(self, atoms):
        self.atoms = atoms
        self.notify()


    def update_id(self, new_id):
        self.atoms.info[ATOMS_ID_FIELD] = new_id
        self.notify()


    def update_labels(self, new_labels):
        self.atoms.info[ATOMS_LABELS_FIELD] = set(new_labels)
        self.notify()


    def colabfit_format(self):
        """
        Formats the attached Atoms object to be in proper ColabFit format:
            - shift to origin
            - rotate to LAMMPS-compliant orientation
            - sort atoms by X, Y, then Z positions
        """
        raise NotImplementedError()

    def __str__(self):
        return "Configuration(name='{}', atoms={})".format(
            self.atoms.info[ATOMS_NAME_FIELD],
            self.atoms
        )


    def __repr__(self):
        return str(self)