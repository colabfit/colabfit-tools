import numpy as np
from ase import Atoms
from ase.io import read

from core import ATOMS_NAME_FIELD, ATOMS_LABELS_FIELD


class BaseConverter:
    """
    A Converter is used to load a list of ase.Atoms objects, and clean them to
    ensure that metadata is stored in expected fields.

    Expected fields:
        atoms.info[ATOMS_NAME_FIELD]:
            A string describing the atoms object

        atoms.info[ATOMS_LABELS_FIELD]:
            A set of strings. Empty set if no labels
    """

    def load(self, file_path, name_field, elements, default_name=''):
        """Dummy wrapper to enforce input fields"""
        return self._load(file_path, name_field, elements, default_name)


class EXYZConverter(BaseConverter):
    def _load(self, file_path, name_field, elements, default_name):
        elements = set(elements)

        images = read(file_path, slice(0, None), format='extxyz')

        for ai, atoms in enumerate(images):
            a_elems = set(atoms.get_chemical_symbols())
            if not a_elems.issubset(elements):
                raise RuntimeError(
                    f"Image {ai} elements {a_elems} is not a subset of "\
                        "{elements}."
                )

            if name_field is None:
                atoms.info[ATOMS_NAME_FIELD] = f"{default_name}_{ai}"
            else:
                if name_field in atoms.info:
                    name = atoms.info[name_field]
                    del atoms.info[name_field]
                    atoms.info[ATOMS_NAME_FIELD] = name
                else:
                    raise RuntimeError(
                        f"Field {name_field} not in atoms.info for index "\
                            f"{len(images)}. Set `name_field=None` "\
                                "to use `default_name`."
                    )

            atoms.info[ATOMS_LABELS_FIELD] = set()

        return images
    
class CFGConverter(BaseConverter):
    def _load(self, file_path, name_field, elements, default_name):

        images = []
        with open(file_path) as cfg_file:
            for line in cfg_file:
                line = line.strip()

                # Skip blank lines
                if len(line) == 0:
                    continue

                # Found beginning of configuration
                if line == 'BEGIN_CFG':
                    natoms = 0
                    cell = None
                    symbols = None
                    positions = None
                    forces = None
                    energy = None
                    virial = None
                    features = {}

                elif line == 'Size':
                    natoms = int(cfg_file.readline().strip())

                elif (line == 'SuperCell') or (line == 'Supercell'):
                    v1 = np.array([float(v.strip()) for v in cfg_file.readline().split()])
                    v2 = np.array([float(v.strip()) for v in cfg_file.readline().split()])
                    v3 = np.array([float(v.strip()) for v in cfg_file.readline().split()])

                    cell = np.array([v1, v2, v3])

                elif 'AtomData' in line:
                    symbols   = []
                    positions = np.zeros((natoms, 3))

                    fields = [l.strip() for l in line.split()[1:]]

                    if 'fx' in fields:
                        forces    = np.zeros((natoms, 3))

                    for ni in range(natoms):
                        newline = [
                            l.strip() for l in cfg_file.readline().split()
                        ]

                        for f, v in zip(fields, newline):
                            if f == 'type':
                                symbols.append(elements[int(v)])
                            elif f == 'cartes_x':
                                positions[ni, 0] = float(v)
                            elif f == 'cartes_y':
                                positions[ni, 1] = float(v)
                            elif f == 'cartes_z':
                                positions[ni, 2] = float(v)
                            elif f == 'fx':
                                forces[ni, 0] = float(v)
                            elif f == 'fy':
                                forces[ni, 1] = float(v)
                            elif f == 'fz':
                                forces[ni, 2] = float(v)

                elif line == 'Energy':
                    energy = float(cfg_file.readline().strip())
                elif 'Stress' in line:
                    check = [l.strip() for l in line.split()]

                    if tuple(check[1:]) != ('xx', 'yy', 'zz', 'yz', 'xz', 'xy'):
                        raise RuntimeError(
                            "CFG file format error. Check 'PlusStress' lines"
                        )

                    tmp = np.array([
                        float(v.strip()) for v in cfg_file.readline().split()
                    ])

                    virial = np.zeros((3, 3))
                    virial[0, 0] = tmp[0]
                    virial[1, 1] = tmp[1]
                    virial[2, 2] = tmp[2]
                    virial[1, 2] = tmp[3]
                    virial[2, 1] = tmp[3]
                    virial[0, 2] = tmp[4]
                    virial[2, 0] = tmp[4]
                    virial[0, 1] = tmp[5]
                    virial[1, 0] = tmp[5]

                elif 'Feature' in line:
                    split = [l.strip() for l in line.split()]

                    feat_name   = split[1]
                    feat_val    = split[2]

                    # Uses list 
                    if feat_name not in features:
                        features[feat_name] = [feat_val]
                    else:
                        features[feat_name].append(feat_val)

                elif line == 'END_CFG':
                    molecule = (cell is None)
                    pbc = None if molecule else True

                    atoms = Atoms(
                        symbols=symbols,
                        positions=positions,
                        pbc=pbc,
                        cell=cell,
                    )

                    if energy is not None:
                        atoms.info['energy'] = energy

                    if forces is not None:
                        atoms.arrays['forces'] = forces

                    if virial is not None:
                        atoms.info['virial'] = virial

                    # Add additional textual information
                    for feat, lst in features.items():
                        atoms.info[feat] = ' '.join(lst)

                    # Parse name, if it exists
                    if name_field is None:
                        atoms.info[ATOMS_NAME_FIELD] = f"{default_name}_{len(images)}"
                    else:
                        if name_field in atoms.info:
                            name = atoms.info[name_field]
                            del atoms.info[name_field]
                            atoms.info[ATOMS_NAME_FIELD] = name
                        else:
                            raise RuntimeError(
                                f"Field {name_field} not in atoms.info for index "\
                                    f"{len(images)}. Set `name_field=None` "\
                                        "to use `default_name`."
                            )

                    atoms.info[ATOMS_LABELS_FIELD] = set()

                    images.append(atoms)