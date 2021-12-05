import datetime
import numpy as np
from ase import Atoms
from string import ascii_lowercase, ascii_uppercase

from colabfit import (
    ATOMS_NAME_FIELD, ATOMS_LABELS_FIELD,
    ATOMS_CONSTRAINTS_FIELD
)


class Configuration(Atoms):
    """
    A Configuration is an extension of an :class:`ase.Atoms` object that is
    guaranteed to have the following fields in its :attr:`info` dictionary:

    - :attr:`~colabfit.ATOMS_NAME_FIELD` = :code:"_name"
    - :attr:`~colabfit.ATOMS_LABELS_FIELD` = :code:"_labels"
    - :attr:`~colabfit.ATOMS_CONSTRAINTS_FIELD` = :code:"_constraints"
    """

    def __init__(
        self, description='', labels=None, constraints=None, *args, **kwargs
        ):
        """
        Constructs a Configuration. Calls :meth:`ase.Atoms.__init__()`, then
        populates the additional required fields.

        Args:

            description (str):
                A human-readable description. Can also be a list of strings.

            
        """
        super().__init__(*args, **kwargs)

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


        # Additional fields for querying

        atomic_symbols = self.get_chemical_symbols()
        processed_fields = process_species_list(atomic_symbols)

        self.info['_description'] = description
        self.info['_last_modified'] = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
        self.info['_elements'] = processed_fields['elements']
        self.info['_nelements'] = processed_fields['nelements']
        self.info['_elements_ratios'] = processed_fields['elements_ratios']
        self.info['_chemical_formula_reduced'] = processed_fields['chemical_formula_reduced']
        self.info['_chemical_formula_anonymous'] = processed_fields['chemical_formula_anonymous']
        self.info['_chemical_formula_hill'] = self.get_chemical_formula()
        self.info['_natoms'] = processed_fields['natoms']
        # self.info['_species'] = processed_fields['species']
        self.info['_dimension_types'] = self.get_pbc().astype(int)
        self.info['_nperiodic_dimensions'] = sum(self.get_pbc())
        self.info['_lattice_vectors'] = np.array(self.get_cell())

    @classmethod
    def from_ase(cls, atoms):
        """
        Generates a :class:`Configuration` from an :code:`ase.Atoms` object.
        """
        # Workaround for bug in todict() fromdict() with constraints.
        # Merge request: https://gitlab.com/ase/ase/-/merge_requests/2574
        if atoms.constraints is not None:
            atoms.constraints = [c.todict() for c in atoms.constraints]

        return cls.fromdict(atoms.todict())


    # def colabfit_format(self):
    #     """
    #     Formats the attached Atoms object to be in proper ColabFit format:
    #         - shift to origin
    #         - rotate to LAMMPS-compliant orientation
    #         - sort atoms by X, Y, then Z positions
    #     """
    #     raise NotImplementedError()


    def __hash__(self):
        """
        Generates a hash for :code:`self` by hashing its length
        positions, (atomic) numbers, simulation cell, and periodic
        boundary conditions.

        Note that the positions and cell vectors are rounded to 8 decimal places
        before being compared
        """

        # constraints_hash = hash(tuple(
        #     hash(tuple(self.info[c])) if c in self.info
        #     else hash(np.array(self.arrays[c]).data.tobytes())
        #     for c in self.info[ATOMS_CONSTRAINTS_FIELD]
        # ))

        return hash((
            len(self),
            # constraints_hash,
            hash(np.round_(self.arrays['positions'], decimals=8).data.tobytes()),
            hash(self.arrays['numbers'].data.tobytes()),
            hash(np.round_(np.array(self.cell), decimals=8).data.tobytes()),
            hash(np.array(self.pbc).data.tobytes()),
        ))


    def __eq__(self, other):
        return hash(self) == hash(other)

    def __str__(self):
        ase_str = super().__str__()
        return "Configuration(name='{}', {})".format(
            self.info[ATOMS_NAME_FIELD],
            ase_str[14:-1]
        )

    # # Don't do this. It will cause a recursion error
    # def __repr__(self):
    #     return str(self)


def process_species_list(atomic_species):
    """Extracts useful metadata from a list of atomic species"""
    natoms = len(atomic_species)
    elements = sorted(list(set(atomic_species)))
    nelements = len(elements)
    elements_ratios = [
        atomic_species.count(el)/natoms for el in elements
    ]

    species_counts = [atomic_species.count(sp) for sp in elements]

    # Build per-element proportions
    from math import gcd
    from functools import reduce
    def find_gcd(list):
        x = reduce(gcd, list)
        return x

    count_gcd = find_gcd(species_counts)
    species_proportions = [sc//count_gcd for sc in species_counts]

    chemical_formula_reduced = ''
    for elem, elem_prop in zip(elements, species_proportions):
        chemical_formula_reduced += '{}{}'.format(
            elem, '' if elem_prop == 1 else str(elem_prop)
        )

    # Replace elements with A, B, C, ...
    species_proportions = sorted(species_proportions, reverse=True)

    chemical_formula_anonymous = ''
    for spec_idx, spec_count in enumerate(species_proportions):
        # OPTIMADE uses A...Z, then Aa..Za, ..., up to Az...Zz

        count1 = spec_idx // 26
        count2 = spec_idx %  26

        if count1 == 0:
            anon_spec = ascii_uppercase[count2]
        else:
            anon_spec = ascii_uppercase[count1] + ascii_lowercase[count2]

        chemical_formula_anonymous += anon_spec
        if spec_count > 1:
            chemical_formula_anonymous += str(spec_count)

    species = []
    for el in elements:
        # https://github.com/Materials-Consortia/OPTIMADE/blob/develop/optimade.rst#7214species
        species.append({
            'name': el,
            'chemical_symbols': [el],
            'concentration': [1.0],
        })

    return {
        'natoms': natoms,
        'elements': elements,
        'nelements': nelements,
        'elements_ratios': elements_ratios,
        'chemical_formula_anonymous': chemical_formula_anonymous,
        'chemical_formula_reduced': chemical_formula_reduced,
        'species': species,
    }

