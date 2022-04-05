import numpy as np
from hashlib import sha512
from ase import Atoms
from string import ascii_lowercase, ascii_uppercase
from Bio.SeqRecord import SeqRecord
from tqdm import tqdm
from colabfit import (
    HASH_LENGTH, HASH_SHIFT,
    ATOMS_NAME_FIELD, ATOMS_LABELS_FIELD,
    ATOMS_CONSTRAINTS_FIELD
)


class BaseConfiguration:
    """The base Configuration class.

    All other Configuration 'types' must subclass this and any other useful classes.

    Configuration classes must pass all necessary unique identifiers as keyword arguments
    to BaseConfiguration.__init. Unique identifiers are values that are needed to uniquely
    identify one Configuration instance from another. These values will be used to produce
    a unique hash for each Configuration instance and be added to the database with
    their associated keyword.

    All Configuration classes must define a self.configuration_summary method. This is used
    to extract any other useful information that will be included (in addition to all unique
    identifiers) in the Configuration's entry in the Database.

    See AtomicConfiguration as an example.

    Attributes:

        info (dict):
            Stores important metadata for a Configuration. At a minimum, it will include
            keywords "_name" and "_labels".
        unique_identifiers (dict):
            Stores all key-value pairs needed to uniquely define a Configuration.
    """

    _unique_identifier_kw = None

    def __init__(self, names=None, labels=None):
        """
        Args:
            names (str, list of str):
                Names to be associated with a Configuration
            labels (str, list of str):
                Labels to be associated with a Configuration
            **unique_identifiers:
                All identifiers needed to uniquely define a Configuration
        """

        self.info = {}
        if names is None:
            self.info[ATOMS_NAME_FIELD] = set()
        else:
            self.info[ATOMS_NAME_FIELD] = set(list(names))
        if labels is None:
            self.info[ATOMS_LABELS_FIELD] = set()
        else:
            self.info[ATOMS_LABELS_FIELD] = set(list(labels))


    @property
    def unique_identifiers(self):
        raise NotImplementedError('All Configuration classes should implement this.')

    @unique_identifiers.setter
    def unique_identifiers(self):
        raise NotImplementedError('All Configuration classes should implement this.')

    def configuration_summary(self):
        """Extracts useful information from a Configuration.

        All Configuration classes should implement this.
        Any useful information that should be included under a Configuration's entry in the Database
        (in addition to its unique identifiers) should be extracted and added to a dict.
        Will be called before inserting data into Database

        Returns:
            dict: Keys and their associated values that will be included under a Configuration's entry in the Database
        """
        raise NotImplementedError('All Configuration classes should implement this.')

    @staticmethod
    def aggregate_configuration_summaries(ids):
        """Aggregates information for given configurations
        All Configuration classes should implement this.
        Similar to configuration_summary, but summarizes information
        for a collection of Configurations

        Args:
            ids:
                IDs of Configurations of interest

        Returns:
            dict: Key-value pairs of information aggregated from multiple Configurations
        """
        raise NotImplementedError('All Configuration classes should implement this.')

    def __hash__(self):
        """Generates a hash for :code:`self`.

        Hashes all values in self.unique_identifiers.
        hashlib is used instead of hash() to avoid hash randomisation.

        Returns:
            int: Value of hash
        """
        if len(self.unique_identifiers) == 0:
            raise Exception('Ensure unique identifiers are properly defined!')
        _hash = sha512()
        for _, v in self.unique_identifiers.items():
            _hash.update(bytes(pre_hash_formatting(v)))
        return int(str(int(_hash.hexdigest(), 16) - HASH_SHIFT)[:HASH_LENGTH])

    def __eq__(self, other):
        """
        Two Configurations are considered to be identical if they have the same
        hash value.
        """
        return hash(self) == hash(other)



class AtomicConfiguration(BaseConfiguration, Atoms):
    # TODO: Modify docstring
    """
    An AtomicConfiguration is an extension of a :class:`BaseConfiguration` and an :class:`ase.Atoms`
    object that is guaranteed to have the following fields in its :attr:`info` dictionary:

    - :attr:`~colabfit.ATOMS_ID_FIELD` = :code:"_id" # TODO: Don't think this is true->Fix
    - :attr:`~colabfit.ATOMS_NAME_FIELD` = :code:"_name"
    - :attr:`~colabfit.ATOMS_LABELS_FIELD` = :code:"_labels"
    # TODO: Reimplement contrainsts
    # - :attr:`~colabfit.ATOMS_CONSTRAINTS_FIELD` = :code:"_constraints"
    """

    unique_identifier_kw = ['atomic_numbers', 'positions', 'cell', 'pbc']

    def __init__(self, names=None, labels=None, **kwargs):
        """
        Constructs an AtomicConfiguration. Calls :meth:`BaseConfiguration.__init__()`
        and :meth:`ase.Atoms.__init__()`

        Args:
            numbers (list/ndarray of int):
                Atomic Numbers
            positions (Anything that can be converted to a ndarray of shape (n, 3)):
                Atomic xyz-positions
            cell (3x3 matrix of float): # TODO: Ensure similar formatting that is used in ASE
                Unit cell vectors
            pbc (list of 3 bool):
                Periodic boundary conditions for each dimension
            names (str, list of str):
                Names to be associated with a Configuration
            labels (str, list of str):
                Labels to be associated with a Configuration
            **kwargs:
                Other keyword arguments that can be passed to :meth:`ase.Atoms.__init__()`
        """

        BaseConfiguration.__init__(
            self,
            names=names,
            labels=labels,
        )

        kwargs['info'] = self.info
        if 'atomic_numbers' in list(kwargs.keys()):
            kwargs['numbers'] = kwargs['atomic_numbers']
            kwargs.pop('atomic_numbers')

        Atoms.__init__(self,**kwargs)

        '''
        if ATOMS_NAME_FIELD in self.info:
            v = self.info[ATOMS_NAME_FIELD]
            if not isinstance(v, list):
                v = set([str(v)])
            else:
                v = set(v)

            self.info[ATOMS_NAME_FIELD] = v
        else:
            self.info[ATOMS_NAME_FIELD] = set()
        # TODO fix how labels are utilized
        if ATOMS_LABELS_FIELD not in self.info:
            if labels is None:
                labels = set()
        else:
            labels = set(self.info[ATOMS_LABELS_FIELD])

        self.info[ATOMS_LABELS_FIELD] = set(labels)

# TODO: Reimplement later

        if ATOMS_CONSTRAINTS_FIELD not in self.info:
            if constraints is None:
                constraints = set()
        else:
            constraints = set(self.info[ATOMS_CONSTRAINTS_FIELD])

        self.info[ATOMS_CONSTRAINTS_FIELD] = set(constraints)
        '''
        # Check for name conflicts in info/arrays; would cause bug in parsing
        if set(self.info.keys()).intersection(set(self.arrays.keys())):
            raise RuntimeError(
                "The same key should not be used in both Configuration.info " \
                "and Configuration.arrays"
            )

    @property
    def unique_identifiers(self):
        return {
            'atomic_numbers': self.get_atomic_numbers(),
            'positions': self.get_positions(),
            'cell': np.array(self.get_cell()),
            'pbc': self.get_pbc().astype(int)
        }

    @unique_identifiers.setter
    def unique_identifiers(self, d):
        if set(self.unique_identifier_kw) != set(list(d.keys())):
            raise RunTimeError("There is a mismatch between keywords!")
        # Make sure ASE values are in sync
        self.arrays['numbers'] = d['atomic_numbers']
        self.arrays['positions'] = d['positions']
        self.cell = d['cell']
        self.pbc = d['pbc']

    def configuration_summary(self):
        """Extracts useful metadata from a Configuration

        Gathers the following information from a Configuration:

        * :code:`nsites`: the total number of atoms
        * :code:`nelements`: the total number of unique element types
        * :code:`elements`: the element types
        * :code:`elements_ratios`: elemental ratio of the species
        * :code:`chemical_formula_reduced`: the reduced chemical formula
        * :code:`chemical_formula_anonymous`: the chemical formula
        * :code:`chemical_formula_hill`: the hill chemical formulae
        * :code:`nperiodic_dimensions`: the numbers of periodic dimensions
        * :code:`dimension_types`: the periodic boundary condition

        Returns:
            dict: Keys and their associated values that will be included under a Configuration's entry in the Database
        """

        atomic_species = self.get_chemical_symbols()

        natoms = len(atomic_species)
        elements = sorted(list(set(atomic_species)))
        nelements = len(elements)
        elements_ratios = [
            atomic_species.count(el) / natoms for el in elements
        ]

        species_counts = [atomic_species.count(sp) for sp in elements]

        # Build per-element proportions
        from math import gcd
        from functools import reduce

        def find_gcd(_list):
            x = reduce(gcd, _list)
            return x

        count_gcd = find_gcd(species_counts)
        species_proportions = [sc // count_gcd for sc in species_counts]

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
            count2 = spec_idx % 26

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
            'nsites': natoms,
            'elements': elements,
            'nelements': nelements,
            'elements_ratios': elements_ratios,
            'chemical_formula_anonymous': chemical_formula_anonymous,
            'chemical_formula_reduced': chemical_formula_reduced,
            'chemical_formula_hill': self.get_chemical_formula(),
            'dimension_types': self.get_pbc().astype(int).tolist(),
            'nperiodic_dimensions': int(sum(self.get_pbc())),
            #'species': species,  # Is this ever used?
        }

    @classmethod
    def from_ase(cls, atoms):
        """
        Generates an :class:`AtomicConfiguration` from an :code:`ase.Atoms` object.
        """
        # Workaround for bug in todict() fromdict() with constraints.
        # Merge request: https://gitlab.com/ase/ase/-/merge_requests/2574
        if atoms.constraints is not None:
            atoms.constraints = [c.todict() for c in atoms.constraints]
#This means kwargs need to be same as those in ASE
        conf = cls.fromdict(atoms.todict())

        for k, v in atoms.info.items():
            if k in [ATOMS_NAME_FIELD, ATOMS_LABELS_FIELD]:
                if not isinstance(v, set):
                    if not isinstance(v, list):
                        v = [v]

                    conf.info[k] = set(v)
                else:
                    conf.info[k] = v
            else:
                conf.info[k] = v

        for k, v in atoms.arrays.items():
            conf.arrays[k] = v

        return conf

    @staticmethod
    def aggregate_configuration_summaries(db, ids, verbose=False):
        """
          Gathers the following information from a collection of configurations:

        * :code:`nconfigurations`: the total number of configurations
        * :code:`nsites`: the total number of atoms
        * :code:`nelements`: the total number of unique element types
        * :code:`elements`: the element types
        * :code:`individual_elements_ratios`: a set of elements ratios generated
          by looping over each configuration, extracting its concentration of
          each element, and adding the tuple of concentrations to the set
        * :code:`total_elements_ratios`: the ratio of the total count of atoms
            of each element type over :code:`natoms`
        * :code:`labels`: the union of all configuration labels
        * :code:`labels_counts`: the total count of each label
        * :code:`chemical_formula_reduced`: the set of all reduced chemical
            formulae
        * :code:`chemical_formula_anonymous`: the set of all anonymous chemical
            formulae
        * :code:`chemical_formula_hill`: the set of all hill chemical formulae
        * :code:`nperiodic_dimensions`: the set of all numbers of periodic
            dimensions
        * :code:`dimension_types`: the set of all periodic boundary choices

        Args:
            db (MongoDatabase object):
                Database client in which to search for IDs
            ids (list):
                IDs of Configurations of interest
            verbose (bool, default=False):
                If True, prints a progress bar

        Returns:
            dict: Aggregated Configuration information
        """
        aggregated_info = {
            'nconfigurations': len(ids),
            'nsites': 0,
            'nelements': 0,
            'chemical_systems': set(),
            'elements': [],
            'individual_elements_ratios': {},
            'total_elements_ratios': {},
            'labels': [],
            'labels_counts': [],
            'chemical_formula_reduced': set(),
            'chemical_formula_anonymous': set(),
            'chemical_formula_hill': set(),
            'nperiodic_dimensions': set(),
            'dimension_types': set(),
        }

        for doc in tqdm(
            db.configurations.find({'_id': {'$in': ids}}),
            desc='Aggregating configuration info',
            disable=not verbose,
            total=len(ids),
            ):
            aggregated_info['nsites'] += doc['nsites']

            aggregated_info['chemical_systems'].add(''.join(doc['elements']))

            for e, er in zip(doc['elements'], doc['elements_ratios']):
                if e not in aggregated_info['elements']:
                    aggregated_info['nelements'] += 1
                    aggregated_info['elements'].append(e)
                    aggregated_info['total_elements_ratios'][e] = er*doc['nsites']
                    aggregated_info['individual_elements_ratios'][e] = set(
                        [np.round_(er, decimals=2)]
                    )
                else:
                    aggregated_info['total_elements_ratios'][e] += er*doc['nsites']
                    aggregated_info['individual_elements_ratios'][e].add(
                        np.round_(er, decimals=2)
                    )

            for l in doc['labels']:
                if l not in aggregated_info['labels']:
                    aggregated_info['labels'].append(l)
                    aggregated_info['labels_counts'].append(1)
                else:
                    idx = aggregated_info['labels'].index(l)
                    aggregated_info['labels_counts'][idx] += 1

            aggregated_info['chemical_formula_reduced'].add(doc['chemical_formula_reduced'])
            aggregated_info['chemical_formula_anonymous'].add(doc['chemical_formula_anonymous'])
            aggregated_info['chemical_formula_hill'].add(doc['chemical_formula_hill'])

            aggregated_info['nperiodic_dimensions'].add(doc['nperiodic_dimensions'])
            aggregated_info['dimension_types'].add(tuple(doc['dimension_types']))

        for e in aggregated_info['elements']:
            aggregated_info['total_elements_ratios'][e] /= aggregated_info['nsites']
            aggregated_info['individual_elements_ratios'][e] = list(aggregated_info['individual_elements_ratios'][e])

        aggregated_info['chemical_systems'] = list(aggregated_info['chemical_systems'])
        aggregated_info['chemical_formula_reduced'] = list(aggregated_info['chemical_formula_reduced'])
        aggregated_info['chemical_formula_anonymous'] = list(aggregated_info['chemical_formula_anonymous'])
        aggregated_info['chemical_formula_hill'] = list(aggregated_info['chemical_formula_hill'])
        aggregated_info['nperiodic_dimensions'] = list(aggregated_info['nperiodic_dimensions'])
        aggregated_info['dimension_types'] = list(aggregated_info['dimension_types'])

        return aggregated_info

    def __str__(self):
        ase_str = super().__str__()
        return "AtomicConfiguration(name={}, {})".format(
            self.info[ATOMS_NAME_FIELD],            ase_str[20:-1]
        )


# TODO: Any other arguments here?
#       Think about how properties are incorporated here-may want to define new class that subclasses SeqRecord first
#       Think about capitalization for hashing purposes
class BioSequenceConfiguration(BaseConfiguration, SeqRecord):
    """
    A BioSequenceConfiguration is an extension of a :class:`BaseConfiguration` and an :class:`Bio.SeqRecord object.`
    """

    unique_identifier_kw = ['sequence']

# TODO: Check seq use cases->may need to be Seq class, be required, etc
    def __init__(self, names=None, labels=None, **kwargs,):
        """
        Constructs a BioSequenceConfiguration. Calls :meth:`BaseConfiguration.__init__()`
        and :meth:`Bio.SeqRecord.__init__()`

        Args:
            seq (str):
                Biological sequence
            names (str, list of str):
                Names to be associated with a Configuration
            labels (str, list of str):
                Labels to be associated with a Configuration
            **kwargs:
                Other keyword arguments that can be passed to :meth:`Bio.SeqRecord.__init__()`
        """

        BaseConfiguration.__init__(self, names=names, labels=labels)

        if 'sequence' in list(kwargs.keys()):
            kwargs['seq'] = kwargs['sequence']
            kwargs.pop('sequence')

        SeqRecord.__init__(self, **kwargs)

    @property
    def unique_identifiers(self):
        return {'sequence': str(self.seq).encode('utf-8')}

    # TODO: create setter here

    # TODO: What things would be needed here-Count/composition, sequence length, etc
    def configuration_summary(self):
        """
        Extracts useful metadata from a sequence.

        Returns:
            dict: Keys and their associated values that will be included under a Configuration's entry in the Database
        """
        return {'seq_length': len(self.unique_identifiers['sequence'])}

    @classmethod
    def from_seqrecord(cls, seqrec):
        """
        Generates a :class:`BioSequenceConfiguration` from a :code:`Bio.SeqRecord` object.
        """
        return cls(
            seq=seqrec.seq,
            id=seqrec.id,
            name=seqrec.name,
            description=seqrec.description,
            dbxrefs=seqrec.dbxrefs[:],
            features=seqrec.features[:],
            annotations=seqrec.annotations.copy(),
            letter_annotations=seqrec.letter_annotations.copy(),
        )


class Configuration(Atoms):
    """
    A Configuration is an extension of an :class:`ase.Atoms` object that is
    guaranteed to have the following fields in its :attr:`info` dictionary:

    - :attr:`~colabfit.ATOMS_ID_FIELD` = :code:"_id"
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
            v = self.info[ATOMS_NAME_FIELD]
            if not isinstance(v, list):
                v = set([str(v)])
            else:
                v = set(v)

            self.info[ATOMS_NAME_FIELD] = v
        else:
            self.info[ATOMS_NAME_FIELD] = set()
        # TODO fix how labels are utilized
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

        # Check for name conflicts in info/arrays; would cause bug in parsing
        if set(self.info.keys()).intersection(set(self.arrays.keys())):
            raise RuntimeError(
                "The same key should not be used in both Configuration.info " \
                "and Configuration.arrays"
            )

        # # Additional fields for querying

        # atomic_symbols = self.get_chemical_symbols()
        # processed_fields = process_species_list(atomic_symbols)

        # self.info['_description'] = description
        # self.info['_last_modified'] = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
        # self.info['_elements'] = processed_fields['elements']
        # self.info['_nelements'] = processed_fields['nelements']
        # self.info['_elements_ratios'] = processed_fields['elements_ratios']
        # self.info['_chemical_formula_reduced'] = processed_fields['chemical_formula_reduced']
        # self.info['_chemical_formula_anonymous'] = processed_fields['chemical_formula_anonymous']
        # self.info['_chemical_formula_hill'] = self.get_chemical_formula()
        # self.info['_natoms'] = processed_fields['natoms']
        # # self.info['_species'] = processed_fields['species']
        # self.info['_dimension_types'] = self.get_pbc().astype(int)
        # self.info['_nperiodic_dimensions'] = sum(self.get_pbc())
        # self.info['_lattice_vectors'] = np.array(self.get_cell())

    @classmethod
    def from_ase(cls, atoms):
        """
        Generates a :class:`Configuration` from an :code:`ase.Atoms` object.
        """
        # Workaround for bug in todict() fromdict() with constraints.
        # Merge request: https://gitlab.com/ase/ase/-/merge_requests/2574
        if atoms.constraints is not None:
            atoms.constraints = [c.todict() for c in atoms.constraints]

        conf = Configuration.fromdict(atoms.todict())

        for k, v in atoms.info.items():
            if k in [ATOMS_NAME_FIELD, ATOMS_LABELS_FIELD]:
                if not isinstance(v, set):
                    if not isinstance(v, list):
                        v = [v]

                    conf.info[k] = set(v)
                else:
                    conf.info[k] = v
            else:
                conf.info[k] = v

        for k, v in atoms.arrays.items():
            conf.arrays[k] = v

        return conf

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

        Note that the positions and cell vectors are rounded to 16 decimal places
        before being compared.

        hashlib is used instead of hash() to avoid hash randomisation.
        """

        # constraints_hash = hash(tuple(
        #     hash(tuple(self.info[c])) if c in self.info
        #     else hash(np.array(self.arrays[c]).data.tobytes())
        #     for c in self.info[ATOMS_CONSTRAINTS_FIELD]
        # ))
        # TODO: Add hash update for constraints like charge, external, etc
        # TODO: Ensure order doesn't affect hash
        _hash = sha512()
        _hash.update(self.arrays['numbers'].data.tobytes()),
        _hash.update(np.round_(self.arrays['positions'], decimals=16).data.tobytes()),
        _hash.update(np.round_(np.array(self.cell), decimals=16).data.tobytes()),
        _hash.update(np.array(self.pbc).data.tobytes()),

        return int(str(int(_hash.hexdigest(), 16) - HASH_SHIFT)[:HASH_LENGTH])

    def __eq__(self, other):
        """
        Two Configurations are considered to be identical if they have the same
        hash value. This means that their atomic positions, atomic numbers,
        simulation cell vectors, and periodic boundary conditions must match to
        within 16 decimal places.
        """
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


def process_species_list(atoms):
    """Extracts useful metadata from a list of atomic species"""
    atomic_species = atoms.get_chemical_symbols()

    natoms = len(atomic_species)
    elements = sorted(list(set(atomic_species)))
    nelements = len(elements)
    elements_ratios = [
        atomic_species.count(el) / natoms for el in elements
    ]

    species_counts = [atomic_species.count(sp) for sp in elements]

    # Build per-element proportions
    from math import gcd
    from functools import reduce
    def find_gcd(list):
        x = reduce(gcd, list)
        return x

    count_gcd = find_gcd(species_counts)
    species_proportions = [sc // count_gcd for sc in species_counts]

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
        count2 = spec_idx % 26

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

# TODO: Check datatypes, decimal rounding, string encodings, etc to ensure consistent hashing
#       Add support for lists, etc
def pre_hash_formatting(v):
    """
    Ensures proper datatypes, precision, etc prior to hashing of unique identifiers

    Args:
        v:
            Value to hash

    Returns:
        Reformatted value

    """
    if isinstance(v, np.ndarray):
        if v.dtype == float:
            return np.round_(v,decimals=16)
        else:
            return v
    else:
        return v
