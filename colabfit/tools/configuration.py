import numpy as np
from hashlib import sha512
from ase import Atoms
from string import ascii_lowercase, ascii_uppercase
from Bio.SeqRecord import SeqRecord
from tqdm import tqdm
from colabfit import (
    HASH_LENGTH, HASH_SHIFT,
    ATOMS_NAME_FIELD, ATOMS_LABELS_FIELD,
    ATOMS_CONSTRAINTS_FIELD,
    SHORT_ID_STRING_NAME
)

class BaseConfiguration:
    """Abstract parent class for all Configurations.

       This class should never be directly instantiated-all other Configuration classes
       must subclass this along with any other useful classes.

       Configuration classes must pass all necessary unique identifiers as individual keyword arguments
       to their associated constructor. Unique identifiers are values that are needed to uniquely
       identify one Configuration instance from another. These values will be used to produce
       a unique hash for each Configuration instance and be added to the database with
       their associated keyword. See :attr:`unique_identifer_kw`.

       All Configuration classes must define a :code:`self.configuration_summary` method. This is used
       to extract any other useful information that will be included (in addition to all unique
       identifiers) in the Configuration's entry in the Database.

       All Configuration classes must also define a :code:`self.aggregate_configuration_summaries` method. This is used
       to extract useful information from a collection of Configurations.

       See :meth:`~colabfit.tools.configuration.AtomicConfiguration` as an example.

       Attributes:

           info (dict):
               Stores important metadata for a Configuration. At a minimum, it will include
               keywords "_name" and "_labels".
           _array_order (array):
                Optional ordering of array unique identifiers so that trivial permutations do not hash
                differently
           unique_identifier_kw (list):
               Class attribute that specifies the keywords to be used for all unique identifiers.
               All Configuration classes should accept each keyword as an argument to their constructor.
           unique_identifiers_kw_types (dict):
               Class attribute that specifies the data types of the unique
               identifier keywords. key = identifier keyword; value = identifier
               data type.
       """

# TODO: Make these read-only so as to avoid user accidentally renaming
    unique_identifier_kw = None
    unique_identifier_kw_types = None

    def __init__(self, names=None, labels=None):
        """
        Args:
            names (str, list of str):
                Names to be associated with a Configuration
            labels (str, list of str):
                Labels to be associated with a Configuration
        """
        self._array_order = None
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

        Returns:
            dict: Keys and their associated values that will be included under a Configuration's entry in the Database
        """
        raise NotImplementedError('All Configuration classes should implement this.')

    @staticmethod
    def aggregate_configuration_summaries(hashes):
        """Aggregates information for given configurations.

        All Configuration classes should implement this.
        Similar to :code:`self.configuration_summary`, but summarizes information
        for a collection of Configurations

        Args:
            hashes:
                hashes of Configurations of interest

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
        for k, v in self.unique_identifiers.items():
            _hash.update(bytes(pre_hash_formatting(k,v,self._array_order)))
        return int(_hash.hexdigest(),16)

    def __eq__(self, other):
        """
        Two Configurations are considered to be identical if they have the same
        hash value.
        """
        return hash(self) == hash(other)
#


class AtomicConfiguration(BaseConfiguration, Atoms):
    # TODO: Modify docstring
    # TODO: Don't think AtomicConfigurations will always have _id
    # TODO: Reimplement constraints
    # - :attr:`~colabfit.ATOMS_CONSTRAINTS_FIELD` = :code:"_constraints"
    """
    An AtomicConfiguration is an extension of a :class:`BaseConfiguration` and an :class:`ase.Atoms`
    object that is guaranteed to have the following fields in its :attr:`info` dictionary:

    - :attr:`~colabfit.ATOMS_NAME_FIELD` = :code:"_name"
    - :attr:`~colabfit.ATOMS_LABELS_FIELD` = :code:"_labels"
    """

    unique_identifier_kw = ['atomic_numbers', 'positions', 'cell', 'pbc']
    unique_identifier_kw_types = {
        'atomic_numbers': int,
        'positions': float,
        'cell': float,
        'pbc': bool,
    }

    def __init__(self, names=None, labels=None, **kwargs):
        """
        Constructs an AtomicConfiguration. Calls :meth:`BaseConfiguration.__init__()`
        and :meth:`ase.Atoms.__init__()`

        Args:
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
        self._array_order = np.lexsort((self.arrays['positions'][:,2],self.arrays['positions'][:,1],self.arrays['positions'][:,0]))
        self._hash = hash(self)
        # sort by x then y then z
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
            raise RuntimeError("There is a mismatch between keywords!")
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
    def aggregate_configuration_summaries(db, hashes, verbose=False):
        """
          Gathers the following information from a collection of Configurations:

        * :code:`nconfigurations`: the total number of configurations
        * :code:`nsites`: the total number of atoms
        * :code:`nelements`: the total number of unique element types
        * :code:`elements`: the element types
        * :code:`individual_elements_ratios`: a set of elements ratios generated
          by looping over each configuration, extracting its concentration of
          each element, and adding the tuple of concentrations to the set
        * :code:`total_elements_ratios`: the ratio of the total count of atoms
          of each element type over :code:`nsites`
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
            db (:code:`MongoDatabase` object):
                Database client in which to search for hashes
            hashes (list):
                hashes of Configurations of interest
            verbose (bool, default=False):
                If True, prints a progress bar

        Returns:
            dict: Aggregated Configuration information
        """
        aggregated_info = {
            'nconfigurations': len(hashes),
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
            db.configurations.find({'hash': {'$in': hashes}}),
            desc='Aggregating configuration info',
            disable=not verbose,
            total=len(hashes),
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



# TODO: string encodings, etc to ensure consistent hashing
#       Add support for lists, etc
def pre_hash_formatting(k,v,ordering):
    """
    Ensures proper datatypes, precision, etc. prior to hashing of unique identifiers

    Args:
        k:
            Key of item to hash
        v:
            Value to hash
        ordering:
            Potential ordering of arrays prior to hashing

    Returns:
        Reformatted value

    """
    # hard code for positions and numbers for now
    if k in ['atomic_numbers', 'positions']:
        v = v[ordering]
    # for now all AtomicConfiguration UIs are defined to be ndarrays
    if isinstance(v, np.ndarray):
        if v.dtype in [np.half, np.single, np.double, np.longdouble]:
            return np.round_(v.astype(np.float64),decimals=16)
        elif v.dtype in [np.int8, np.int16, np.int32, np.int64]:
            return v.astype(np.int64)
        else:
            return v
    else:
        return v
