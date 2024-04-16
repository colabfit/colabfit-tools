import datetime
import json
import re
import time
import dateutil
from collections import defaultdict
from functools import partial
from hashlib import sha512
from string import ascii_lowercase, ascii_uppercase

import numpy as np
from ase import Atoms
from Bio.SeqRecord import SeqRecord
from pyspark.sql.types import (
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)
from tqdm import tqdm

from colabfit import ATOMS_LABELS_FIELD, ATOMS_NAME_FIELD

config_schema = StructType(
    [
        StructField("id", StringType(), False),
        StructField("hash", StringType(), False),
        StructField("last_modified", TimestampType(), False),
        StructField("dataset_ids", StringType(), True),  # ArrayType(StringType())
        StructField("metadata", StringType(), True),
        StructField("chemical_formula_hill", StringType(), True),
        StructField("chemical_formula_reduced", StringType(), True),
        StructField("chemical_formula_anonymous", StringType(), True),
        StructField("elements", StringType(), True),  # ArrayType(StringType())
        StructField("elements_ratios", StringType(), True),  # ArrayType(IntegerType())
        StructField("atomic_numbers", StringType(), True),  # ArrayType(IntegerType())
        StructField("nsites", IntegerType(), True),
        StructField("nelements", IntegerType(), True),
        StructField("nperiodic_dimensions", IntegerType(), True),
        StructField("cell", StringType(), True),  # ArrayType(ArrayType(DoubleType()))
        StructField("dimension_types", StringType(), True),  # ArrayType(IntegerType())
        StructField("pbc", StringType(), True),  # ArrayType(IntegerType())
        StructField(
            "positions", StringType(), True
        ),  # ArrayType(ArrayType(DoubleType()))
        StructField("names", StringType(), True),  # ArrayType(StringType())
    ]
)


def _empty_dict_from_schema(schema):
    empty_dict = {}
    for field in schema:
        empty_dict[field.name] = None
    return empty_dict


def stringify_lists(row_dict):
    """
    Replace list/tuple fields with comma-separated strings.
    Spark and Vast both support array columns, but the connector does not,
    so keeping cell values in list format crashes the table.
    TODO: Remove when no longer necessary
    """
    for key, val in row_dict.items():
        if (
            isinstance(val, np.ndarray)
            or isinstance(val, list)  # noqa W503
            or isinstance(val, tuple)  # noqa W503
            or isinstance(val, dict)  # noqa W503
        ):
            row_dict[key] = str(val)
    return row_dict


class BaseConfiguration:
    """Abstract parent class for all Configurations.

    This class should never be directly instantiated-all other Configuration classes
    must subclass this along with any other useful classes.

    Configuration classes must pass all necessary unique identifiers as individual
    keyword arguments to their associated constructor. Unique identifiers are values
    that are needed to uniquely identify one Configuration instance from another.
    These values will be used to produce a unique hash for each Configuration instance
    and be added to the database with their associated keyword. See
    :attr:`unique_identifer_kw`.

    All Configuration classes must define a :code:`self.configuration_summary` method.
    This is used to extract any other useful information that will be included (in
    addition to all unique identifiers) in the Configuration's entry in the Database.

    All Configuration classes must also define a
    :code:`self.aggregate_configuration_summaries` method. This is used to extract
    useful information from a collection of Configurations.

    See :meth:`~colabfit.tools.configuration.AtomicConfiguration` as an example.

    Attributes:

        info (dict):
            Stores important metadata for a Configuration. At a minimum, it will include
            keywords "_name" and "_labels".
        _array_order (array):
             Optional ordering of array unique identifiers so that trivial permutations
             do not hash differently
        unique_identifier_kw (list):
            Class attribute that specifies the keywords to be used for all unique
            identifiers.
            All Configuration classes should accept each keyword as an argument to
            their constructor.
        unique_identifier_kw_types (dict):
            Class attribute that specifies the data types of the unique
            identifier keywords. key = identifier keyword; value = identifier
            data type.
    """

    def __init__(self, names=None):
        """
        Args:
            names (str, list of str):
                Names to be associated with a Configuration
        """
        self.unique_identifier_kw = []
        self.unique_identifier_kw_types = {}
        self._array_order = None
        self.info = {}
        if names is None:
            self.info[ATOMS_NAME_FIELD] = set()
        else:
            self.info[ATOMS_NAME_FIELD] = set(list(names))

    @property
    def unique_identifiers(self):
        raise NotImplementedError("All Configuration classes should implement this.")

    @unique_identifiers.setter
    def unique_identifiers(self, values):
        raise NotImplementedError("All Configuration classes should implement this.")

    def configuration_summary(self):
        """Extracts useful information from a Configuration.

        All Configuration classes should implement this.
        Any useful information that should be included under a Configuration's entry in
        the Database (in addition to its unique identifiers) should be extracted and
        added to a dict.

        Returns:
            dict: Keys and their associated values that will be included under a
            Configuration's entry in the Database
        """
        raise NotImplementedError("All Configuration classes should implement this.")

    @staticmethod
    def aggregate_configuration_summaries(db, hashes):
        """Aggregates information for given configurations.

        All Configuration classes should implement this.
        Similar to :code:`self.configuration_summary`, but summarizes information
        for a collection of Configurations

        Args:
            db:
                database where aggregation should occur
            hashes:
                hashes of Configurations of interest

        Returns:
            dict: Key-value pairs of information aggregated from multiple Configurations
        """
        raise NotImplementedError("All Configuration classes should implement this.")

    def __hash__(self):
        """Generates a hash for :code:`self`.

        Hashes all values in self.unique_identifiers.
        hashlib is used instead of hash() to avoid hash randomisation.

        Returns:
            int: Value of hash
        """
        if len(self.unique_identifiers) == 0:
            raise Exception("Ensure unique identifiers are properly defined!")
        _hash = sha512()
        for k, v in self.unique_identifiers.items():
            _hash.update(bytes(pre_hash_formatting(k, v, self._array_order)))
        return int(_hash.hexdigest(), 16)

    def __eq__(self, other):
        """
        Two Configurations are considered to be identical if they have the same
        hash value.
        """
        return hash(self) == hash(other)


#


class AtomicConfiguration(BaseConfiguration, Atoms):
    """
    An AtomicConfiguration is an extension of a :class:`BaseConfiguration` and an
    :class:`ase.Atoms` object that is guaranteed to have the following fields in
    its :attr:`info` dictionary:

    - :attr:`~colabfit.ATOMS_NAME_FIELD` = :code:"_name"
    - :attr:`~colabfit.ATOMS_LABELS_FIELD` = :code:"_labels"
    """

    unique_identifier_kw = ["atomic_numbers", "positions", "cell", "pbc"]
    unique_identifier_kw_types = {
        "atomic_numbers": int,
        "positions": float,
        "cell": float,
        "pbc": bool,
    }

    def __init__(self, names=None, **kwargs):
        """
        Constructs an AtomicConfiguration. Calls :meth:`BaseConfiguration.__init__()`
        and :meth:`ase.Atoms.__init__()`

        Args:
            names (str, list of str):
                Names to be associated with a Configuration
            **kwargs:
                Other keyword arguments that can be passed to
                :meth:`ase.Atoms.__init__()`
        """

        BaseConfiguration.__init__(self, names=names)
        self.metadata = None

        kwargs["info"] = self.info
        if "atomic_numbers" in list(kwargs.keys()):
            kwargs["numbers"] = kwargs["atomic_numbers"]
            kwargs.pop("atomic_numbers")

        Atoms.__init__(self, **kwargs)
        self._array_order = np.lexsort(
            (
                self.arrays["positions"][:, 2],
                self.arrays["positions"][:, 1],
                self.arrays["positions"][:, 0],
            )
        )
        self._hash = hash(self)
        self.id = f"CO_{self._hash}"
        # Check for name conflicts in info/arrays; would cause bug in parsing
        if set(self.info.keys()).intersection(set(self.arrays.keys())):
            raise RuntimeError(
                "The same key should not be used in both Configuration.info and "
                "Configuration.arrays"
            )

    def set_metadata(self, co_md_map):
        """Sets metadata for a configuration from a property map.

        Args:
            co_md_map (dict): Property map of metadata to be used to set configuration
            metadata for a configuration.

        """
        gathered_fields = {}
        for md_field in co_md_map.keys():
            if "value" in co_md_map[md_field]:
                v = co_md_map[md_field]["value"]
            elif "field" in co_md_map[md_field]:
                field_key = co_md_map[md_field]["field"]

                if field_key in self.info:
                    v = self.info[field_key]
                elif field_key in self.arrays:
                    v = self.arrays[field_key]
                else:
                    # No keys are required; ignored if missing
                    continue
            else:
                # No keys are required; ignored if missing
                continue

            if "units" in co_md_map[md_field]:
                gathered_fields[md_field] = {
                    f"{md_field}": v,
                    f"{md_field}_unit": co_md_map[md_field]["units"],
                }
            else:
                gathered_fields[md_field] = v
        self.metadata = json.dumps(gathered_fields)

    @property
    def unique_identifiers(self):
        return {
            "atomic_numbers": self.get_atomic_numbers(),
            "positions": self.get_positions(),
            "cell": np.array(self.get_cell()),
            "pbc": self.get_pbc().astype(int),
        }

    @unique_identifiers.setter
    def unique_identifiers(self, d):
        if set(self.unique_identifier_kw) != set(list(d.keys())):
            raise RuntimeError("There is a mismatch between keywords!")
        # Make sure ASE values are in sync
        self.arrays["numbers"] = d["atomic_numbers"]
        self.arrays["positions"] = d["positions"]
        self.cell = d["cell"]
        self.pbc = d["pbc"]

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
            dict: Keys and their associated values that will be included under a
            Configuration's entry in the Database
        """

        atomic_species = self.get_chemical_symbols()

        natoms = len(atomic_species)
        elements = sorted(list(set(atomic_species)))
        nelements = len(elements)
        elements_ratios = [atomic_species.count(el) / natoms for el in elements]

        species_counts = [atomic_species.count(sp) for sp in elements]

        # Build per-element proportions
        from functools import reduce
        from math import gcd

        def find_gcd(_list):
            x = reduce(gcd, _list)
            return x

        count_gcd = find_gcd(species_counts)
        species_proportions = [sc // count_gcd for sc in species_counts]

        chemical_formula_reduced = ""
        for elem, elem_prop in zip(elements, species_proportions):
            chemical_formula_reduced += "{}{}".format(
                elem, "" if elem_prop == 1 else str(elem_prop)
            )

        # Replace elements with A, B, C, ...
        species_proportions = sorted(species_proportions, reverse=True)

        chemical_formula_anonymous = ""
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
            species.append(
                {
                    "name": el,
                    "chemical_symbols": [el],
                    "concentration": [1.0],
                }
            )

        return {
            "nsites": natoms,
            "elements": elements,
            "nelements": nelements,
            "elements_ratios": elements_ratios,
            "chemical_formula_anonymous": chemical_formula_anonymous,
            "chemical_formula_reduced": chemical_formula_reduced,
            "chemical_formula_hill": self.get_chemical_formula(),
            "dimension_types": self.get_pbc().astype(int).tolist(),
            "nperiodic_dimensions": int(sum(self.get_pbc())),
            # 'species': species,  # Is this ever used?
        }

    def to_spark_row(self):
        co_dict = _empty_dict_from_schema(config_schema)
        co_dict["hash"] = self._hash
        co_dict["id"] = self.id
        co_dict["cell"] = self.cell.array
        co_dict["positions"] = self.positions
        co_dict["names"] = self.info[ATOMS_NAME_FIELD]
        co_dict["pbc"] = self.pbc
        co_dict["last_modified"] = dateutil.parser.parse(
            datetime.datetime.now(tz=datetime.timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
        )
        co_dict["atomic_numbers"] = self.numbers
        co_dict["metadata"] = self.metadata
        co_dict.update(self.configuration_summary())
        # co_dict = stringify_lists(co_dict)
        return co_dict

    @classmethod
    def from_ase(cls, atoms):
        """
        Generates an :class:`AtomicConfiguration` from an :code:`ase.Atoms` object.
        """
        # Workaround for bug in todict() fromdict() with constraints.
        # Merge request: https://gitlab.com/ase/ase/-/merge_requests/2574
        if atoms.constraints is not None:
            atoms.constraints = [c.todict() for c in atoms.constraints]
        # This means kwargs need to be same as those in ASE
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
    def aggregate_configuration_summaries(db, co_hashes, verbose=False):
        """
        Gathers the following information from a collection of Configurations:

        * :code:`nconfigurations`: the total number of configurations
        * :code:`nsites`: the total number of atoms
        * :code:`nelements`: the total number of unique element types
        * :code:`elements`: the element types
        # * :code:`individual_elements_ratios`: a set of elements ratios generated
        #   by looping over each configuration, extracting its concentration of
        #   each element, and adding the tuple of concentrations to the set
        * :code:`total_elements_ratios`: the ratio of the total count of atoms
        of each element type over :code:`nsites`
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
                Database client in which to search for dataset-ID
            co_hashes (list of str):
                List of hashes of configurations to aggregate. /
            verbose (bool, default=False):
                If True, prints a progress bar

        Returns:
            dict: Aggregated Configuration information
        """
        s = time.time()
        hash_batch_size = 10000
        hash_batches = [
            co_hashes[i : i + hash_batch_size]
            for i in range(0, len(co_hashes), hash_batch_size)
        ]
        partial_agg = partial(agg, db_name=db.database_name, uri=db.uri)
        batch_results = []
        for batch in tqdm(
            hash_batches,
            desc="Aggregating configuration formula info",
            disable=not verbose,
        ):
            batch_results.append(partial_agg(co_hashes=batch))
        chemical_systems = set()
        chemical_formula_hill = set()
        chemical_formula_anonymous = set()
        chemical_formula_reduced = set()
        chemical_formula_hill_counts = defaultdict(int)

        for res in batch_results:
            chemical_systems.update(res["chemical_systems"])
            chemical_formula_hill.update(res["chemical_formula_hill"])
            chemical_formula_anonymous.update(res["chemical_formula_anonymous"])
            chemical_formula_reduced.update(res["chemical_formula_reduced"])
            for doc in res["chemical_formula_hill_counts"]:
                formula = doc["_id"]
                count = doc["count"]
                chemical_formula_hill_counts[formula] += count

        totals_pipeline_contents = [
            {
                "$facet": {
                    "total_configurations": [
                        {"$group": {"_id": None, "total_configurations": {"$sum": 1}}},
                        {"$project": {"_id": 0, "total_configurations": 1}},
                    ],
                    "nsites_total": [
                        {"$group": {"_id": None, "nsites_total": {"$sum": "$nsites"}}},
                        {"$project": {"_id": 0, "nsites_total": 1}},
                    ],
                    "set_elements": [
                        {"$unwind": "$elements"},
                        {
                            "$group": {
                                "_id": None,
                                "elements": {"$addToSet": "$elements"},
                            }
                        },
                        {"$project": {"_id": 0, "elements": 1}},
                    ],
                    "nperiodic_dimensions": [
                        {
                            "$group": {
                                "_id": None,
                                "nperiodic_dimensions": {
                                    "$addToSet": "$nperiodic_dimensions"
                                },
                            }
                        },
                        {"$project": {"_id": 0, "nperiodic_dimensions": 1}},
                    ],
                    "dimension_types": [
                        {
                            "$group": {
                                "_id": None,
                                "dimension_types": {"$addToSet": "$dimension_types"},
                            }
                        },
                        {"$project": {"_id": 0, "dimension_types": 1}},
                    ],
                }
            }
        ]
        nconfigurations = 0
        nsites = 0
        elements = set()
        nperiodic_dimensions = set()
        dimension_types = set()
        for co_hash_batch in tqdm(
            hash_batches,
            desc="Aggregating configuration totals info",
            disable=not verbose,
        ):
            query = {"hash": {"$in": co_hash_batch}}
            pipeline = [{"$match": query}, *totals_pipeline_contents]
            res = next(db.configurations.aggregate(pipeline))
            nconfigurations += res["total_configurations"][0]["total_configurations"]
            nsites += res["nsites_total"][0]["nsites_total"]
            elements.update(res["set_elements"][0]["elements"])
            nperiodic_dimensions.update(
                set(res["nperiodic_dimensions"][0]["nperiodic_dimensions"])
            )
            dimension_types.update(
                set([tuple(x) for x in res["dimension_types"][0]["dimension_types"]])
            )

        elem_match = re.compile(r"(?P<elem>[A-Z][a-z]?)(?P<num>\d*)")
        elem_count = defaultdict(int)
        for formula, count in chemical_formula_hill_counts.items():
            elems = elem_match.findall(formula)
            for elem, e_count in elems:
                elem_count[elem] += (int(e_count) * count) if e_count else count
        total_elems = sum(elem_count.values())
        total_elements_ratios = {
            k: elem_count[k] / total_elems for k in sorted(list(elem_count.keys()))
        }

        aggregated_info = {
            "nconfigurations": nconfigurations,
            "nsites": nsites,
            "nelements": len(elements),
            "chemical_systems": list(chemical_systems),
            "elements": list(elements),
            "total_elements_ratios": total_elements_ratios,
            "chemical_formula_reduced": list(chemical_formula_reduced),
            "chemical_formula_anonymous": list(chemical_formula_anonymous),
            "chemical_formula_hill": list(chemical_formula_hill),
            "nperiodic_dimensions": list(nperiodic_dimensions),
            "dimension_types": list(dimension_types),
        }
        print("Configuration aggregation time:", time.time() - s)

        return aggregated_info

    def __str__(self):
        ase_str = super().__str__()
        return "AtomicConfiguration(name={}, {})".format(
            self.info[ATOMS_NAME_FIELD], ase_str[20:-1]
        )


class BioSequenceConfiguration(BaseConfiguration, SeqRecord):
    """
    A BioSequenceConfiguration is an extension of a :class:`BaseConfiguration` and an
    :class:`Bio.SeqRecord object.`
    """

    unique_identifier_kw = ["sequence"]

    def __init__(
        self,
        names=None,
        **kwargs,
    ):
        """
        Constructs a BioSequenceConfiguration. Calls
        :meth:`BaseConfiguration.__init__()` and :meth:`Bio.SeqRecord.__init__()`

        Args:
            names (str, list of str):
                Names to be associated with a Configuration
            labels (str, list of str):
                Labels to be associated with a Configuration
            **kwargs:
                Other keyword arguments that can be passed to
                :meth:`Bio.SeqRecord.__init__()`
        """

        BaseConfiguration.__init__(self, names=names)

        if "sequence" in list(kwargs.keys()):
            kwargs["seq"] = kwargs["sequence"]
            kwargs.pop("sequence")

        SeqRecord.__init__(self, **kwargs)

    @property
    def unique_identifiers(self):
        return {"sequence": str(self.seq).encode("utf-8")}

    # TODO: create setter here

    # TODO: What things would be needed here-Count/composition, sequence length, etc
    def configuration_summary(self):
        """
        Extracts useful metadata from a sequence.

        Returns:
            dict: Keys and their associated values that will be included under a
            Configuration's entry in the Database
        """
        return {"seq_length": len(self.unique_identifiers["sequence"])}

    @classmethod
    def from_seqrecord(cls, seqrec):
        """
        Generates a :class:`BioSequenceConfiguration` from a :code:`Bio.SeqRecord`
        object.
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
def pre_hash_formatting(k, v, ordering):
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
    if k in ["atomic_numbers", "positions"]:
        v = v[ordering]
    # for now all AtomicConfiguration UIs are defined to be ndarrays
    if isinstance(v, np.ndarray):
        if v.dtype in [np.half, np.single, np.double, np.longdouble]:
            return np.round_(v.astype(np.float64), decimals=16)
        elif v.dtype in [np.int8, np.int16, np.int32, np.int64]:
            return v.astype(np.int64)
        else:
            return v
    else:
        return v


def agg(db_name, uri, co_hashes):
    """Gather large aggregation documents in batches"""
    from colabfit.tools.database import MongoDatabase

    client = MongoDatabase(database_name=db_name, uri=uri)
    query = {"hash": {"$in": co_hashes}}
    pipeline = [
        {"$match": query},
        {
            "$facet": {
                "chemical_formula_hill_counts": [
                    {
                        "$group": {
                            "_id": "$chemical_formula_hill",
                            "count": {"$sum": 1},
                        }
                    },
                ],
                "chemical_formula_anonymous": [
                    {
                        "$group": {
                            "_id": None,
                            "chemical_formula_anonymous": {
                                "$addToSet": "$chemical_formula_anonymous"
                            },
                        }
                    },
                    {"$project": {"_id": 0, "chemical_formula_anonymous": 1}},
                ],
                "chemical_formula_reduced": [
                    {
                        "$group": {
                            "_id": None,
                            "chemical_formula_reduced": {
                                "$addToSet": "$chemical_formula_reduced"
                            },
                        }
                    },
                    {"$project": {"_id": 0, "chemical_formula_reduced": 1}},
                ],
                "chemical_systems": [
                    {
                        "$group": {
                            "_id": None,
                            "chemical_systems": {"$addToSet": "$elements"},
                        }
                    },
                    {"$project": {"_id": 0, "chemical_systems": 1}},
                ],
            }
        },
    ]
    results = next(client.configurations.aggregate(pipeline))
    chemical_systems = [
        "".join(sorted(x)) for x in results["chemical_systems"][0]["chemical_systems"]
    ]

    chemical_formula_hill_counts = results["chemical_formula_hill_counts"]
    chemical_formula_hill = [x["_id"] for x in chemical_formula_hill_counts]
    chemical_formula_anonymous = results["chemical_formula_anonymous"][0][
        "chemical_formula_anonymous"
    ]
    chemical_formula_reduced = results["chemical_formula_reduced"][0][
        "chemical_formula_reduced"
    ]

    client.close()

    return {
        "chemical_systems": chemical_systems,
        "chemical_formula_reduced": chemical_formula_reduced,
        "chemical_formula_anonymous": chemical_formula_anonymous,
        "chemical_formula_hill": chemical_formula_hill,
        "chemical_formula_hill_counts": chemical_formula_hill_counts,
    }
