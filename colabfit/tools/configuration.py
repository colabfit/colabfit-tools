import datetime
from string import ascii_lowercase, ascii_uppercase
from types import NoneType

import dateutil
import numpy as np
from ase import Atoms

from colabfit import ATOMS_LABELS_FIELD, ATOMS_NAME_FIELD
from colabfit.tools.schema import config_schema
from colabfit.tools.utilities import (
    _empty_dict_from_schema,
    _hash,
    _parse_unstructured_metadata,
)


class AtomicConfiguration(Atoms):
    """
    An AtomicConfiguration is an extension of a :class:`BaseConfiguration` and an
    :class:`ase.Atoms` object that is guaranteed to have the following fields in
    its :attr:`info` dictionary:

    - :attr:`~colabfit.ATOMS_NAME_FIELD` = :code:"_name"
    - :attr:`~colabfit.ATOMS_LABELS_FIELD` = :code:"_labels"
    """

    def __init__(
        self,
        co_md_map=None,
        info=None,
        **kwargs,
    ):
        """
        Constructs an AtomicConfiguration. Calls :meth:`BaseConfiguration.__init__()`
        and :meth:`ase.Atoms.__init__()`

        Args:
            co_md_map (dict) optional:
                Property map of metadata to be used to set configuration metadata for a
                configuration. This should be called at creation of AtomicConfiguration
                in order to include metadata in the hash.
            **kwargs:
                Other keyword arguments that can be passed to
                :meth:`ase.Atoms.__init__()`
        """
        names = info[ATOMS_NAME_FIELD]
        if "atomic_numbers" in list(kwargs.keys()):
            kwargs["numbers"] = kwargs.pop("atomic_numbers")

        Atoms.__init__(self, **kwargs)
        self._array_order = np.lexsort(
            (
                self.arrays["positions"][:, 2],
                self.arrays["positions"][:, 1],
                self.arrays["positions"][:, 0],
            )
        )
        self.unique_identifier_kw = [
            "atomic_numbers",
            "positions_00",
            "cell",
            "pbc",
            "metadata_id",
        ]
        self.unique_identifier_kw.extend([f"positions_{i:02d}" for i in range(1, 20)])
        self.info = info
        self.metadata = self.set_metadata(co_md_map)
        if isinstance(names, str):
            self.names = [names]
        else:
            self.names = names

        labels = self.info.pop(ATOMS_LABELS_FIELD, None)
        if isinstance(labels, str):
            self.labels = [labels]
        elif not isinstance(labels, (list, NoneType)):
            raise TypeError("Labels must be a string or a list of strings or None")
        else:
            self.labels = labels
        self.spark_row = self.to_spark_row()
        self._hash = hash(self)
        self.id = f"CO_{self._hash}"

        self.spark_row["id"] = self.id
        self.spark_row["hash"] = str(self._hash)
        # self.spark_row["dataset_ids"] = [self.dataset_id]
        self.spark_row = self.spark_row
        # Check for name conflicts in info/arrays; would cause bug in parsing
        if set(self.info.keys()).intersection(set(self.arrays.keys())):
            raise RuntimeError(
                "The same key should not be used in both Configuration.info and "
                "Configuration.arrays"
            )

    def set_metadata(self, co_md_map):
        """
        Returns metadata for a configuration from a property map.
        This should be called at creation of AtomicConfiguration in order
        to include metadata in the hash.

        Args:
            co_md_map (dict): Property map of metadata to be used to set configuration
            metadata for a configuration.

        """
        if co_md_map is None:
            co_md_map = {}
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

        return _parse_unstructured_metadata(gathered_fields)

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

    def set_dataset_id(self, dataset_id):
        self.dataset_id = dataset_id
        self.spark_row["dataset_ids"] = [dataset_id]

    def to_spark_row(self):
        co_dict = _empty_dict_from_schema(config_schema)
        co_dict["cell"] = self.cell.array.astype(float).tolist()
        co_dict["positions_00"] = self.positions.astype(float).tolist()
        co_dict["names"] = self.names
        co_dict["labels"] = self.labels
        co_dict["pbc"] = self.pbc.astype(bool).tolist()
        co_dict["last_modified"] = dateutil.parser.parse(
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        co_dict["atomic_numbers"] = self.numbers.astype(int).tolist()
        if self.metadata is not None:
            co_dict.update(self.metadata)
        co_dict.update(self.configuration_summary())
        return co_dict

    @classmethod
    def from_ase(self, atoms, co_md_map=None):
        """
        Generates an :class:`AtomicConfiguration` from an :code:`ase.Atoms` object.
        """
        # Workaround for bug in todict() fromdict() with constraints.
        # Merge request: https://gitlab.com/ase/ase/-/merge_requests/2574
        # This means kwargs need to be same as those in ASE
        dct = atoms.todict()
        kw = {name: dct.pop(name) for name in ["numbers", "positions", "cell", "pbc"]}
        constraints = dct.pop("constraints", None)
        if constraints:
            constraints = [c.todict() for c in atoms.constraints]
            from ase.constraints import dict2constraint

            constraints = [dict2constraint(d) for d in constraints]
        info = dct.pop("info", None)
        for k, v in info.items():
            if k in [ATOMS_NAME_FIELD, ATOMS_LABELS_FIELD]:
                if not isinstance(v, set):
                    if not isinstance(v, list):
                        v = [v]
                    info[k] = list(set(v))
                else:
                    info[k] = list(v)
            else:
                info[k] = v
        config = self(
            constraint=constraints,
            celldisp=dct.pop("celldisp", None),
            info=info,
            co_md_map=co_md_map,
            **kw,
        )
        natoms = len(atoms)
        for name, arr in dct.items():
            assert len(arr) == natoms, name
            assert isinstance(arr, np.ndarray)
            config.arrays[name] = arr
        # for k, v in atoms.arrays.items():
        #     conf.arrays[k] = v

        return config

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
        return NotImplementedError

    def __str__(self):
        ase_str = super().__str__()
        return "AtomicConfiguration(name={}, {})".format(
            self.info[ATOMS_NAME_FIELD], ase_str[20:-1]
        )

    def __hash__(self):
        return _hash(self.spark_row, sorted(self.unique_identifier_kw), False)
