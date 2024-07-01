import datetime
import itertools
import json
import os
import tempfile
import warnings
from copy import deepcopy

# import dateutil
import dateutil.parser

# EDN_KEY_MAP = {
#     "energy": "unrelaxed-potential-energy",
#     "forces": "unrelaxed-potential-forces",
#     "stress": "unrelaxed-cauchy-stress",
#     "virial": "unrelaxed-cauchy-stress",
# }
import kim_edn
import numpy as np
from kim_property import (
    check_instance_optional_key_marked_required_are_present,
    kim_property_create,
)
from kim_property.create import KIM_PROPERTIES
from kim_property.definition import PROPERTY_ID as VALID_KIM_ID

from colabfit.tools.configuration import AtomicConfiguration
from colabfit.tools.schema import property_object_schema
from colabfit.tools.utilities import (
    _empty_dict_from_schema,
    _hash,
    _parse_unstructured_metadata,
)

# from ase.units import create_units


# HASH_LENGTH = 12
# HASH_SHIFT = 0
# # HASH_SHIFT = 2**63

# ID_FORMAT_STRING = "{}_{}_{:0d}"

# MAX_STRING_LENGTH = 255
# STRING_DTYPE_SPECIFIER = f"S{MAX_STRING_LENGTH}"

# SHORT_ID_STRING_NAME = "colabfit-id"
# EXTENDED_ID_STRING_NAME = "extended-id"

# ATOMS_NAME_FIELD = "_name"
# ATOMS_LABELS_FIELD = "_labels"
# ATOMS_LAST_MODIFIED_FIELD = "_last_modified"
# ATOMS_CONSTRAINTS_FIELD = "_constraints"

# DEFAULT_PROPERTY_NAME = (
#     "configuration-nonorthogonal-periodic-3d-cell-fixed-particles-fixed"
# )


# UNITS = create_units("2014")

# # Make GPa the base unit
# UNITS["bar"] = 1e-4  # bar to GPa
# UNITS["kilobar"] = 1e-1  # kilobar to GPa
# UNITS["pascal"] = 1e-9  # pascal to GPa
# UNITS["GPa"] = 1

# UNITS["angstrom"] = UNITS["Ang"]

# OPENKIM_PROPERTY_UNITS = {
#     "energy": "eV",
#     "forces": "eV/angstrom",
#     "stress": "GPa",
#     "unrelaxed-potential-energy": "eV",
#     "unrelaxed-potential-forces": "eV/angstrom",
#     "unrelaxed-cauchy-stress": "GPa",
# }


# These are fields that are related to the geometry of the atomic structure
# or the OpenKIM Property Definition and shouldn't be used for equality checks
_ignored_fields = [
    "property-title",
    "property-description",
    "instance-id",
    "species",
    "unrelaxed-configuration-positions",
    "unrelaxed-periodic-cell-vector-1",
    "unrelaxed-periodic-cell-vector-2",
    "unrelaxed-periodic-cell-vector-3",
]
_hash_ignored_fields = [
    "id",
    "hash",
    "last_modified",
    "energy_conjugate_with_forces",
    "energy_conjugate_with_forces_unit",
    "energy_conjugate_with_forces_per_atom",
    "energy_conjugate_with_forces_reference",
    "energy_conjugate_with_forces_reference_unit",
    "energy_conjugate_with_forces_property_id",
    "energy_conjugate_with_forces_column",
    "multiplicity",
    "metadata_path",
    "metadata_size",
]


def energy_to_schema(prop_name, en_prop: dict):
    new_name = prop_name.replace("-", "_")
    en_dict = {
        f"{new_name}_property_id": en_prop["property-id"],
        f"{new_name}": en_prop["energy"]["source-value"],
        f"{new_name}_unit": en_prop["energy"]["source-unit"],
        f"{new_name}_per_atom": en_prop["per-atom"]["source-value"],
    }
    ref_en = en_prop.get("reference-energy")
    if ref_en is not None:
        ref_en = ref_en["source-value"]
    ref_unit = en_prop.get("reference-energy")
    if ref_unit is not None:
        ref_unit = ref_unit["source-unit"]
    en_dict[f"{new_name}_reference"] = ref_en
    en_dict[f"{new_name}_reference_unit"] = ref_unit

    return en_dict


def atomic_forces_to_schema(af_prop: dict):
    af_dict = {
        "atomic_forces_property_id": af_prop["property-id"],
        "atomic_forces_00": af_prop["forces"]["source-value"],
        "atomic_forces_unit": af_prop["forces"]["source-unit"],
    }
    return af_dict


def cauchy_stress_to_schema(cs_prop: dict):
    cs_dict = {
        "cauchy_stress_property_id": cs_prop["property-id"],
        "cauchy_stress": cs_prop["stress"]["source-value"],
        "cauchy_stress_unit": cs_prop["stress"]["source-unit"],
        "cauchy_stress_volume_normalized": cs_prop["volume-normalized"]["source-value"],
    }
    return cs_dict


def band_gap_to_schema(bg_prop: dict):
    bg_dict = {
        "band_gap_property_id": bg_prop["property-id"],
        "band_gap": bg_prop["band-gap"]["source-value"],
        "band_gap_unit": bg_prop["band-gap"]["source-unit"],
    }
    return bg_dict


prop_to_row_mapper = {
    "energy": energy_to_schema,
    "atomic-forces": atomic_forces_to_schema,
    "cauchy-stress": cauchy_stress_to_schema,
    "band-gap": band_gap_to_schema,
}


def md_from_map(pmap_md, config: AtomicConfiguration) -> tuple:
    """
    Extract metadata from a property map.
    Returns metadata dict as a JSON string, method, and software.
    """
    gathered_fields = {}
    for md_field in pmap_md.keys():
        if "value" in pmap_md[md_field]:
            v = pmap_md[md_field]["value"]
        elif "field" in pmap_md[md_field]:
            field_key = pmap_md[md_field]["field"]

            if field_key in config.info:
                v = config.info[field_key]
            elif field_key in config.arrays:
                v = config.arrays[field_key]
            else:
                # No keys are required; ignored if missing
                continue
        else:
            # No keys are required; ignored if missing
            continue

        if "units" in pmap_md[md_field]:
            gathered_fields[md_field] = {
                "source-value": v,
                "source-unit": pmap_md[md_field]["units"],
            }
        else:
            gathered_fields[md_field] = {"source-value": v}
    method = gathered_fields.pop("method", None)
    software = gathered_fields.pop("software", None)
    if method is not None:
        method = method["source-value"]
    if software is not None:
        software = software["source-value"]
    return gathered_fields, method, software


class PropertyParsingError(Exception):
    pass


class MissingPropertyFieldWarning(Warning):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class InvalidPropertyDefinition(Exception):
    pass


# TODO:Remove Configuration and add Property Setting
class Property(dict):
    """
    A Property is used to store the results of some kind of calculation or
    experiment, and should be mapped to an `OpenKIM Property Definition
    <https://openkim.org/doc/schema/properties-framework/>`_. Best
    practice is for the Property to also point to one or more
    PropertySettings objects that fully define the conditions under which the
    Property was obtained.

    Attributes:

        definition (dict):
            A KIM Property Definition

        instance (dict):
            A dictionary defining an OpenKIM Property Instance.
            For more details, see the `OpenKIM Property Framework
            <https://openkim.org/doc/schema/properties-framework/>`_
            documentation. In most cases, this dictionary should not be manually
            constructed by the user. Instead, a Property Definition and a
            Configuration should be passed to the :meth:`from_definition`
            function.


        property_map (dict):
            key = a string that can be used as a key like :code:`self.instance[key]`

            value = A sub-dictionary with the following keys:

            * :attr:`field`:
                A field name used to access :attr:`Configuration.info` or
                :attr:`Configuration.arrays`
            * :attr:`units`:
                A string matching one of the units names in
                `ase.units <https://wiki.fysik.dtu.dk/ase/ase/units.html>`_.
                These units will be used to convert the given units to eV, Angstrom,
                a.m.u., Kelvin, ... For compound units (e.g. "eV/Ang"), the string will
                be split on '*' and '/'. The raw data will be multiplied by the first
                unit and anything preceded by a '*'. It will be divided by anything
                preceded by a '/'.




    """

    _observers = []

    def __init__(
        self,
        definitions,
        instance,
        property_map=None,
        metadata=None,
        convert_units=False,
        dataset_id=None,
        energy_conjugate=None,
    ):
        """
        Args:

            definitions (list(dict)):
                KIM Property Definitions

            instance (dict):
                A dictionary defining an OpenKIM Property Instance

            property_map (dict):
                A property map as described in the Property attributes section.

            convert_units (bool):
                If True, converts units to those expected by ColabFit. Default
                is False
        """
        self.unique_identifier_kw = [
            k
            for k in property_object_schema.fieldNames()
            if k not in _hash_ignored_fields
        ]
        self.instance = instance
        self.definitions = definitions

        if property_map is not None:
            self.property_map = dict(property_map)
        else:
            self.property_map = {}
        self.metadata = _parse_unstructured_metadata(metadata)

        if convert_units:
            self.convert_units()
        self.energy_conjugate = energy_conjugate
        self.chemical_formula_hill = instance.pop("chemical_formula_hill")
        self.spark_row = self.to_spark_row()
        self._hash = hash(self)
        self.spark_row["hash"] = self._hash
        self._id = f"PO_{self._hash}"
        self.spark_row["id"] = self._id
        if dataset_id is not None:
            self.spark_row["dataset_id"] = dataset_id

    @property
    def instance(self):
        return self._instance

    @instance.setter
    def instance(self, edn):
        self._instance = edn

        fields = []
        for key in self._instance:
            if key in _ignored_fields:
                continue

            # fields.append(EDN_KEY_MAP.get(key, key))
            fields.append(key)

        self._property_fields = fields

    @property
    def property_fields(self):
        return self._property_fields

    @classmethod
    def get_kim_instance(
        cls,
        definition,
        # configuration
    ):
        """
        A function for constructing a Property given a property setting hash, a property
        definition, and a property map.


        Args:

            definition (dict):
                A valid KIM Property Definition

            configuration (AtomicConfiguration):
                An AtomicConfiguration object from which to extract the property data

            property_map (dict):
                A property map as described in the Property attributes section.

        """

        global KIM_PROPERTIES

        load_from_existing = False

        if isinstance(definition, dict):
            # TODO: this is probably slowing things down a bit
            dummy_dict = deepcopy(definition)

            # Spoof if necessary
            if VALID_KIM_ID.match(dummy_dict["property-id"]) is None:
                # Invalid ID. Try spoofing it
                dummy_dict["property-id"] = "tag:@,0000-00-00:property/"
                dummy_dict["property-id"] += definition["property-id"]

            load_from_existing = dummy_dict["property-id"] in KIM_PROPERTIES
            definition_name = dummy_dict["property-id"]

        elif isinstance(definition, str):
            if os.path.isfile(definition):
                dummy_dict = kim_edn.load(definition)

                # Check if you need to spoof the property ID to trick OpenKIM
                if VALID_KIM_ID.match(dummy_dict["property-id"]) is None:
                    # Invalid ID. Try spoofing it
                    dummy_dict["property-id"] = (
                        "tag:@,0000-00-00:property/" + dummy_dict["property-id"]
                    )
                    warnings.warn(
                        "Invalid KIM property-id; Temporarily "
                        f"renaming to {dummy_dict['property-id']}"
                    )

                load_from_existing = dummy_dict["property-id"] in KIM_PROPERTIES
                definition_name = dummy_dict["property-id"]

            else:
                # Then this has to be an existing (or added) KIM Property Definition
                load_from_existing = True

                # It may have been added previously, but spoofed
                if VALID_KIM_ID.match(definition) is None:
                    # Invalid ID. Try spoofing it
                    definition_name = "tag:@,0000-00-00:property/" + definition

        else:
            raise InvalidPropertyDefinition(
                "Property definition must either be a dictionary "
                "or a path to an EDN file"
            )

        if load_from_existing:
            instance = kim_edn.loads(
                kim_property_create(
                    instance_id=1,
                    property_name=definition_name,
                )
            )[0]
        else:
            with tempfile.NamedTemporaryFile("w") as tmp:
                # Hack to avoid the fact that "property-name" has to be a dictionary
                # in order for OpenKIM's check_property_definition to work

                if "property-name" in dummy_dict:
                    del dummy_dict["property-name"]

                tmp.write(json.dumps(dummy_dict))
                tmp.flush()

                instance = kim_edn.loads(
                    kim_property_create(
                        instance_id=1,
                        property_name=tmp.name,
                    )
                )[0]

        # update_edn_with_conf(instance, configuration)
        return instance

    @classmethod
    def from_definition(
        cls,
        definitions,
        configuration,
        property_map,
        energy_conjugate=None,
        # convert_units=False
    ):
        """
        A function for constructing a Property given a property setting hash, a property
        definition, and a property map.


        Args:

            definition (dict):
                A valid KIM Property Definition

            configuration (AtomicConfiguration):
                An AtomicConfiguration object from which to extract the property data

            property_map (dict):
                A property map as described in the Property attributes section.

            energy_conjugate (str):
                The energy column that is conjugate with forces, to be used for
                values in the columns "energy_conjugate_with_forces*"

        """
        pdef_dict = {pdef["property-name"]: pdef for pdef in definitions}
        instances = {
            pdef_name: cls.get_kim_instance(pdef)
            for pdef_name, pdef in pdef_dict.items()
        }
        props_dict = {}
        pi_md = None
        for pname, pmap_list in property_map.items():
            instance = instances.get(pname, None)
            if pname == "_metadata":
                pi_md, method, software = md_from_map(pmap_list, configuration)
                props_dict["method"] = method
                props_dict["software"] = software
            elif instance is None:
                raise PropertyParsingError(f"Property {pname} not found in definitions")
            else:
                instance = instance.copy()
                for pmap_i, pmap in enumerate(pmap_list):
                    for key, val in pmap.items():
                        if "value" in val:
                            # Default value provided
                            data = val["value"]
                        elif val["field"] in configuration.info:
                            data = configuration.info[val["field"]]
                        elif val["field"] in configuration.arrays:
                            data = configuration.arrays[val["field"]]
                        else:
                            # Key not found on configurations. Will be checked later
                            continue

                        if isinstance(data, (np.ndarray, list)):
                            data = np.atleast_1d(data).tolist()
                        elif isinstance(data, np.integer):
                            data = int(data)
                        elif isinstance(data, np.floating):
                            data = float(data)
                        elif isinstance(data, (str, bool, int, float)):
                            pass
                        instance[key] = {
                            "source-value": data,
                        }

                        if (val["units"] != "None") and (val["units"] is not None):
                            instance[key]["source-unit"] = val["units"]
                # hack to get around OpenKIM requiring the property-name be a dict
                prop_name_tmp = pdef_dict[pname].pop("property-name")

                check_instance_optional_key_marked_required_are_present(
                    instance, pdef_dict[pname]
                )
                pdef_dict[pname]["property-name"] = prop_name_tmp

                # Would we be handling multiple instances of the same property?
                # props_dict[pname].append(
                #     {k: v for k, v in instance.items() if k not in _ignored_fields}
                # )
                props_dict[pname] = {k: v for k, v in instance.items()}
        props_dict["chemical_formula_hill"] = configuration.get_chemical_formula()
        props_dict["configuration_id"] = configuration.id

        return cls(
            definitions=definitions,
            property_map=property_map,
            instance=props_dict,
            metadata=pi_md,
            dataset_id=configuration.dataset_id,
            energy_conjugate=energy_conjugate,
            # convert_units=convert_units,
        )

    def to_spark_row(self):
        """
        Convert the Property to a Spark Row object
        """
        row_dict = _empty_dict_from_schema(property_object_schema)
        row_dict.update(self.metadata)
        for key, val in self.instance.items():
            if key == "method":
                row_dict["method"] = val
            elif key == "software":
                row_dict["software"] = val
            elif key == "_metadata":
                continue
            elif key == "configuration_id":
                row_dict["configuration_id"] = val
            elif "energy" in key:
                row_dict.update(prop_to_row_mapper["energy"](key, val))
            else:
                row_dict.update(prop_to_row_mapper[key](val))
        row_dict["last_modified"] = dateutil.parser.parse(
            datetime.datetime.now(tz=datetime.timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
        )
        row_dict["chemical_formula_hill"] = self.chemical_formula_hill
        row_dict["multiplicity"] = 1
        # if self.energy_conjugate is not None:
        #     if row_dict[self.energy_conjugate] is None:
        #         raise warnings.warn(
        #             f"Energy conjugate {self.energy_conjugate} not found in property"
        #         )
        #     row_dict["energy_conjugate_with_forces"] = row_dict[self.energy_conjugate]
        #     row_dict["energy_conjugate_with_forces_unit"] = row_dict[
        #         f"{self.energy_conjugate}_unit"
        #     ]
        #     row_dict["energy_conjugate_with_forces_per_atom"] = row_dict[
        #         f"{self.energy_conjugate}_per_atom"
        #     ]
        #     row_dict["energy_conjugate_with_forces_reference"] = row_dict[
        #         f"{self.energy_conjugate}_reference"
        #     ]
        #     row_dict["energy_conjugate_with_forces_reference_unit"] = row_dict[
        #         f"{self.energy_conjugate}_reference_unit"
        #     ]
        #     row_dict["energy_conjugate_with_forces_property_id"] = row_dict[
        #         f"{self.energy_conjugate}_property_id"
        #     ]
        #     row_dict["energy_conjugate_with_forces_column"] = self.energy_conjugate
        return row_dict

    def convert_units(self):
        """
        For each key in :attr:`self.property_map`, convert :attr:`self.edn[key]`
        from its original units to the expected ColabFit-compliant units.
        """

        for key, val in self.property_map.items():
            edn_key = EDN_KEY_MAP.get(key, key)

            units = val["units"]

            split_units = list(
                itertools.chain.from_iterable(
                    [sp.split("/") for sp in units.split("*")]
                )
            )

            if edn_key not in self.instance:
                continue

            val = np.array(self.instance[edn_key]["source-value"], dtype=np.float64)

            val *= float(UNITS[split_units[0]])

            for u in split_units[1:]:
                if units[units.find(u) - 1] == "*":
                    val *= UNITS[u]
                elif units[units.find(u) - 1] == "/":
                    val /= UNITS[u]
                else:
                    raise RuntimeError(
                        "There may be something wrong with the units: {}".format(u)
                    )

            self.instance[edn_key] = {
                "source-value": val.tolist(),
                "source-unit": OPENKIM_PROPERTY_UNITS[key],
            }

            self.property_map[key]["units"] = self.instance[edn_key]["source-unit"]

    def __hash__(self):

        return _hash(self.spark_row, self.unique_identifier_kw)

    def __eq__(self, other):
        """
        Returns False if any of the following conditions are true:

        - Properties point to settings with different calculation methods
        - Properties point to different configurations
        - OpenKIM EDN fields differ in any way

        Note that comparison is performed by hashing
        """

        return hash(self) == hash(other)

    def todict(self):
        return self.instance

    def keys(self):
        """Overloaded dictionary function for getting the keys of :attr:`self.edn`"""
        return self.instance.keys()

    def __setitem__(self, k, v):
        """Overloaded :meth:`dict.__setitem__` for setting the values of
        :attr:`self.edn`"""
        edn_key = EDN_KEY_MAP.get(k, k)

        if k in self.instance:
            self.instance[k] = v
        elif edn_key in self.instance:
            self.instance[edn_key] = v
        else:
            KeyError(f"Field '{k}' not found in Property.edn. Returning None")

    def __getitem__(self, k):
        """Overloaded :meth:`dict.__getitem__` for getting the values of
        :attr:`self.edn`"""
        edn_key = EDN_KEY_MAP.get(k, k)

        if k in self.instance:
            return self.instance[k]
        elif edn_key in self.instance:
            return self.instance[edn_key]
        else:
            warnings.warn(f"Field '{k}' not found in Property.edn. Returning None")

            return None

    def get_data(self, k):
        """
        First checks if :code:`k` is in :code:`self.edn`. If not, checks under
        possible pseudonyms. If nothing exists, returns :code:`np.nan`

        Returns:
            data (np.array or np.nan):
                :attr:`self[k]['source-value']` if :code:`k` is a valid key,
                else :code:`np.nan`.

        """

        edn_key = EDN_KEY_MAP.get(k, k)

        if k in self.instance:
            return np.atleast_1d(self[k]["source-value"])
        elif edn_key in self.instance:
            return np.atleast_1d(self[edn_key]["source-value"])
        else:
            return np.nan

    def __delitem__(self, k):
        edn_key = EDN_KEY_MAP.get(k, k)

        del self.instance[edn_key]

    def __str__(self):
        return "Property(properties={})".format(
            # self.instance["instance-id"],
            [pdef["property-name"] for pdef in self.definitions],
        )

    def __repr__(self):
        return str(self)


class PropertyHashError(Exception):
    pass
