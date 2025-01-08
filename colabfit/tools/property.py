import datetime
import itertools
import json
import os
import tempfile
import warnings
from collections import namedtuple
from copy import deepcopy

import dateutil.parser
import kim_edn
import numpy as np
from ase.units import create_units
from kim_property import (
    check_instance_optional_key_marked_required_are_present,
    kim_property_create,
)
from kim_property.definition import PROPERTY_ID as VALID_KIM_ID

from colabfit.tools.configuration import AtomicConfiguration
from colabfit.tools.schema import property_object_schema
from colabfit.tools.utilities import (
    _empty_dict_from_schema,
    _hash,
    _parse_unstructured_metadata,
)

from kim_property.create import KIM_PROPERTIES

EDN_KEY_MAP = {
    "energy": "unrelaxed-potential-energy",
    "forces": "unrelaxed-potential-forces",
    "stress": "unrelaxed-cauchy-stress",
    "virial": "unrelaxed-cauchy-stress",
}

UNITS = create_units("2014")

UNITS["angstrom"] = UNITS["Ang"]
UNITS["bohr"] = UNITS["Bohr"]
UNITS["hartree"] = UNITS["Hartree"]
UNITS["rydberg"] = UNITS["Rydberg"]
UNITS["debye"] = UNITS["Debye"]
UNITS["kbar"] = UNITS["bar"] * 1000

prop_info = namedtuple("prop_info", ["key", "unit", "dtype"])
energy_info = prop_info("energy", "eV", float)
force_info = prop_info("forces", "eV/angstrom", list)
stress_info = prop_info("stress", "eV/angstrom^3", list)
MAIN_KEY_MAP = {
    "energy": energy_info,
    "atomic-forces": force_info,
    "cauchy-stress": stress_info,
    "atomization-energy": energy_info,
    "adsorption-energy": energy_info,
    "formation-energy": energy_info,
    "band-gap": energy_info,
}

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
    "multiplicity",
    "metadata_path",
    "metadata_size",
]


def energy_to_schema(prop_name, en_prop: dict):
    if en_prop.get("energy") is None:
        return {}
    new_name = prop_name.replace("-", "_")
    en_dict = {
        f"{new_name}": en_prop["energy"]["source-value"],
        f"{new_name}_unit": en_prop["energy"]["source-unit"],
        f"{new_name}_per_atom": en_prop["per-atom"]["source-value"],
    }
    ref_en_dict = en_prop.get("reference-energy")
    if ref_en_dict is not None:
        ref_en = ref_en_dict["source-value"]
        ref_unit = ref_en_dict["source-unit"]
        en_dict[f"{new_name}_reference"] = ref_en
        en_dict[f"{new_name}_reference_unit"] = ref_unit

    return en_dict


def atomic_forces_to_schema(af_prop: dict):
    if af_prop.get("forces") is None:
        return {}
    af_dict = {
        "atomic_forces_00": af_prop["forces"]["source-value"],
        "atomic_forces_unit": af_prop["forces"]["source-unit"],
    }
    return af_dict


def cauchy_stress_to_schema(cs_prop: dict):
    if cs_prop.get("stress") is None:
        return {}
    cs_dict = {
        "cauchy_stress": cs_prop["stress"]["source-value"],
        "cauchy_stress_unit": cs_prop["stress"]["source-unit"],
        "cauchy_stress_volume_normalized": cs_prop["volume-normalized"]["source-value"],
    }
    return cs_dict


def band_gap_to_schema(bg_prop: dict):
    if bg_prop.get("energy") is None:
        return {}
    bg_dict = {
        "electronic_band_gap": bg_prop["energy"]["source-value"],
        "electronic_band_gap_unit": bg_prop["energy"]["source-unit"],
        "electronic_band_gap_type": bg_prop.get("type", {"source-value": "direct"})[
            "source-value"
        ],
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
                continue  # No keys are required; ignored if missing
        else:
            continue  # No keys are required; ignored if missing

        if "units" in pmap_md[md_field]:
            gathered_fields[md_field] = {
                "source-value": v,
                "source-unit": pmap_md[md_field]["units"],
            }
        else:
            gathered_fields[md_field] = {"source-value": v}
    method = gathered_fields.pop("method", None)
    software = gathered_fields.pop("software", None)
    if "property_keys" not in gathered_fields:
        raise RuntimeError(
            "'property_keys' must be provided in the property map. If defined, check that property_keys is formatted as {'property_keys': {'value': {'key1 : value1}}}"  # noqa E501
        )
    if method is not None:
        method = method["source-value"]
    if software is not None:
        software = software["source-value"]
    return gathered_fields, method, software


class PropertyParsingError(Exception):
    pass


class InvalidPropertyDefinition(Exception):
    pass


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
        standardize_energy=True,
        nsites=None,
        dataset_id=None,
    ):
        """
        Args:
            definitions (list(dict)):
                KIM Property Definitions
            instance (dict):
                A dictionary defining an OpenKIM Property Instance
            property_map (dict):
                A property map as described in the Property attributes section.
            standardize_energy (bool):
                If True, converts units to those expected by ColabFit. Default
                is True
        """
        # self.unique_identifier_kw = [
        #     k
        #     for k in property_object_schema.fieldNames()
        #     if k not in _hash_ignored_fields
        # ]
        self.unique_identifier_kw = [
            "adsorption_energy",
            "atomic_forces_00",
            "atomization_energy",
            "cauchy_stress",
            "cauchy_stress_volume_normalized",
            "chemical_formula_hill",
            "configuration_id",
            "dataset_id",
            "electronic_band_gap",
            "electronic_band_gap_type",
            "energy",
            "formation_energy",
            "metadata_id",
            "method",
            "software",
        ]
        self.unique_identifier_kw.extend(
            [f"atomic_forces_{i:02d}" for i in range(1, 20)]
        )
        self.instance = instance
        self.definitions = definitions
        self.nsites = nsites
        if property_map is not None:
            self.property_map = dict(property_map)
        else:
            self.property_map = {}
        self.metadata = _parse_unstructured_metadata(metadata)
        self.chemical_formula_hill = instance.pop("chemical_formula_hill")
        if standardize_energy:
            self.standardize_energy()
        if dataset_id is not None:
            self.dataset_id = dataset_id
        self.spark_row = self.to_spark_row()
        self._hash = _hash(self.spark_row, self.unique_identifier_kw, False)
        self.spark_row["hash"] = str(self._hash)
        self._id = f"PO_{self._hash}"
        if len(self._id) > 28:
            self._id = self._id[:28]
        self.spark_row["id"] = self._id

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
            fields.append(key)
        self._property_fields = fields

    @property
    def property_fields(self):
        return self._property_fields

    @classmethod
    def get_kim_instance(
        cls,
        definition,
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
        # global KIM_PROPERTIES
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
        return instance

    @classmethod
    def from_definition(
        cls,
        definitions,
        configuration,
        property_map,
        standardize_energy=True,
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
                p_info = MAIN_KEY_MAP.get(pname, None)
                if p_info is None:
                    print(f"property {pname} not found in MAIN_KEY_MAP")
                    continue
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
                if p_info.key not in instance:
                    print(f"Property {p_info.key} not found in {pname}")
                    pdef_dict.pop(pname)
                    continue
                # hack to get around OpenKIM requiring the property-name be a dict
                prop_name_tmp = pdef_dict[pname].pop("property-name")

                check_instance_optional_key_marked_required_are_present(
                    instance, pdef_dict[pname]
                )
                pdef_dict[pname]["property-name"] = prop_name_tmp
                props_dict[pname] = {k: v for k, v in instance.items()}
        props_dict["chemical_formula_hill"] = configuration.get_chemical_formula()
        props_dict["configuration_id"] = configuration.id

        return cls(
            definitions=definitions,
            property_map=property_map,
            instance=props_dict,
            metadata=pi_md,
            dataset_id=configuration.dataset_id,
            standardize_energy=standardize_energy,
            nsites=configuration.spark_row["nsites"],
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
        row_dict["dataset_id"] = self.dataset_id
        return row_dict

    def standardize_energy(self):
        """
        For each key in :attr:`self.property_map`, convert :attr:`self.edn[key]`
        from its original units to the expected ColabFit-compliant units.
        """
        for prop_name, prop_dict in self.instance.items():
            if prop_name not in MAIN_KEY_MAP.keys():
                continue
            p_info = MAIN_KEY_MAP[prop_name]
            if p_info.key not in prop_dict:
                print(f"Property {p_info.key} not found in {prop_name}")
                continue
            units = prop_dict[p_info.key]["source-unit"]
            if p_info.dtype == list:
                prop_val = np.array(
                    prop_dict[p_info.key]["source-value"], dtype=np.float64
                )
            else:
                prop_val = prop_dict[p_info.key]["source-value"]
            if "reference-energy" in prop_dict:
                if prop_dict["reference-energy"]["source-unit"] != units:
                    raise RuntimeError(
                        "Units of the reference energy and energy must be the same"
                    )
                else:
                    prop_val += prop_dict["reference-energy"]["source-value"]

            if "per-atom" in prop_dict:
                if prop_dict["per-atom"]["source-value"] is True:
                    if self.nsites is None:
                        raise RuntimeError("nsites must be provided to convert per-atom")
                    prop_val *= self.nsites

            if units != p_info.unit:
                split_units = list(
                    itertools.chain.from_iterable(
                        [sp.split("/") for sp in units.split("*")]
                    )
                )
                powers = []
                for i, sp_un in enumerate(split_units):
                    if "^" in sp_un:
                        split = sp_un.split("^")
                        powers.append(int(split[1]))
                        split_units[i] = split[0]
                    else:
                        powers.append(1)
                if powers[0] != 1:
                    prop_val *= np.power(float(UNITS[split_units[0]]), powers[0])
                else:
                    prop_val *= float(UNITS[split_units[0]])
                for u, power in zip(split_units[1:], powers[1:]):
                    un = UNITS[u]
                    if power != 1:
                        un = np.power(un, power)
                    if units[units.find(u) - 1] == "*":
                        prop_val *= un
                    elif units[units.find(u) - 1] == "/":
                        prop_val /= un
                    else:
                        raise RuntimeError(
                            f"There may be something wrong with the units: {u}"
                        )
            if p_info.dtype == list:
                prop_val = prop_val.tolist()
            self.instance[prop_name][p_info.key] = {
                "source-value": prop_val,
                "source-unit": p_info.unit,
            }

    def __hash__(self):

        return _hash(
            self.spark_row,
            sorted(self.unique_identifier_kw),
        )

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
            [pdef["property-name"] for pdef in self.definitions],
        )

    def __repr__(self):
        return str(self)


property_info = namedtuple(
    "property_info",
    ["property_name", "field", "units", "original_file_key", "additional"],
)


class PropertyMap:
    """
    A class to store and check metadata and key/field mappings for Property objects.

    Initialize with list of property definitions, then add properties with set_property
    or set_properties.
    Add metadata fields with set_metadata_field: 'software', 'method', 'input' required.
    Use get_property_map to validate and return a complete property map to use with
    Property objects.
    Attributes:
    -----------
    property_definitions : list
        List of property definitions.
    properties : dict
        Property map of property names to keys/fields and units.
    _metadata : dict
        Metadata information (equivalent to PI_METADATA from prior implementation).
    Methods:
    --------
    set_metadata_field(key: str, value: any, dynamic=False):
        Sets a metadata field with the given key and value. If dynamic is True, metadata
        will be populated from a field in the AtomicConfiguration object
        (ie: atom.info[field]). Otherwise a constant value. Default is False.
    get_metadata():
        Validates and returns a dictionary of metadata.
    set_property(property_name: str, field: str, units: str, original_file_key: str,
                                additional: list[tuple] = []):
        Sets a property with the details from dict or property_info namedtuple.
    set_properties(properties: list[dict | property_info]):
        Sets multiple properties from a list of dictionaries or property_info objects.
    get_property(property_name: str):
        Returns property details by name of property.
    validate_metadata():
        Ensure required fields are set in metadata.
    validate_properties():
        Ensure required fields and units are set in properties.
    get_property_map():
        Validates, returns the complete property map including metadata and properties.
    """

    def __init__(self, property_definitions: list):
        self.property_definitions = {p["property-name"]: p for p in property_definitions}
        self._metadata = {
            "software": {"value": None, "required": True},
            "method": {"value": None, "required": True},
            "input": {"value": None, "required": True},
            "property_keys": {
                "value": {
                    p["property-name"]: None for p in self.property_definitions.values()
                },
            },
        }
        self.properties = {}
        for name in self.property_definitions:
            main_key = MAIN_KEY_MAP[name].key
            self.properties[name] = {main_key: {"field": None, "units": None}}

    def set_metadata_field(self, key: str, value, dynamic=False):
        if dynamic:
            self._metadata[key] = {"field": value}
        else:
            self._metadata[key] = {"value": value}

    def get_metadata(self):
        self.validate_metadata()
        return {
            k: v for k, v in self._metadata.items() if (v.get("value") or v.get("field"))
        }

    def set_property(
        self,
        property_name: str,
        field: str,
        units: str,
        original_file_key: str,
        additional: list[tuple] = [],
    ):
        if property_name not in self.properties:
            raise KeyError(f"Property not included in PropertyMap: {property_name}")
        self.properties[property_name][MAIN_KEY_MAP[property_name].key] = {
            "field": field,
            "units": units,
        }
        self._metadata["property_keys"]["value"][property_name] = original_file_key
        for add in additional:
            key, value = add
            if key in self.property_definitions[property_name]:
                self.properties[property_name][key] = value
            else:
                raise KeyError(f"Key '{key}' not found in property '{property_name}'")

    def set_properties(self, properties: list[dict | property_info]):
        for prop in properties:
            if isinstance(prop, dict):
                prop_name = prop["property_name"]
                field = prop["field"]
                units = prop["units"]
                original_file_key = prop["original_file_key"]
                additional = prop.get("additional", [])
                self.set_property(prop_name, field, units, original_file_key, additional)
            elif isinstance(prop, property_info):
                prop_name = prop.property_name
                field = prop.field
                units = prop.units
                original_file_key = prop.original_file_key
                additional = prop.additional
                if additional is None:
                    additional = []
                self.set_property(prop_name, field, units, original_file_key, additional)
        self.validate_properties()

    def get_property(self, property_name: str):
        if property_name not in self.properties:
            raise KeyError(f"Property not included in PropertyMap: {property_name}")
        return self.properties[property_name]

    def validate_metadata(self):
        for key, value in self._metadata.items():
            if (
                value.get("required")
                and value.get("value") is None
                and value.get("field") is None
            ):
                raise ValueError(f"Metadata key '{key}' is required but not set.")
        for prop_name in self.properties.keys():
            if self._metadata["property_keys"]["value"][prop_name] is None:
                raise ValueError(
                    f"Metadata must have 'original_file_key' set for each property. None set for '{prop_name}'."
                )

    def validate_properties(self):
        for prop_name, prop in self.properties.items():
            main_key = prop.get(MAIN_KEY_MAP[prop_name].key)
            if (
                main_key is None
                or main_key["field"] is None
                or main_key["units"] is None
            ):
                raise ValueError(
                    f"Property '{prop_name}' must have 'field' and 'units' set."
                )
            check_dict = {
                k: v
                for k, v in self.property_definitions[prop_name].items()
                if k
                not in [
                    "property-name",
                    "property-id",
                    "property-title",
                    "property-description",
                ]
            }
            for key, val in check_dict.items():
                if isinstance(prop.get(key), dict):
                    prop_view = prop[key]
                else:
                    prop_view = prop
                if (
                    val.get("required")
                    and prop_view.get("value") is None
                    and prop_view.get("field") is None
                ):
                    raise ValueError(f"Property '{prop_name}' must have '{key}' set.")
                elif not val.get("required") and prop_view.get(key) is None:
                    continue
                elif val.get("has-unit") and prop_view.get("units") is None:
                    raise ValueError(f"Property '{prop_name}' must have 'units' set.")
                elif val.get("has-unit") is False and prop_view.get("units") is not None:
                    raise ValueError(
                        f"Property '{prop_name}' must have key {key}: 'units' set to None."
                    )
            if self._metadata["property_keys"]["value"][prop_name] is None:
                raise ValueError(
                    f"Property '{prop_name}' must have 'original_file_key' set."
                )

    def get_property_map(self):
        self.validate_metadata()
        self.validate_properties()
        return {
            "_metadata": self.get_metadata(),
            **{k: [v] for k, v in self.properties.items()},
        }


class PropertyHashError(Exception):
    pass
