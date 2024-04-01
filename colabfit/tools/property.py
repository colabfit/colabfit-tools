import itertools
import json
import os
import tempfile
import warnings
from collections import defaultdict
from copy import deepcopy
from hashlib import sha512

import numpy as np
import pyspark
from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    FloatType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from colabfit.tools.configuration import AtomicConfiguration

HASH_LENGTH = 12
HASH_SHIFT = 0
# HASH_SHIFT = 2**63

ID_FORMAT_STRING = "{}_{}_{:0d}"

MAX_STRING_LENGTH = 255
STRING_DTYPE_SPECIFIER = f"S{MAX_STRING_LENGTH}"

# _DBASE_COLLECTION           = '_databases'
_DATABASE_NAME = "colabfit_database"
_CONFIGS_COLLECTION = "configurations"
_PROPS_COLLECTION = "property_instances"
_PROPDEFS_COLLECTION = "property_definitions"
_DATAOBJECT_COLLECTION = "data_objects"
_METADATA_COLLECTION = "metadata"
_CONFIGSETS_COLLECTION = "configuration_sets"
_DATASETS_COLLECTION = "datasets"
_AGGREGATED_INFO_COLLECTION = "aggregated_info"

SHORT_ID_STRING_NAME = "colabfit-id"
EXTENDED_ID_STRING_NAME = "extended-id"

ATOMS_NAME_FIELD = "_name"
ATOMS_LABELS_FIELD = "_labels"
ATOMS_LAST_MODIFIED_FIELD = "_last_modified"
ATOMS_CONSTRAINTS_FIELD = "_constraints"

DEFAULT_PROPERTY_NAME = (
    "configuration-nonorthogonal-periodic-3d-cell-fixed-particles-fixed"
)

from ase.units import create_units

UNITS = create_units("2014")

# Make GPa the base unit
UNITS["bar"] = 1e-4  # bar to GPa
UNITS["kilobar"] = 1e-1  # kilobar to GPa
UNITS["pascal"] = 1e-9  # pascal to GPa
UNITS["GPa"] = 1

UNITS["angstrom"] = UNITS["Ang"]

OPENKIM_PROPERTY_UNITS = {
    "energy": "eV",
    "forces": "eV/angstrom",
    "stress": "GPa",
    "unrelaxed-potential-energy": "eV",
    "unrelaxed-potential-forces": "eV/angstrom",
    "unrelaxed-cauchy-stress": "GPa",
}

EDN_KEY_MAP = {
    "energy": "unrelaxed-potential-energy",
    "forces": "unrelaxed-potential-forces",
    "stress": "unrelaxed-cauchy-stress",
    "virial": "unrelaxed-cauchy-stress",
}
import kim_edn
from kim_property import (
    check_instance_optional_key_marked_required_are_present,
    kim_property_create,
)
from kim_property.create import KIM_PROPERTIES
from kim_property.definition import PROPERTY_ID as VALID_KIM_ID
from kim_property.instance import check_property_instances

# from colabfit import STRING_DTYPE_SPECIFIER, UNITS, OPENKIM_PROPERTY_UNITS, EDN_KEY_MAP

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


def _empty_dict_from_schema(schema):
    empty_dict = {}
    for field in schema:
        empty_dict[field.name] = None
    return empty_dict


def md_from_map(pmap_md, config: AtomicConfiguration):
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
    return json.dumps(gathered_fields)


property_object_schema = StructType(
    [
        StructField("id", StringType(), False),
        StructField("hash", StringType(), False),
        StructField("last_modified", TimestampType(), False),
        StructField("configuration_ids", StringType(), True),  # ArrayType(StringType())
        StructField("dataset_ids", StringType(), True),  # ArrayType(StringType())
        StructField("metadata", StringType(), True),
        StructField("chemical_formula_hill", StringType(), True),
        StructField("potential_energy", DoubleType(), True),
        StructField("potential_energy_unit", StringType(), True),
        StructField("potential_energy_per_atom", BooleanType(), True),
        StructField("potential_energy_reference", DoubleType(), True),
        StructField("potential_energy_reference_unit", StringType(), True),
        StructField("potential_energy_property_id", StringType(), True),
        StructField(
            "atomic_forces", StringType(), True
        ),  # ArrayType(ArrayType(DoubleType()))
        StructField("atomic_forces_unit", StringType(), True),
        StructField("atomic_forces_property_id", StringType(), True),
        StructField(
            "cauchy_stress", StringType(), True
        ),  # ArrayType(ArrayType(DoubleType()))
        StructField("cauchy_stress_unit", StringType(), True),
        StructField("cauchy_stress_volume_normalized", BooleanType(), True),
        StructField("cauchy_stress_property_id", StringType(), True),
        StructField("free_energy", DoubleType(), True),
        StructField("free_energy_unit", StringType(), True),
        StructField("free_energy_per_atom", BooleanType(), True),
        StructField("free_energy_reference", DoubleType(), True),
        StructField("free_energy_reference_unit", StringType(), True),
        StructField("free_energy_property_id", StringType(), True),
        StructField("band_gap", DoubleType(), True),
        StructField("band_gap_unit", StringType(), True),
        StructField("band_gap_property_id", StringType(), True),
        StructField("formation_energy", DoubleType(), True),
        StructField("formation_energy_unit", StringType(), True),
        StructField("formation_energy_per_atom", BooleanType(), True),
        StructField("formation_energy_reference", DoubleType(), True),
        StructField("formation_energy_reference_unit", StringType(), True),
        StructField("formation_energy_property_id", StringType(), True),
        StructField("adsorption_energy", DoubleType(), True),
        StructField("adsorption_energy_unit", StringType(), True),
        StructField("adsorption_energy_per_atom", BooleanType(), True),
        StructField("adsorption_energy_reference", DoubleType(), True),
        StructField("adsorption_energy_reference_unit", StringType(), True),
        StructField("adsorption_energy_property_id", StringType(), True),
        StructField("atomization_energy", DoubleType(), True),
        StructField("atomization_energy_unit", StringType(), True),
        StructField("atomization_energy_per_atom", BooleanType(), True),
        StructField("atomization_energy_reference", DoubleType(), True),
        StructField("atomization_energy_reference_unit", StringType(), True),
        StructField("atomization_energy_property_id", StringType(), True),
    ]
)


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
            or isinstance(val, list)
            or isinstance(val, tuple)
            or isinstance(val, dict)
        ):
            row_dict[key] = str(val)
    return row_dict


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
                A field name used to access :attr:`Configuration.info` or :attr:`Configuration.arrays`
            * :attr:`units`:
                A string matching one of the units names in `ase.units <https://wiki.fysik.dtu.dk/ase/ase/units.html>`_.
                These units will be used to convert the given units to eV,
                Angstrom, a.m.u., Kelvin, ... For compound units (e.g. "eV/Ang"), the string will be split on
                '*' and '/'. The raw data will be multiplied by the first unit
                and anything preceded by a '*'. It will be divided by anything
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
        self.instance = instance
        self.instance_json = json.dumps(instance)
        self.definitions = definitions

        if property_map is not None:
            self.property_map = dict(property_map)
        else:
            self.property_map = {}
        if metadata is not None:
            self.metadata = metadata

        if convert_units:
            self.convert_units()

        self._hash = hash(self)

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
        cls, definitions, configuration, property_map, convert_units=False
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
        print(definitions)
        pdef_dict = {pdef["property-name"]: pdef for pdef in definitions}
        instances = {
            pdef_name: cls.get_kim_instance(pdef)
            for pdef_name, pdef in pdef_dict.items()
        }

        # props_dict = defaultdict(list)
        props_dict = {}
        for pname, pmap_list in property_map.items():
            instance = instances.get(pname, None)
            if instance is None:
                raise PropertyParsingError(f"Property {pname} not found in definitions")
            else:
                instance = instance.copy()
            for pmap_i, pmap in enumerate(pmap_list):
                pmap_copy = dict(pmap)
                if "_metadata" in pmap_copy:
                    pi_md = pmap_copy.pop("_metadata")
                for key, val in pmap_copy.items():
                    print(key, ":", val)
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
                    elif isinstance(data, (str, bool, int, float)):
                        pass
                    elif np.issubdtype(data.dtype, np.integer):
                        data = int(data)
                    elif np.issubdtype(data.dtype, np.float):
                        data = float(data)

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

        # return props_dict
        return cls(
            definitions=definitions,
            property_map=property_map,
            instance=props_dict,
            metadata=pi_md,
            convert_units=convert_units,
        )

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
        """
        Hashes the Property by hashing its EDN.
        """
        _hash = sha512()
        for p_name, p_dict in self.instance.items():
            for key, val in p_dict.items():
                if key in _ignored_fields:
                    continue
                try:
                    hashval = np.round_(
                        np.array(val["source-value"]), decimals=12
                    ).data.tobytes()
                except (TypeError, KeyError):
                    try:
                        hashval = np.array(
                            val["source-value"], dtype=STRING_DTYPE_SPECIFIER
                        ).data.tobytes()
                    except (TypeError, KeyError):
                        try:
                            hashval = np.array(
                                val, dtype=STRING_DTYPE_SPECIFIER
                            ).data.tobytes()
                        except Exception as e:
                            raise PropertyHashError(
                                "Could not hash key {}: {}. Error type {}".format(
                                    key, val, type(e)
                                )
                            )

                _hash.update(hashval)
                # What if values are identical but are added in different units? Should these hash to unique PIs?
                if "source-unit" in val:
                    _hash.update(str(val["source-unit"]).encode("utf-8"))

        return int(_hash.hexdigest(), 16)

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
        """Overloaded :meth:`dict.__setitem__` for setting the values of :attr:`self.edn`"""
        edn_key = EDN_KEY_MAP.get(k, k)

        if k in self.instance:
            self.instance[k] = v
        elif edn_key in self.instance:
            self.instance[edn_key] = v
        else:
            KeyError(f"Field '{k}' not found in Property.edn. Returning None")

    def __getitem__(self, k):
        """Overloaded :meth:`dict.__getitem__` for getting the values of :attr:`self.edn`"""
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


# Eric->Do we need to do this?
def update_edn_with_conf(edn, conf):
    edn["species"] = {"source-value": conf.get_chemical_symbols()}

    lattice = np.array(conf.get_cell()).tolist()

    edn["unrelaxed-periodic-cell-vector-1"] = {
        "source-value": lattice[0],
        "source-unit": "angstrom",
    }

    edn["unrelaxed-periodic-cell-vector-2"] = {
        "source-value": lattice[1],
        "source-unit": "angstrom",
    }

    edn["unrelaxed-periodic-cell-vector-3"] = {
        "source-value": lattice[2],
        "source-unit": "angstrom",
    }

    edn["unrelaxed-configuration-positions"] = {
        "source-value": conf.positions.tolist(),
        "source-unit": "angstrom",
    }


class PropertyHashError(Exception):
    pass
