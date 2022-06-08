import os
import json
import tempfile
import warnings
import itertools
import numpy as np
from copy import deepcopy
from hashlib import sha512

import kim_edn
from kim_property.instance import check_property_instances
from kim_property import (
    kim_property_create,
    check_instance_optional_key_marked_required_are_present
)

from kim_property.definition import PROPERTY_ID as VALID_KIM_ID
from kim_property.create import KIM_PROPERTIES

from colabfit import (
    HASH_LENGTH, HASH_SHIFT,
    DEFAULT_PROPERTY_NAME, STRING_DTYPE_SPECIFIER, UNITS,
    OPENKIM_PROPERTY_UNITS, EDN_KEY_MAP
)

# These are fields that are related to the geometry of the atomic structure
# or the OpenKIM Property Definition and shouldn't be used for equality checks
_ignored_fields = [
    #'property-id',         # removed so that if two different properties have identical values they will still hash differently
    'property-title',
    'property-description',
    'instance-id',
    'species',
    'unrelaxed-configuration-positions',
    'unrelaxed-periodic-cell-vector-1',
    'unrelaxed-periodic-cell-vector-2',
    'unrelaxed-periodic-cell-vector-3',
]


class PropertyParsingError(Exception):
    pass

class MissingPropertyFieldWarning(Warning):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

class InvalidPropertyDefinition(Exception):
    pass

class Property(dict):
    """
    A Property is used to store the results of some kind of calculation or
    experiment, and should be mapped to an `OpenKIM Property Definition <https://openkim.org/doc/schema/properties-framework/>`_. Best
    practice is for the Property to also point to one or more
    PropertySettings objects that fully define the conditions under which the
    Property was obtained.

    Attributes:

        definition (dict):
            A KIM Property Definition

        instance (dict):
            A dictionary defining an OpenKIM Property Instance.
            For more details, see the `OpenKIM Property Framework <https://openkim.org/doc/schema/properties-framework/>`_
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

        configuration_ids (list):
            A list of IDs of Configuration objects.

        settings (PropertySettings):
            A :class:`~colabfit.tools.property_settings.PropertySettings` object
            defining the conditions under which the propoerty was obtained. This
            is allowed to be None, but it is highly recommended that it be
            provided.
    """

    _observers = []

    def __init__(
        self,
        definition,
        instance,
        configuration_ids,
        property_map=None,
        settings=None,
        instance_id=1,
        convert_units=False
        ):
        """
        Args:

            definition (dict):
                A KIM Property Definition

            instance (dict):
                A dictionary defining an OpenKIM Property Instance

            configuration_ids (list):
                A list of IDs of ColabFit Configuration objects

            settings (PropertySettings):
                A `colabfit.property.PropertySettings` objects specifying how to
                compute the property.

            property_map (dict):
                A property map as described in the Property attributes section.

            instance_id (int):
                A positive non-zero integer

            convert_units (bool):
                If True, converts units to those expected by ColabFit. Default
                is False
        """

        global KIM_PROPERTIES

        load_from_existing = False

        if isinstance(definition, dict):
            dummy_dict = deepcopy(definition)

            # Spoof if necessary
            if VALID_KIM_ID.match(dummy_dict['property-id']) is None:
                # Invalid ID. Try spoofing it
                dummy_dict['property-id'] = 'tag:@,0000-00-00:property/'
                dummy_dict['property-id'] += definition['property-id']
                # warnings.warn(f"Invalid KIM property-id; Temporarily renaming to {dummy_dict['property-id']}")

            load_from_existing = dummy_dict['property-id'] in KIM_PROPERTIES
            definition_name = dummy_dict['property-id']

        elif isinstance(definition, str):
            if os.path.isfile(definition):
                dummy_dict = kim_edn.load(definition)

                # Check if you need to spoof the property ID to trick OpenKIM
                if VALID_KIM_ID.match(dummy_dict['property-id']) is None:
                    # Invalid ID. Try spoofing it
                    dummy_dict['property-id'] = 'tag:@,0000-00-00:property/' + dummy_dict['property-id']
                    warnings.warn(f"Invalid KIM property-id; Temporarily renaming to {dummy_dict['property-id']}")

                load_from_existing = dummy_dict['property-id'] in KIM_PROPERTIES
                definition_name = dummy_dict['property-id']

            else:
                # Then this has to be an existing (or added) KIM Property Definition
                load_from_existing = True

                # It may have been added previously, but spoofed
                if VALID_KIM_ID.match(definition) is None:
                    # Invalid ID. Try spoofing it
                    definition_name = 'tag:@,0000-00-00:property/' + definition
        else:
            raise InvalidPropertyDefinition(
                "Property definition must either be a dictionary or a path to "\
                "an EDN file"
            )

        self.definition = definition
        self.name = self.definition['property-id']

        # if load_from_existing:
        #     instance = kim_edn.loads(kim_property_create(
        #         instance_id=instance_id,
        #         property_name=definition,
        #     ))[0]
        # else:
        #     with tempfile.NamedTemporaryFile('w') as tmp:
        #         tmp.write(json.dumps(dummy_dict))
        #         tmp.flush()

        #         instance = kim_edn.loads(kim_property_create(
        #             instance_id=instance_id,
        #             property_name=tmp.name,
        #         ))[0]

        # if instance is None:
        #     self.instance = kim_edn.loads(kim_property_create(
        #         instance_id=instance_id,
        #         property_name=self.name
        #     ))[0]
        # else:
        #     # if name != PROPERTY_ID_TO_PROPERTY_NAME[edn['property-id']]:
        #     #     raise RuntimeError(
        #     #         "`name` does not match `edn['property_name']`"
        #     #     )

        self.instance = instance
        self.instance['property-id'] = definition_name

        # Hack to avoid the fact that "property-name" has to be a dictionary
        # in order for OpenKIM's check_property_definition to work
        tmp = self.definition['property-name']
        del self.definition['property-name']

        check_instance_optional_key_marked_required_are_present(
            self.instance,
            self.definition,
            # KIM_PROPERTIES[self.instance['property-id']]
        )

        self.definition['property-name'] = tmp

        if load_from_existing:
            check_property_instances(
                self.instance,
                fp_path=KIM_PROPERTIES
            )
        else:
            with tempfile.NamedTemporaryFile('w') as tmp:
                tmp.write(json.dumps(dummy_dict))
                tmp.flush()

                check_property_instances(
                    self.instance,
                    fp=tmp.name
                )

        if property_map is not None:
            self.property_map = dict(property_map)
        else:
            self.property_map = {}

        # Delete any un-used properties from the property_map
        delkeys = []
        for key in self.property_map:
            edn_key = EDN_KEY_MAP.get(key, key)

            if edn_key not in self.instance:
                delkeys.append(key)

        for key in delkeys:
            del self.property_map[key]

        if convert_units:
            self.convert_units()

        if len(configuration_ids) < 1:
            raise RuntimeError(
                "`Property.configuration_ids` must contain at least 1 entry"
            )
        # Eric-> Do we ever use this
        self.configuration_ids = configuration_ids

        # Add settings
        if settings is not None:
            self.settings = settings
        else:
            self.settings = []

        self._hash = hash(self)

    @property
    def instance(self):
        return self._instance


    @instance.setter
    def instance(self, edn):
        self._instance = edn

        fields = []
        for key in self._instance:
            if key in _ignored_fields: continue

            # fields.append(EDN_KEY_MAP.get(key, key))
            fields.append(key)

        self._property_fields = fields


    @property
    def property_fields(self):
        return self._property_fields


    @classmethod
    def from_definition(
        cls, definition, configuration, property_map,
        settings=None, instance_id=1, convert_units=False
    ):

        """
        A function for constructing a Property given a configuration, a property
        definition, and a property map.


        Args:

            definition (dict):
                A valid KIM Property Definition

            configuration (Configuration):
                A Configuration objects from which to extract the property data

            property_map (dict):
                A property map as described in the Property attributes section.

        """

        global KIM_PROPERTIES

        load_from_existing = False

        if isinstance(definition, dict):

            # TODO: this is probably slowing things down a bit
            dummy_dict = deepcopy(definition)

            # Spoof if necessary
            if VALID_KIM_ID.match(dummy_dict['property-id']) is None:
                # Invalid ID. Try spoofing it
                dummy_dict['property-id'] = 'tag:@,0000-00-00:property/'
                dummy_dict['property-id'] += definition['property-id']
                # warnings.warn(f"Invalid KIM property-id; Temporarily renaming to {dummy_dict['property-id']}")

            load_from_existing = dummy_dict['property-id'] in KIM_PROPERTIES
            definition_name = dummy_dict['property-id']

        elif isinstance(definition, str):
            if os.path.isfile(definition):
                dummy_dict = kim_edn.load(definition)

                # Check if you need to spoof the property ID to trick OpenKIM
                if VALID_KIM_ID.match(dummy_dict['property-id']) is None:
                    # Invalid ID. Try spoofing it
                    dummy_dict['property-id'] = 'tag:@,0000-00-00:property/' + dummy_dict['property-id']
                    warnings.warn(f"Invalid KIM property-id; Temporarily renaming to {dummy_dict['property-id']}")

                load_from_existing = dummy_dict['property-id'] in KIM_PROPERTIES
                definition_name = dummy_dict['property-id']

            else:
                # Then this has to be an existing (or added) KIM Property Definition
                load_from_existing = True

                # It may have been added previously, but spoofed
                if VALID_KIM_ID.match(definition) is None:
                    # Invalid ID. Try spoofing it
                    definition_name = 'tag:@,0000-00-00:property/' + definition
        else:
            raise InvalidPropertyDefinition(
                "Property definition must either be a dictionary or a path to "\
                "an EDN file"
            )

        if load_from_existing:
            instance = kim_edn.loads(kim_property_create(
                instance_id=instance_id,
                property_name=definition_name,
            ))[0]
        else:
            with tempfile.NamedTemporaryFile('w') as tmp:
                # Hack to avoid the fact that "property-name" has to be a dictionary
                # in order for OpenKIM's check_property_definition to work

                if 'property-name' in dummy_dict:
                    del dummy_dict['property-name']

                tmp.write(json.dumps(dummy_dict))
                tmp.flush()

                instance = kim_edn.loads(kim_property_create(
                    instance_id=instance_id,
                    property_name=tmp.name,
                ))[0]

        update_edn_with_conf(instance, configuration)

        for key, val in property_map.items():
            if val['field'] in configuration.info:
                data = configuration.info[val['field']]
            elif val['field'] in configuration.arrays:
                data = configuration.arrays[val['field']]
            else:
                # Key not found on configurations. Will be checked later
                continue
                # raise MissingPropertyFieldWarning(
                #     f"Key '{key}' not found during Property.from_defintion()"
                # )

            if isinstance(data, (np.ndarray, list)):
                data = np.atleast_1d(data).tolist()
            elif isinstance(data, (str, bool, int, float)):
                pass
            elif np.issubdtype(data.dtype, np.integer):
                data = int(data)
            elif np.issubdtype(data.dtype, np.float):
                data = float(data)

            instance[key] = {
                'source-value': data,
            }

            if (val['units'] != 'None') and (val['units'] is not None):
                instance[key]['source-unit'] = val['units']

        return cls(
            definition=definition,
            configuration_ids=[configuration],
            property_map=property_map,
            settings=settings,
            instance=instance,
            convert_units=convert_units,
        )


    # @classmethod
    # def Default(
    #     cls, conf, property_map, settings=None, instance_id=1,
    #     convert_units=False
    #     ):
    #     f"""
    #     Constructs a default property for storing common properties like energy,
    #     force, stress, ...

    #     Uses {DEFAULT_PROPERTY_NAME} as the Property Definition.

    #     Assumes that the properties, if provided, are stored in the following
    #     ways:
    #         energy: `conf.info[property_map['energy']]`
    #         forces: `conf.arrays[property_map['forces']]`
    #         stress: `conf.info[property_map['stress']]`
    #         ...

    #     Note that some fields have been given pseudonyms:

    #     (key=pseudonym, value=original name)
    #     {EDN_KEY_MAP}
    #     """

    #     edn = kim_edn.loads(kim_property_create(
    #         instance_id=instance_id, property_name=DEFAULT_PROPERTY_NAME
    #     ))[0]

    #     update_edn_with_conf(edn, conf)

    #     for key, val in property_map.items():
    #         if (val['field'] not in conf.info) and (val['field'] not in conf.arrays):
    #             field_not_found_message = 'Key "{}" not found in atoms.info '\
    #                 'or atoms.arrays'.format(key)

    #             warnings.warn(
    #                 field_not_found_message,
    #                 category=MissingPropertyFieldWarning
    #             )

    #             continue

    #         # TODO: WHY is this code here??
    #         if val['field'] in conf.info:
    #             data = conf.info[val['field']]

    #             conf.info[val['field']] = data
    #         elif val['field'] in conf.arrays:
    #             data = conf.arrays[val['field']]

    #             conf.arrays[val['field']] = data
    #         else:
    #             # Key not found on configurations. Don't throw error.
    #             pass

    #         data = np.atleast_1d(data)
    #         if data.size == 1:
    #             data = float(data)
    #         else:
    #             data = data.tolist()

    #         edn_key = EDN_KEY_MAP.get(key, key)

    #         if edn_key == 'unrelaxed-potential-energy':
    #             edn[edn_key] = {
    #                 'source-value': data,
    #                 'source-unit': val['units'],
    #             }
    #         if edn_key == 'unrelaxed-potential-forces':
    #             edn[edn_key] = {
    #                 'source-value': data,
    #                 'source-unit': val['units']
    #             }
    #         if edn_key == 'unrelaxed-cauchy-stress':
    #             data = np.array(data)
    #             if np.prod(data.shape) == 9:
    #                 data = data.reshape((3, 3))
    #                 data = np.array([
    #                     data[0, 0],
    #                     data[1, 1],
    #                     data[2, 2],
    #                     data[1, 2],
    #                     data[0, 2],
    #                     data[0, 1],
    #                 ]).tolist()

    #             edn[edn_key] = {
    #                 'source-value': data,
    #                 'source-unit': val['units']
    #             }

    #     return cls(
    #         name=DEFAULT_PROPERTY_NAME,
    #         configurations=[str(hash(conf))],
    #         property_map=property_map,
    #         settings=settings,
    #         edn=edn,
    #         convert_units=convert_units,
    #     )


    def convert_units(self):
        """
        For each key in :attr:`self.property_map`, convert :attr:`self.edn[key]`
        from its original units to the expected ColabFit-compliant units.
        """

        for key, val in self.property_map.items():
            edn_key = EDN_KEY_MAP.get(key, key)

            units = val['units']

            split_units = list(itertools.chain.from_iterable([
                sp.split('/') for sp in units.split('*')
            ]))

            if edn_key not in self.instance: continue

            val = np.array(self.instance[edn_key]['source-value'], dtype=np.float64)

            val *= float(UNITS[split_units[0]])

            for u in split_units[1:]:
                if units[units.find(u)-1] == '*':
                    val *= UNITS[u]
                elif units[units.find(u)-1] == '/':
                    val /= UNITS[u]
                else:
                    raise RuntimeError(
                        "There may be something wrong with the units: "\
                            "{}".format(u)
                    )

            self.instance[edn_key] = {
                'source-value': val.tolist(),
                'source-unit': OPENKIM_PROPERTY_UNITS[key]
            }

            self.property_map[key]['units'] = self.instance[edn_key]['source-unit']


    def __hash__(self):
        """
        Hashes the Property by hashing its EDN.
        #Change below
        Note that the property hash also
        depends upon the hashes of the linked configurations; this is to handle
        the case where two properties happen to be the same even though their
        underlying configurations are different.
        """

        _hash = sha512()
        for key, val in self.instance.items():
            if key in _ignored_fields: continue
            try:
                hashval =  np.round_(
                    np.array(val['source-value']), decimals=12
                ).data.tobytes()
            except:
                try:
                    hashval = np.array(
                        val['source-value'], dtype=STRING_DTYPE_SPECIFIER
                    ).data.tobytes()
                except:
                    try:
                        hashval = np.array(val,dtype=STRING_DTYPE_SPECIFIER).data.tobytes()
                    except:
                        raise PropertyHashError(
                            "Could not hash key {}: {}".format(key, val)
                        )

            _hash.update(hashval)
            # What if values are identical but are added in different units? Should these hash to unique PIs?
            if 'source-unit' in val:
                _hash.update(str(val['source-unit']).encode('utf-8'))
        # Don't hash cids
        #for cid in self.configuration_ids:
        #    _hash.update(cid.encode('utf-8'))
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
            KeyError(
                f"Field '{k}' not found in Property.edn. Returning None"
            )


    def __getitem__(self, k):
        """Overloaded :meth:`dict.__getitem__` for getting the values of :attr:`self.edn`"""
        edn_key = EDN_KEY_MAP.get(k, k)

        if k in self.instance:
            return self.instance[k]
        elif edn_key in self.instance:
            return self.instance[edn_key]
        else:
            warnings.warn(
                f"Field '{k}' not found in Property.edn. Returning None"
            )

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
            return np.atleast_1d(self[k]['source-value'])
        elif edn_key in self.instance:
            return np.atleast_1d(self[edn_key]['source-value'])
        else:
            return np.nan


    def __delitem__(self, k):
        edn_key = EDN_KEY_MAP.get(k, k)

        del self.instance[edn_key]


    def __str__(self):
        return "Property(instance_id={}, name='{}')".format(
            self.instance['instance-id'],
            # PROPERTY_ID_TO_PROPERTY_NAME[self.edn['property-id']]
            self.name
        )


    def __repr__(self):
        return str(self)

# Eric->Do we need to do this?
def update_edn_with_conf(edn, conf):

    edn['species'] = {
        'source-value': conf.get_chemical_symbols()
    }

    lattice = np.array(conf.get_cell()).tolist()

    edn['unrelaxed-periodic-cell-vector-1'] = {
        'source-value': lattice[0],
        'source-unit': 'angstrom'
    }

    edn['unrelaxed-periodic-cell-vector-2'] = {
        'source-value': lattice[1],
        'source-unit': 'angstrom'
    }

    edn['unrelaxed-periodic-cell-vector-3'] = {
        'source-value': lattice[2],
        'source-unit': 'angstrom'
    }

    edn['unrelaxed-configuration-positions'] = {
        'source-value': conf.positions.tolist(),
        'source-unit': 'angstrom'
    }


class PropertyHashError(Exception):
    pass
