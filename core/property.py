import json
import warnings
import itertools
import numpy as np

import kim_edn
from kim_property.instance import check_property_instances
from kim_property import (
    kim_property_create,
    get_properties,
    check_instance_optional_key_marked_required_are_present
)
from kim_property.pickle import unpickle_kim_properties

KIM_PROPERTIES, PROPERTY_NAME_ERTY_ID, \
    PROPERTY_ID_TO_PROPERTY_NAME = unpickle_kim_properties()

available_kim_properties = get_properties()

from core import EFS_PROPERTY_NAME
from core import UNITS, OPENKIM_PROPERTY_UNITS

# These are fields that are related to the geometry of the atomic structure, and
# should be checked in the Configuration object, not here.
ignored_fields = [
    'a',
    'species',
    'unrelaxed-configuration-positions',
    'unrelaxed-periodic-cell-vector-1',
    'unrelaxed-periodic-cell-vector-2',
    'unrelaxed-periodic-cell-vector-3',
]



class Property:
    """
    A Property is used to store the results of some kind of calculation or
    experiment, and should be mapped to an OpenKIM Property Definition. Best
    practice is for the Property to also point to one or more
    PropertySettings objects that fully define the conditions under which the
    Property was obtained.

    A Property will be observed by zero to many Dataset objects.

    Attributes:
        edn (dict):
            A dictionary defining an OpenKIM Property Instance in EDN format.

        configurations (list):
            A list of `colabfit.Configuration` objects

        settings (list):
            A list of PropertySettings objects defining the conditions under
            which the propoerty was obtained. This list is allowed to be empty,
            but it is highly recommended that it have at least one entry.
    """

    _observers = []

    def __init__(
        self,
        name,
        configurations,
        property_map,
        settings=None,
        edn=None,
        instance_id=1,
        convert_units=False
        ):
        """
        Args:
            name (str):
                Short OpenKIM Property Definition name

            conf (Configuration):
                A ColabFit Configuration object

            property_map (dict):
                key = a string that can be used as a key like `self.edn[key]`
                value = A sub-dictionary with the following structure:
                    {
                        'field': A field name used to access atoms.info
                            or atoms.arrays
                        'units': A string matching one of the units names in
                            ase.units (https://wiki.fysik.dtu.dk/ase/ase/units.html)
                    }

                These units will be used to convert the given units to eV,
                Angstrom, a.m.u., Kelvin, ...

                For compound units (e.g. "eV/Ang"), the string will be split on
                '*' and '/'. The raw data will be multiplied by the first unit
                and anything preceded by a '*'. It will be divided by anything
                preceded by a '/'.

            settings (PropertySettings):
                A `colabfit.PropertySettings` objects specifying how to compute
                the property.

            instance_id (int):
                A positive non-zero integer

            convert_units (bool):
                If True, converts units to those expected by ColabFit. Default
                is False
        """

        if edn is None:
            self.edn = kim_edn.loads(kim_property_create(
                instance_id=instance_id, property_name=name
            ))[0]
        else:
            if name != PROPERTY_ID_TO_PROPERTY_NAME[edn['property-id']]:
                raise RuntimeError(
                    "`name` does not match `edn['property_name']`"
                )

            self.edn = edn

        check_instance_optional_key_marked_required_are_present(
            self.edn,
            available_kim_properties[self.edn['property-id']]
        )

        check_property_instances(
            self.edn,
            fp_path=available_kim_properties
        )

        # units_cleaned = {}
        # for key, val in property_map.items():
        #     if key == 'energy':
        #         key = 'unrelaxed-potential-energy'
        #     elif key == 'forces':
        #         key = 'unrelaxed-potential-forces'
        #     elif key == 'stress':
        #         key = 'unrelaxed-cauchy-stress'

        #     units_cleaned[key] = {}
        #     units_cleaned[key] = val

        self.property_map = property_map

        if convert_units:
            self.convert_units()

        if len(configurations) < 1:
            raise RuntimeError(
                "`Property.configurations` must contain at least 1 entry"
            )

        self.configurations = configurations

        # Add settings
        self.settings = settings


    @classmethod
    def EFS(
        cls, conf, property_map, settings=None, instance_id=1,
        convert_units=False
        ):
        """
        Constructs a property for storing energy/forces/stress data of an
        `colabfit.Configuration` object.

        Assumes that the properties, if provided, are stored in the following
        ways:
            energy: `conf.atoms.info[property_map['energy']]`
            forces: `conf.atoms.arrays[property_map['forces']]`
            stress: `conf.atoms.info[property_map['stress']]`

        Note: `property_map` can use the following aliases:
            - 'energy' instead of 'unrelaxed-potential-energy'
            - 'forces' instead of 'unrelaxed-potential-forces'
            - 'stress' instead of 'unrelaxed-cauchy-stress'
        """

        edn = kim_edn.loads(kim_property_create(
            instance_id=instance_id, property_name=EFS_PROPERTY_NAME
        ))[0]

        update_edn_with_conf(edn, conf)

        # print('in EFS', property_map)

        for key, val in property_map.items():
            if (val['field'] not in conf.atoms.info) and (val['field'] not in conf.atoms.arrays):
                field_not_found_message = 'Key "{}" not found in atoms.info '\
                    'or atoms.arrays. Available keys are: {}'.format(
                        val['field'],
                        list(conf.atoms.info.keys())
                        + list(conf.atoms.arrays.keys())
                    )
                warnings.warn(field_not_found_message)

                continue

            if (key == 'energy') or (key == 'unrelaxed-potential-energy'):
                edn['unrelaxed-potential-energy'] = {
                    'source-value': conf.atoms.info[val['field']],
                    'source-unit': val['units'],
                }
            if (key == 'forces') or (key == 'unrelaxed-potential-forces'):
                edn['unrelaxed-potential-forces'] = {
                    'source-value': conf.atoms.arrays[val['field']].tolist(),
                    'source-unit': val['units']
                }
            if (key == 'stress') or (key == 'unrelaxed-cauchy-stress'):
                stress = conf.atoms.info[val['field']]
                if stress.shape == (3, 3):
                    stress = [
                        stress[0, 0],
                        stress[1, 1],
                        stress[2, 2],
                        stress[1, 2],
                        stress[0, 2],
                        stress[0, 1],
                    ]
                else:
                    stress = stress.tolist()

                edn['unrelaxed-cauchy-stress'] = {
                    'source-value': stress,
                    'source-unit': val['units']
                }

        # print(edn.keys())

        return cls(
            name=EFS_PROPERTY_NAME,
            configurations=[conf],
            property_map=property_map,
            settings=settings,
            edn=edn,
            convert_units=convert_units,
        )


    def convert_units(self):
        """
        For each key in `original_units`, convert `edn[key]` from
        `original_units[key]` to the expected ColabFit-compliant units.
        """

        for key, val in self.property_map.items():
            units = val['units']

            split_units = list(itertools.chain.from_iterable([
                sp.split('/') for sp in units.split('*')
            ]))

            val = np.array(self.edn[key]['source-value'])

            val *= UNITS[split_units[0]]

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

            self.edn[key] = {
                'source-value': val.tolist(),
                'source-unit': OPENKIM_PROPERTY_UNITS[key]
            }

            self.property_map[key]['units'] = self.edn[key]['source-unit']


    def __hash__(self):
        return hash((
            hash(self.settings),
            tuple([hash(c) for c in self.configurations]),
            json.dumps(self.edn)
        ))


    def __eq__(self, other):
        """
        Returns False if any of the following conditions are true:

        - Properties point to settings with different calculation methods
        - Properties point to different configurations
        - OpenKIM EDN fields differ in any way
        """

        if self.settings is not None:
            if other.settings is None:
                return False

            if self.settings != other.settings:
                return False

        if set(self.configurations) != set(other.configurations):
            return False

        for my_field, my_val in self.edn.items():
            # Check if the field exists
            if my_field not in other.edn:
                return False

            # Compare value if it's not a field that should be ignored
            if my_field not in ignored_fields:
                if my_val != other.edn[my_field]:
                    return False

        return True


    def __neq__(self, other):
        return not self.__eq__(other)


    def __str__(self):
        return "Property(instance_id={}, name='{}')".format(
            self.edn['instance-id'],
            PROPERTY_ID_TO_PROPERTY_NAME[self.edn['property-id']]
        )


    def __repr__(self):
        return str(self)


def update_edn_with_conf(edn, conf):

    edn['species'] = {
        'source-value': conf.atoms.get_chemical_symbols()
    }

    lattice = np.array(conf.atoms.get_cell()).tolist()

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
        'source-value': conf.atoms.positions.tolist(),
        'source-unit': 'angstrom'
    }

