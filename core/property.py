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

from core import EFS_PROPERTY_NAME, ATOMS_ID_FIELD, ATOMS_NAME_FIELD
from core import UNITS, OPENKIM_PROPERTY_UNITS


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
        units,
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

            units (dict):
                key = a string that can be used as a key like `self.edn[key]`
                value = A string matching one of the units names in ase.units
                    (https://wiki.fysik.dtu.dk/ase/ase/units.html)

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

        units_cleaned = {}
        for key, val in units.items():
            if key == 'energy':
                key = 'unrelaxed-potential-energy'
            elif key == 'forces':
                key = 'unrelaxed-potential-forces'
            elif key == 'stress':
                key = 'unrelaxed-cauchy-stress'

            units_cleaned[key] = val

        self.units = units_cleaned
        if convert_units:
            self.convert_units(units_cleaned)

        if len(configurations) < 1:
            raise RuntimeError(
                "`Property.configurations` must contain at least 1 entry"
            )

        self.configurations = configurations

        # Add settings
        self.settings = settings


    @classmethod
    def EFS(
        cls, conf, property_fields, settings=None, instance_id=1,
        convert_units=False
        ):
        """
        Constructs a property for storing energy/forces/stress data of an
        `colabfit.Configuration` object.

        Assumes that the properties, if provided, are stored in the following
        ways:
            energy: `conf.atoms.info[property_fields['energy']]`
            forces: `conf.atoms.arrays[property_fields['forces']]`
            stress: `conf.atoms.info[property_fields['stress']]`

        Note: `property_fields` can use the following aliases:
            - 'energy' instead of 'unrelaxed-potential-energy'
            - 'forces' instead of 'unrelaxed-potential-forces'
            - 'stress' instead of 'unrelaxed-cauchy-stress'
        """

        edn = kim_edn.loads(kim_property_create(
            instance_id=instance_id, property_name=EFS_PROPERTY_NAME
        ))[0]

        update_edn_with_conf(edn, conf)

        property_fields = dict(property_fields)

        units = {}

        if 'energy' in conf.atoms.info:

            if 'energy' in property_fields:
                key = 'energy'
            else:
                key = 'unrelaxed-potential-energy'

            units['unrelaxed-potential-energy'] = property_fields[key][1]

            edn['unrelaxed-potential-energy'] = {
                'source-value': conf.atoms.info[property_fields[key][0]],
                'source-unit': units['unrelaxed-potential-energy']
            }

        if 'forces' in conf.atoms.arrays:

            if 'forces' in property_fields:
                key = 'forces'
            else:
                key = 'unrelaxed-potential-forces'

            units['unrelaxed-potential-forces'] = property_fields[key][1]

            edn['unrelaxed-potential-forces'] = {
                'source-value': conf.atoms.arrays[property_fields[key][0]].tolist(),
                'source-unit': property_fields[key][1]
            }

        if 'stress' in conf.atoms.info:

            if 'stress' in property_fields:
                key = 'stress'
            else:
                key = 'unrelaxed-cauchy-stress'

            units['unrelaxed-cauchy-stress'] = property_fields[key][1]

            stress = conf.atoms.info[property_fields[key][0]]
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
                'source-unit': property_fields[key][1]
            }

        return cls(
            name=EFS_PROPERTY_NAME,
            configurations=[conf],
            units=units,
            settings=settings,
            edn=edn,
            convert_units=convert_units,
        )

    def convert_units(self, original_units):
        """
        For each key in `original_units`, convert `edn[key]` from
        `original_units[key]` to the expected ColabFit-compliant units.
        """

        # Avoid change in place
        original_units = dict(original_units)

        for key, units in original_units.items():
            if key not in self.units:
                continue

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

            self.units[key] = self.edn[key]['source-unit']


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

