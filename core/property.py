import itertools
import numpy as np
from bson import ObjectId

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
from core.observable import Observable


class Property(Observable):
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

            settings (list):
                A list of `colabfit.PropertySettings` objects

            instance_id (int):
                A positive non-zero integer
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

        convert_units(self.edn, units)

        if len(configurations) < 1:
            raise RuntimeError(
                "`Property.configurations` must contain at least 1 entry"
            )

        self.configurations = configurations

        # Add Property as observer of linked configurations
        for c in self.configurations:
            c.attach(self)

        # Add settings
        if settings is None:
            settings = []
        self.settings = settings


    @classmethod
    def EFS(cls, conf, units, settings=None, instance_id=1):
        """
        Constructs a property for storing energy/forces/stress data of an
        `colabfit.Configuration` object.

        Assumes that the properties, if provided, are stored in the following
        ways:
            energy: `conf.atoms.info['energy']`
            forces: `conf.atoms.arrays['forces']`
            stress: `conf.atoms.info['stress']`

        Note: `units` can use the following aliases:
            - 'energy' instead of 'unrelaxed-potential-energy'
            - 'forces' instead of 'unrelaxed-potential-forces'
            - 'stress' instead of 'unrelaxed-cauchy-stress'
        """

        edn = kim_edn.loads(kim_property_create(
            instance_id=instance_id, property_name=EFS_PROPERTY_NAME
        ))[0]

        update_edn_with_conf(edn, conf)

        units = dict(units)

        if 'energy' in conf.atoms.info:
            edn['unrelaxed-potential-energy'] = {
                'source-value': conf.atoms.info['energy'],
                'source-unit': units['energy'] if 'energy' in units
                    else units['unrelaxed-potential-energy']
            }

        if 'forces' in conf.atoms.arrays:
            edn['unrelaxed-potential-forces'] = {
                'source-value': conf.atoms.arrays['forces'],
                'source-unit': units['forces'] if 'forces' in units
                    else units['unrelaxed-potential-forces']
            }

        if 'stress' in conf.atoms.info:
            edn['unrelaxed-cauchy-stress'] = {
                'source-value': conf.atoms.info['stress'],
                'source-unit': units['stress'] if 'stress' in units
                    else units['unrelaxed-cauchy-stress']
            }

        return cls(
            name=EFS_PROPERTY_NAME,
            configurations=[conf],
            units=units,
            settings=settings,
            edn=edn
        )


    def attach(self, observer):
        self._observers.append(observer)


    def detach(self, observer):
        self._observers.remove(observer)


    def notify(self):
        for observer in self._observers:
            observer.update(self)


    def update_atoms(self, atoms):
        self.atoms = atoms
        self.notify()


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


def convert_units(edn, original_units):
    """
    For each key in `original_units`, convert `edn[key]` from
    `original_units[key]` to the expected ColabFit-compliant units.
    """

    for key, units in original_units.items():
        if key == 'energy':
            key = 'unrelaxed-potential-energy'
        elif key == 'forces':
            key = 'unrelaxed-potential-forces'
        elif key == 'stress':
            key = 'unrelaxed-cauchy-stress'

        split_units = list(itertools.chain.from_iterable([
            sp.split('/') for sp in units.split('*')
        ]))

        val = edn[key]['source-value']

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

        edn[key] = {
            'source-value': val,
            'source-unit': OPENKIM_PROPERTY_UNITS['pressure']
        }