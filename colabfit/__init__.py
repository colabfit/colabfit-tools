__version__ = "0.0.1"
__author__  = "ColabFit"
__credit__  = "University of Minnesota"


ATOMS_ID_FIELD          = '_id'
ATOMS_NAME_FIELD        = '_name'
ATOMS_LABELS_FIELD      = '_labels'
ATOMS_CONSTRAINTS_FIELD = '_constraints'

DEFAULT_PROPERTY_NAME = 'configuration-nonorthogonal-periodic-3d-cell-fixed-particles-fixed'

from ase.units import create_units
UNITS = create_units('2014')

# Make GPa the base unit
UNITS['bar'] = 1e-4  # bar to GPa
UNITS['kilobar'] = 1e-1  # kilobar to GPa
UNITS['pascal'] = 1e-9  # pascal to GPa
UNITS['GPa'] = 1

UNITS['angstrom'] = UNITS['Ang']

OPENKIM_PROPERTY_UNITS = {
    'energy': 'eV',
    'forces': 'eV/angstrom',
    'stress': 'GPa',
    'unrelaxed-potential-energy': 'eV',
    'unrelaxed-potential-forces': 'eV/angstrom',
    'unrelaxed-cauchy-stress': 'GPa',
}


EDN_KEY_MAP = {
    'energy': 'unrelaxed-potential-energy',
    'forces': 'unrelaxed-potential-forces',
    'stress': 'unrelaxed-cauchy-stress',
}