ATOMS_ID_FIELD    = '_id'
ATOMS_NAME_FIELD    = '_name'
ATOMS_LABELS_FIELD  = '_labels'

EFS_PROPERTY_NAME = 'configuration-nonorthogonal-periodic-3d-cell-fixed-particles-fixed'
from ase.units import create_units
UNITS = create_units('2014')
if 'bar' not in UNITS:
    UNITS['bar'] = 1e-4  # bar to GPa
if 'kilobar' not in UNITS:
    UNITS['kilobar'] = 1e-1  # kilobar to GPa
if 'pascal' not in UNITS:
    UNITS['pascal'] = 1e-9  # pascal to GPa

OPENKIM_PROPERTY_UNITS = {
    'unrelaxed-potential-energy': 'eV',
    'unrelaxed-potential-forces': 'eV/angstrom',
    'unrelaxed-cauchy-stress': 'GPa',
}

