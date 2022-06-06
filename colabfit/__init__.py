__version__ = "0.0.1"
__author__  = "ColabFit"
__credit__  = "University of Minnesota"

HASH_LENGTH = 12
HASH_SHIFT = 0
# HASH_SHIFT = 2**63

ID_FORMAT_STRING= '{}_{:05d}_{:03d}'

MAX_STRING_LENGTH = 255
STRING_DTYPE_SPECIFIER = f'S{MAX_STRING_LENGTH}'

# _DBASE_COLLECTION           = '_databases'
_DATABASE_NAME              = 'colabfit_database'
_CONFIGS_COLLECTION         = 'configurations'
_PROPS_COLLECTION           = 'property_instances'
_PROPDEFS_COLLECTION        = 'property_definitions'
_PROPSETTINGS_COLLECTION    = 'property_settings'
_CONFIGSETS_COLLECTION      = 'configuration_sets'
_DATASETS_COLLECTION        = 'datasets'

SHORT_ID_STRING_NAME = 'short-id'
EXTENDED_ID_STRING_NAME = 'short-id'

ATOMS_NAME_FIELD            = '_name'
ATOMS_LABELS_FIELD          = '_labels'
ATOMS_LAST_MODIFIED_FIELD   = '_last_modified'
ATOMS_CONSTRAINTS_FIELD     = '_constraints'

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
    'virial': 'unrelaxed-cauchy-stress',
}