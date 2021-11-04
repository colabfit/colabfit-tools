import sys
sys.path.append('.')
from colabfit.tools.dataset import Dataset, load_data
from colabfit.tools.property_settings import PropertySettings
from colabfit.tools.transformations import ExtractCauchyStress

# Define transformations to use when parsing data
T = {'stress': ExtractCauchyStress()}

# Create a skeleton Dataset
ds1 = Dataset('InP_JPCA2020')

ds1.authors = ['M. A. Cusentino', 'M. A. Wood', 'A. P. Thompson']
ds1.links = [
    'https://pubs.acs.org/doi/10.1021/acs.jpca.0c02450',
    'https://github.com/FitSNAP/FitSNAP/tree/master/examples/InP_JPCA2020'
]
ds1.description  = \
    'This data set was used to generate a multi-element linear '\
    'SNAP potential for InP as published in Cusentino, M.A. et. al, J. Chem. '\
    'Phys. (2020). Intended to produce an interatomic potential for '\
    'indium phosphide capable of capturing high-energy defects that '\
    'result from radiation damage cascades.'

# Add configurations, with their linked data if it exists
ds1.configurations = load_data(
    file_path='/home/josh/colabfit/data/FitSNAP/InP_JPCA2020.extxyz',
    file_format='xyz',
    name_field='name',
    elements=['In', 'P'],
    default_name=ds1.name
)

# Rename a property name to a known name
ds1.rename_property('virial', 'stress')

ds1.property_map = {
    # ColabFit name: {'field': ASE field name, 'units': str}
    'energy': {'field': 'energy', 'units': 'eV'},
    'forces': {'field': 'forces', 'units': 'eV/Ang'},
    'stress': {'field': 'stress', 'units': 'kilobar'},
}

# Extract the properties from the configurations and loads them to Dataset.data
ds1.parse_data(
    transformations=T,
    convert_units=False
)

# Build configuration sets by setting the cs_regexes dictionary
ds1.cs_regexes = {
    'default':
        'Curated configurations for producing an interatomic '\
        'potential for indium phosphide capable of capturing high-energy '\
        'defects that result from radiation damage cascades',
    '^Bulk':
        'Ground state configuration for bulk zinc blende',
    '^EOS':
        'Bulk zinc blende with uniform expansion and compression',
    '^Shear':
        'Bulk zincblende with random cell shape modifications',
    '^Strain':
        'Uniaxially strained bulk zinc blende',
    '^a(In|P)':
        'Antisite defects in InP',
    '^aa':
        'Diantisite defects',
    '^i(In|P)':
        'Interstitial defects in InP',
    '^vP':
        'Vacancy defects in InP',
    '^vv':
        'Divacancy defects in InP',
    '^s_a(In|P|a)':
        'No description',
    '^s_i(In|P)':
        'No description',
    '^s_v(In|P)':
        'No description',
    '^s_vv':
        'No description',
}

# Apply configuration labels by setting the cs_regexes dictionary
ds1.co_label_regexes = {
    'Bulk|EOS|Shear|Strain':
        'zincblende',
    'EOS':
        'eos',
    'Shear|Strain':
        'strain',
    '^a(In|P)':
        'antisite',
    '^aa':
        'diantisite',
    '^i(In|P)':
        'interstitial',
    '^v(In|P|v)':
        'vacancy',
}

ds1.ps_regexes = {
    '.*':
        PropertySettings(
            method='VASP',
            description='energies/forces/stresses',
            files=None,
            labels=['PBE', 'GGA'],
        )
}

# Resync to make sure metadata is updated everywhere necessary
ds1.resync()

print(ds1)

ds1.to_markdown(
    'tests/files',
    'dummy_written1.md',
    'dummy_written1.extxyz',
    'xyz'
)

# Create a skeleton Dataset
ds2 = Dataset('MoNbTaVW_PRB2021')

ds2.authors      = ['J. Byggmästar', 'K. Nordlund', 'F. Djurabekova']
ds2.links = [
    'https://journals.aps.org/prb/abstract/10.1103/PhysRevB.104.104101',
    'https://doi.org/10.23729/1b845398-5291-4447-b417-1345acdd2eae'
]
ds2.description  = \
    'This dataset was originally designed to fit a GAP model '\
    'for the Mo-Nb-Ta-V-W quinary system that was used to study segregation '\
    'and defects in the body-centered-cubic refractory high-entropy alloy '\
    'MoNbTaVW.'

# Add configurations, with their linked data if it exists
ds2.configurations = load_data(
    file_path='/home/josh/colabfit/data/acclab_helsinki/MoNbTaVW_cleaned.xyz',
    file_format='xyz',
    name_field='name',
    elements=['Mo', 'Nb', 'Ta', 'V', 'W'],
    default_name=ds2.name
)

# Rename a property name to a known name
ds2.rename_property('virial', 'stress')

ds2.property_map = {
    # ColabFit name: {'field': ASE field name, 'units': str}
    'energy': {'field': 'energy', 'units': 'eV'},
    'forces': {'field': 'force',  'units': 'eV/Ang'},
    'stress': {'field': 'stress', 'units': 'kilobar'},
}

# Extract the properties from the configurations
ds2.parse_data(
    transformations=T,
    convert_units=False
)

# Build configuration sets by setting the cs_regexes dictionary
ds2.cs_regexes = {
    'default':
        'A variety of Mo-Nb-Ta-V-W structures',
    'bcc_distorted':
        'BCC configurations with random strains up to +/- 30% to '\
        'help train the far-from-equilibrium elastic response',
    'binary_alloys':
        'Binary BCC alloys sampling 10 different concentrations '\
        'from A_0.05B_0.95 to A_0.95B_0.05 and 3 different lattice constants '\
        'for every composition. Atoms are randomly ordered and shifted '\
        'slightly from their lattice positions.',
    '^composition':
        'Ternary, quaternary, and quinary BCC alloys. 3 linearly '\
        'spaced compositions were sampled, each with 3 different lattice '\
        'constants. Atoms are randomly ordered and shifted slightly from '\
        'their lattice positions.',
    'di-sia':
        'Configurations with two self-interstitial atoms',
    'di-vacancy':
        'Configurations with two vacancies',
    'dimer':
        'Dimers for fitting repulsive terms',
    'gamma_surface':
        'Configurations representing the full gamma surface',
    'hea_ints':
        '1-5 interstitial atoms randomly inserted into HEA lattices and '\
        'relaxed with a partially-trained tabGAP model',
    'hea_short_range':
        'Randomly placed unrelaxed interstitial atom in HEAs '\
        'to fit repulsion inside crystals, making sure that the closest '\
        'interatomic distance is not too short for DFT to be unreliable '\
        '(> 1.35 Ang)',
    'hea_small':
        'Bulk equiatomic quinary HEAs. Atoms are randomly ordered and shifted '\
        'slightly from their lattice positions. The lattice constant is '\
        'randomised in the range 3-3.4 Angstrom',
    'hea_surface':
        'Disordered HEA surfaces, including some of the '\
        'damaged/molten surface configurations from an existing pure W '\
        'dataset that were turned into HEAs',
    'hea_vacancies':
        '1-5 vacancies randomly inserted into HEA lattices, then '\
        'relaxed with a partially-trained tabGAP model',
    'isolated_atom':
        'Isolated atoms',
    '^liquid__':
        'Liquid configurations',
    'liquid_composition':
        'Liquid equiatomic binary, ternary, quaternary, and quinary alloys at '\
        'different densities',
    'liquid_hea':
        'Liquid HEA configurations',
    'mcmd':
        'Equiatomic quinary alloys generated via active learning by running '\
        'MCMD with a partially-trained tabGAP model.',
    'ordered_alloys':
        'Ordered binary, ternary, and quaternary alloys (always as a BCC '\
        'lattice, but with different crystal symmetries of the elemental '\
        'sublattices)',
    'phonon':
        'MD snapshots taken at 1000K for three different volumes',
    '^short_range':
        'BCC crystals with random interstitial atom defects to capture '\
        'short-range many-body dynamics',
    '^sia':
        'Configurations with a self-interstitial defect',
    'surf_liquid':
        'Damaged and molten surfaces',
    'tri-vacancy':
        'Configurations with three vacancies',
    '^vacancy':
        'Configurations with one vacancy',
}

# Apply configuration labels by setting the cs_regexes dictionary
ds2.co_label_regexes = {
    'alloys':
        'bcc',
    'bcc':
        'bcc strain',
    'di-vacancy':
        ['vacancy', 'divacancy'],
    'dimer':
        ['dimer', 'warning', 'large_forces', 'repulsive'],
    'gamma_surface':
        'gamma_surface',
    'hea':
        'hea',
    'hea_ints':
        'interstitial',
    'isolated_atom':
        'isolated_atom',
    'liquid':
        'liquid',
    'mcmd':
        'aimd',
    'phonon':
        'aimd',
    'short_range':
        ['bcc', 'interstitial', 'repulsive', 'large_forces', 'warning'],
    'sia':
        'interstitial',
    'surface':
        'surface',
    'tri-vacancy':
        ['vacancy', 'divacancy', 'trivacancy'],
    'vacancies':
        'vacancy',
    'vacancy':
        'vacancy',
}

ds2.ps_regexes = {
    '.*':
        PropertySettings(
            method='VASP',
            description='energies/forces/stresses',
            files=None,
            labels=['PBE', 'GGA'],
        )
}

# Resync to make sure metadata is updated everywhere necessary
ds2.resync()

ds2.to_markdown(
    'tests/files/',
    'dummy_written2.md',
    'dummy_written2.extxyz',
    'xyz'
)

ds1 = Dataset.from_markdown('tests/files/dummy_written1.md', transformations=T)
ds2 = Dataset.from_markdown('tests/files/dummy_written2.md', transformations=T)

# print(ds1)
# print(ds2)

nested_ds = Dataset('nested-ds')

nested_ds.attach_dataset(ds1)
nested_ds.attach_dataset(ds2)

nested_ds.resync()

print(nested_ds)

nested_ds.to_markdown(
    'tests/files/',
    'dummy_nested.md',
    'dummy_nested.extxyz',
    'xyz')