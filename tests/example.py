import sys
sys.path.append('.')
from core.dataset import Dataset, load_configurations
from core.property_settings import PropertySettings

# Create a skeleton Dataset
ds1 = Dataset('Example ColabFit Dataset')

ds1.authors      = ['J. Vita', 'E. Tadmor', 'R. Elliott', 'S. Martiniani']
ds1.links        = ['https://colabfit.openkim.org/']
ds1.description  = 'This is an example Dataset generated for testing purposes.'

# Add configurations, with their linked data if it exists
ds1.configurations = load_configurations(
    file_path='/home/josh/colabfit/data/FitSNAP/InP_JPCA2020.extxyz',
    # file_path='/home/josh/colabfit/data/acclab_helsinki/MoNbTaVW_cleaned.xyz',
    file_format='xyz',
    name_field='name',
    elements=['In', 'P'],
    # elements=['Mo', 'Nb', 'Ta', 'V', 'W'],
    default_name=ds1.name
)

# Rename a property name to a known name
ds1.rename_property('virial', 'stress')

properties = {
    # ColabFit name: (ASE field name, units)
    'energy': ('energy', 'eV'),
    'forces': ('forces', 'eV/Ang'),
    # 'forces': ('force', 'eV/Ang'),
    'stress': ('stress', 'kilobar'),
}

# Extract the properties from the configurations
ds1.load_data(properties, convert_units=False)

# Build configuration sets by setting the cs_regexes dictionary
ds1.cs_regexes = {
    'default': 'Curated configurations for producing an interatomic '\
        'potential for indium phosphide capable of capturing high-energy '\
        'defects that result from radiation damage cascades',
    '^Bulk': 'Ground state configuration for bulk zinc blende',
    '^EOS': 'Bulk zinc blende with uniform expansion and compression',
    '^Shear': 'Bulk zincblende with random cell shape modifications',
    '^Strain': 'Uniaxially strained bulk zinc blende',
    '^a(In|P)': 'Antisite defects in InP',
    '^aa': 'Diantisite defects',
    '^i(In|P)': 'Interstitial defects in InP',
    '^vP': 'Vacancy defects in InP',
    '^vv': 'Divacancy defects in InP',
    '^s_a(In|P|a)': 'No description',
    '^s_i(In|P)': 'No description',
    '^s_v(In|P)': 'No description',
    '^s_vv': 'No description',
}

# Apply configuration labels by setting the cs_regexes dictionary
ds1.co_label_regexes = {
    'Bulk|EOS|Shear|Strain': 'zincblende',
    'EOS': 'eos',
    'Shear|Strain': 'strain',
    '^a(In|P)': 'antisite',
    '^aa': 'diantisite',
    '^i(In|P)': 'interstitial',
    '^v(In|P|v)': 'vacancy',
}

ds1.ps_regexes = {
    '.*':
        PropertySettings(
            method='VASP',
            description='energies/forces',
            files=None,
            labels=['PBE-GGA'],
        )
}

# Resync to make sure metadata is updated everywhere necessary
ds1.resync()

print(ds1)

# nested_ds = Dataset('nested-ds')
# nested_ds.data.append(ds)
# nested_ds.resync()

# print(nested_ds)

ds1.to_markdown('tests/files/dummy_written.md', 'tests/files/dummy_written.extxyz', 'xyz')