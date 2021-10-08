import os
import sys
sys.path.append('.')
from core.dataset import Dataset, load_configurations

ds = Dataset('test')

configs1 = load_configurations(
    file_path='/home/josh/colabfit/data/FitSNAP/InP_JPCA2020.extxyz',
    file_format='xyz',
    name_field='name',
    elements=['In', 'P'],
    default_name=ds.name
)

ds.configurations = configs1

properties = {'energy': 'eV', 'forces': 'eV/Ang', 'stress': 'kilobar'}

ds.rename_property('virial', 'stress')

ds.load_data(properties)

ds.cs_regexes = {
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

ds.co_label_regexes = {
    'Bulk|EOS|Shear|Strain': 'zincblende',
    'EOS': 'eos',
    'Shear|Strain': 'strain',
    '^a(In|P)': 'antisite',
    '^aa': 'diantisite',
    '^i(In|P)': 'interstitial',
    '^v(In|P|v)': 'vacancy',
}

ds.resync()

print(ds)
print('N_configs:', len(ds.configurations))
print('N_data:', len(ds.data))
print('Configuration sets:')
for cs in ds.configuration_sets:
    print('\t', cs)
