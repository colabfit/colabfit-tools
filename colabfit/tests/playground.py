import sys
sys.path.append('.')

from tools.dataset import Dataset

ds1 = Dataset.from_markdown(
    '/home/josh/colabfit/data/formatted/V_PRM2019/README.md',
    convert_units=True,
    verbose=True
)

# ds2 = Dataset.from_markdown(
#     '/home/josh/colabfit/data/formatted/V_PRM2019/README.md',
#     convert_units=True,
#     verbose=True
# )

# ds1.merge(ds2)
# n_dup = ds1.clean()

print(ds1.configurations[0])

names = [
    'InP_JPCA2020',
    'MoNbTaVW_PRB2021',
    'Mo_PRM2019',
    'Nb_PRM2019',
    'Ta_Linear_JCP2015',
    'Ta_PRM2019',
    'V_PRM2019',
    'WBe_PRB2019',
    'W_PRB2019',
    'TiZrHfTa_APS2021',
    'CuPd_CMS2019',
    'CoNbV_CMS2019',
    'AlNiTi_CMS2019',
]

master = Dataset('master')

for i, name in enumerate(names):
    print('Merging', name)

    master.merge(
        Dataset.from_markdown(
            f'/home/josh/colabfit/data/formatted/{name}/README.md',
            convert_units=True,
            verbose=True
        ),
        clean=False
    )

n_dup = master.clean()

print(f'Discovered {n_dup} duplicate data entries')

master.resync()

# print(master)