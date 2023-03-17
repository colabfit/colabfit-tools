from colabfit.tools.database import MongoDatabase, load_data
from colabfit.tools.configuration import AtomicConfiguration
from colabfit.tools.property_definitions import potential_energy_pd, atomic_forces_pd, cauchy_stress_pd
import argparse




if __name__ == "__main__":
    #parser = get_parser()
    #args = parser.parse_args()
    #ip=args.ip

    client = MongoDatabase('test',  nprocs=1,drop_database=True)


    # In[ ]:

    #Loads data, specify reader function if not "usual" file format
    configurations = load_data(
        file_path='/home/eric/Downloads/AlNiTi/train_1st_stage.cfg',
        file_format='cfg',
        name_field=None,
        elements=['Al', 'Ni', 'Ti'],
        default_name='train_1st_stage',
        verbose=True,
        generator=False
    )

    configurations += load_data(
        file_path='/home/eric/Downloads/AlNiTi/train_2nd_stage.cfg',
        file_format='cfg',
        name_field=None,
        elements=['Al', 'Ni', 'Ti'],
        default_name='train_2nd_stage',
        verbose=True,
        generator=False
    )


    # In[ ]:


    client.insert_property_definition(potential_energy_pd)
    client.insert_property_definition(atomic_forces_pd)
    client.insert_property_definition(cauchy_stress_pd)


    # In[ ]:


    property_map = {
        'potential-energy': [{
            'energy':   {'field': 'energy',  'units': 'eV'},
            'per-atom': {'field': 'per-atom', 'units': None},
    # For metadata want: software, method (DFT-XC Functional), basis information, more generic parameters 
            '_metadata': {
                'software': {'value':'VASP'},
            }
        }],
        
        'atomic-forces': [{
            'forces':   {'field': 'forces',  'units': 'eV/Ang'},
                '_metadata': {
                'software': {'value':'VASP'},
            }
        
        }],
        
        'cauchy-stress': [{
            'stress':   {'field': 'virial',  'units': 'GPa'},
            
                    '_metadata': {
                'software': {'value':'VASP'},
            }

        }],
    }


    # In[ ]:


    def tform(c):
        c.info['per-atom'] = False


    # In[ ]:


    ids = list(client.insert_data(
        configurations,
        property_map=property_map,
        generator=False,
        transform=tform,
        verbose=True
    ))

    all_co_ids, all_pr_ids = list(zip(*ids))


    #matches to data CO "name" field
    cs_regexes = {
        '.*':
            'Configurations generated using active learning by iteratively '\
            'fitting a MTP model, identifying configurations that required the '\
            'MTP to extrapolate, re-computing the energies/forces/structures of '\
            'those configurations with DFT, then retraining the MTP model.',
        'train_1st_stage':
            'Configurations used in the first stage of training',
        'train_2nd_stage':
            'Configurations used in the second stage of training',
    }

    cs_names=['all','1st_stage','2nd_stage']


    cs_ids = []

    for i, (regex, desc) in enumerate(cs_regexes.items()):
        co_ids = client.get_data(
            'configurations',
            fields='hash',
            query={'hash': {'$in': all_co_ids}, 'names': {'$regex': regex}},
            ravel=True
        ).tolist()

        print(f'Configuration set {i}', f'({regex}):'.rjust(22), f'{len(co_ids)}'.rjust(7))

        cs_id = client.insert_configuration_set(co_ids, description=desc,name=cs_names[i])

        cs_ids.append(cs_id)


    # In[ ]:


    ds_id = client.insert_dataset(
        cs_ids=cs_ids,
        pr_hashes=all_pr_ids,
        name='AlNiTi_CMS2019',
        authors=[
            'K. Gubaev', 'E. V. Podryabinkin', 'G. L. W. Hart', 'A. V. Shapeev'
        ],
        links=[
            'https://www.sciencedirect.com/science/article/pii/S0927025618306372?via%3Dihub',
            'https://gitlab.com/kgubaev/accelerating-high-throughput-searches-for-new-alloys-with-active-learning-data',
        ],
        description =  'This dataset was generated using the following active '\
        'learning scheme: 1) candidate structures relaxed by a partially-trained '\
        'MTP model, 2) structures for which the MTP had to perform extrapolation '\
        'are passed to DFT to be re-computed, 3) the MTP is retrained included '\
        'the structures that were re-computed with DFT, 4) steps 1-3 are repeated '\
        'until the MTP does not extrapolate on any of the original candidate '\
        'structures. The original candidate structures for this dataset included '\
        'about 375,000 binary and ternary structures enumerating all possible '\
        'unit cells with different symmetries (BCC, FCC, and HCP) and different '\
        'number of atoms',
        resync=True,
        verbose=True,
    )

