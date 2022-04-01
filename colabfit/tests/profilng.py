import numpy as np
from ase import Atoms

from colabfit.tools.database import MongoDatabase
from colabfit.tools.configuration import Configuration

def main():
    database = MongoDatabase('debug', drop_database=True)

    configurations = []
    for i in range(1, 1001):
        atoms = Atoms(
            symbols=f'H{i}',
            positions=np.random.random((i, 3)),
            cell=[[1,0,0],[0,1,0],[0,0,1]],
        )

        atoms.info['property-data.data'] = np.random.random((100, 1000))

        configurations.append(Configuration.from_ase(atoms))

    property_map = {
        'property-data': [{
            'data': {'field': 'property-data.data', 'units': None}
        }]
    }

    database.insert_property_definition({
        'property-id': 'property-data',
        'property-title': 'A title',
        'property-description': 'A description',

        'data': {
            'type': 'float',
            'has-unit': False,
            'extent': [100,1000],
            'required': True,
            'description': 'Some data'
        }
    })

    ids = database.insert_data(
        configurations,
        property_map,
        generator=True
    )

    co_ids, pr_ids = list(zip(*ids))

    rebuilt_configs = database.get_configurations(
        co_ids, pr_ids, attach_properties=True
    )


if __name__ == '__main__':
    main()