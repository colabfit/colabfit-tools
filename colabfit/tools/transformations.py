import numpy as np

__all__ = [
    'BaseTransform',
    'AddDivide',
    'PerAtomEnergies',
    'ReshapeForces',
    'Sequential'
]


class BaseTransform:
    """
    A Transform is used for processing raw data before loading it into a
    Dataset. For example for things like subtracting off a reference energy or
    extracting the 6-component version of the cauchy stress from a 3x3 matrix.
    """

    def __init__(self, tform):
        self._tform = tform

    def __call__(self, data, configuration=None):
        return self._tform(data, configuration)


class AddDivide(BaseTransform):
    """Adds a scalar to the data, then divides by a scalar"""
    def __init__(self, add, div):
        super(AddDivide, self).__init__(lambda x, c: (x+add)/div)

    def __str__(self):
        return 'AddDivide'

        
class PerAtomEnergies(BaseTransform):
    """Divides the energy by the number of atoms"""
    def __init__(self):
        def wrapper(data, configuration=None):
            return data/len(configuration)

        super(PerAtomEnergies, self).__init__(wrapper)

    def __str__(self):
        return 'PerAtomEnergies'


class ReshapeForces(BaseTransform):
    """Reshapes forces into an (N, 3) matrix"""
    def __init__(self):
        def reshape(data, configuration=None):
            data = np.array(data)
            n = np.prod(data.shape)//3

            return data.reshape((n, 3))

        super(ReshapeForces, self).__init__(reshape)

    def __str__(self):
        return 'ReshapeForces'


class Sequential(BaseTransform):
    def __init__(self, *args):
        def wrapper(data, configuration=None):
            for f in args:
                data = f(data, configuration)
            return data

        super(Sequential, self).__init__(wrapper)