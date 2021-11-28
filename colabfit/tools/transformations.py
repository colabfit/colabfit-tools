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

    def __call__(self, data, configurations):
        return self._tform(data, configurations)


class SubtractDivide(BaseTransform):
    """Adds a scalar to the data, then divides by a scalar"""
    def __init__(self, sub, div):
        super(SubtractDivide, self).__init__(lambda x, c: (x-sub)/div)

    def __str__(self):
        return 'SubtractDivide'


class PerAtomEnergies(BaseTransform):
    """Divides the energy by the number of atoms"""
    def __init__(self):
        def wrapper(data, configurations=None):
            return data/len(configurations[0])

        super(PerAtomEnergies, self).__init__(wrapper)

    def __str__(self):
        return 'PerAtomEnergies'


class ReshapeForces(BaseTransform):
    """Reshapes forces into an (N, 3) matrix"""
    def __init__(self):
        def reshape(data, configurations=None):
            data = np.array(data)
            n = np.prod(data.shape)//3

            return data.reshape((n, 3))

        super(ReshapeForces, self).__init__(reshape)

    def __str__(self):
        return 'ReshapeForces'


class Sequential(BaseTransform):
    """
    An object used for defining a chain of Transformations to be performed
    sequentially. For example:

    .. code-block:: python

        Sequential(
            PerAtomEnergies(),
            SubtractDivide(sub=<reference_energy>, div=1)
        )
    """
    def __init__(self, *args):
        def wrapper(data, configurations=None):
            for f in args:
                data = f(data, configurations)
            return data

        super(Sequential, self).__init__(wrapper)