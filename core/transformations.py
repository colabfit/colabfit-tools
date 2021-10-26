import numpy as np


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


class SubtractReference(BaseTransform):
    """Subtracts a reference energy off of the raw energy"""
    def __init__(self, ref_eng):
        super(SubtractReference, self).__init__(lambda x, c: x - ref_eng)


class ExtractCauchyStress(BaseTransform):
    """Extracts the 6-component vector from a full 3x3 stress matrix"""
    def __init__(self):
        def extract(data, configuration=None):
            data = np.array(data)
            return np.array([
                data[0, 0],
                data[1, 1],
                data[2, 2],
                data[1, 2],
                data[0, 2],
                data[0, 1],
            ]).tolist()

        super(ExtractCauchyStress, self).__init__(extract)


class ReshapeForces(BaseTransform):
    """Reshapes forces into an (N, 3) matrix"""
    def __init__(self):
        def reshape(data, configuration=None):
            data = np.array(data)
            n = np.prod(data.shape)//3

            return data.reshape((n, 3))

        super(ReshapeForces, self).__init__(reshape)


class Sequential(BaseTransform):
    def __init__(self, *args):
        def wrapper(data, configuration=None):
            for f in args:
                data = f(data, configuration)
            return data

        super(Sequential, self).__init__(wrapper)