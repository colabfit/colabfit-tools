__all__ = [
    'Dataset'
]

class Dataset:
    """
    A Dataset is essentially a wrapper around a Mongo database with
    some optimized queries and built-in comparison operators.

    A Dataset consists of the following collections:

    * :code:`summary`:
        A collection for storing summary information about a Dataset. For
        example, author names, external links (publications, repositories),
        descriptions, and basic statistics.
    * :code:`configurations`:
        A collection of :class:`~colabfit.tools.configuration.Configuration`
        documents.
    * :code:`data`:
        Either a collection of

    Attributes:


    """

    def __init__(self, client, name=''):
        self.client = client

        self._id = self.client.datasets[name]

        self.name = name

    @property
    def name(self):
        return self.client

    @property