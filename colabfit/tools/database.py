from colabfit import (
    _MONGO_DB, _CONFIG_COLLECTION, _CONSET_COLLECTION,
    _PROP_COLLECTION, _PROPSET_COLLECTION, _DSETS_COLLECTION
)

from montydb import MontyClient

from colabfit.tools.configuration import Configuration


class Database:

    def __init__(self, uri=None):

        """
        Args:

            uri (str, optional):
                A path to a directory where the database should live on disk.
                May also be a URI to a running :class:`mongod` instance. If
                None, defaults to using in-memory storage.
        """

        if uri is None:
            # I did this because ':memory:' is a weird default argument...
            uri  = ":memory:"  # for in-memory MontyDB

        self.client = MontyClient(uri)

        self.database = self.client[_MONGO_DB]

        self.configurations     = self.database[_CONFIG_COLLECTION]
        self.properties         = self.database[_PROP_COLLECTION]
        self.property_settings  = self.database[_PROPSET_COLLECTION]
        self.configuration_sets = self.database[_CONSET_COLLECTION]
        self.datasets           = self.database[_DSETS_COLLECTION]


    def add_configurations(self, configurations):
        """
        Adds the Configurations into the database

        Args:

            configurations (Configuration or list):
                The Configurations to be added to the database
        """

        if isinstance(configurations, Configuration):
            configurations = [configurations]

        self.configurations.insert_many((co.todict() for co in configurations))


    def add_properties(self, properties):
        if not isinstance(properties, list):
            properties = [properties]

        self.properties.insert_many((pr.to_dict() for pr in properties))


    def parse_data(self, properties):
        pass