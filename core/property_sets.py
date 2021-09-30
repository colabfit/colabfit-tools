import itertools

from kim_property.instance import check_property_instances
from kim_property import (
    get_properties,
    check_instance_optional_key_marked_required_are_present
)
from kim_property.pickle import unpickle_kim_properties

KIM_PROPERTIES, PROPERTY_NAME_ERTY_ID, \
    PROPERTY_ID_TO_PROPERTY_NAME = unpickle_kim_properties()


class PropertySet:
    """
    Attributes:
        properties (list):
            A list of OpenKIM Property Instances, stored as nested dictionaries
            (EDN format)

        property_names (list):
            A list of strings of short KIM property names contained in the
            collection
    """

    def __init__(self):
        pass


    @property
    def properties(self):
        return self._properties


    @properties.setter
    def properties(self, properties):
        self._properties = properties

        self.property_names = [
            PROPERTY_ID_TO_PROPERTY_NAME[prop['property-id']]
            for prop in properties
        ]