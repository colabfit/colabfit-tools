from abc import ABC, abstractmethod
from typing import Dict, Any, List
from colabfit.tools.vast.utils import get_last_modified, _hash


class DataObject(ABC):
    """
    Abstract base class for data objects in the VAST framework.

    This class defines the interface and common functionality for all data objects,
    including methods for accessing metadata and computing hashes.
    """

    def __init__(self):
        self._hash: str = None
        self.row_dict: Dict[str, Any] = {}

    @abstractmethod
    def to_row_dict(self) -> Dict[str, Any]:
        """Convert entity to row dictionary representation."""
        pass

    @abstractmethod
    def get_identifier_keys(self) -> List[str]:
        """Return list of keys used for hashing/identification."""
        pass

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get the metadata associated with the data object.

        Returns:
            A dictionary containing the metadata.
        """
        return self.metadata

    def _generate_hash_and_id(self):
        """Generate hash and ID after row_dict is populated."""
        self._hash = str(_hash(self.row_dict, self.get_identifier_keys(), False))
        self.row_dict["hash"] = self._hash
        self.row_dict["last_modified"] = get_last_modified()

    def __hash__(self):
        return int(self._hash) if self._hash else super().__hash__()
