import numpy as np
from hashlib import sha512

from colabfit import HASH_LENGTH, HASH_SHIFT, STRING_DTYPE_SPECIFIER
class PropertySettings:
    """
    This class is used to store information useful for reproducing a Property.

    Attributes:

        method (str):
            A short string describing the method used for computing the
            properties (e.g., 'VASP', 'QuantumEspresso', 'experiment', ...)

        description (str):
            A human-readable description of the settings.

        fields (dict):
            A dictionary of additional information. key = name of the field;


            .. code-block:: python

                value = {
                    'source-value': <key_for_extracting_from_configuration>,
                    'source-units': <string_specifying_units> or None
                    }

            For more details onw how to build this dictionary, refer to the
            "property map" description in the documentation, which follows a
            similar structure. These fields will be used to extract data from a
            Configuration object when inserting into the database.

        files (list):
            A list of 2-tuples, where the first value of each tuple is the name
            of a file, and the second value is the contents of the file.

        labels (list):
            A list of strings; generated by parsing `files`, `description`, and
            `method`.
    """

    def __init__(
        self,
        method='',
        description='',
        fields=None,
        files=None,
        labels=None,
    ):

        if files is None:
            files = []

        if files:
            for ftup in files:
                if not isinstance(ftup, tuple):
                    raise RuntimeError(
                        "PropertySettings files should be 2-tuples of "\
                            "(file_name, file_contents)."
                    )

        if labels is None:
            labels = set()
        elif isinstance(labels, str):
            labels = set([labels])
        else:
            labels = set(labels)

        if fields is None:
            fields = {}

        self.method         = method
        self.fields         = fields
        self.files          = files
        self.description    = description
        self.labels         = labels

        self.parse_labels_from_files()


    def parse_labels_from_files(self):
        pass


    def __hash__(self,):
        """
        Hashes method, description, field contents, and file contents.
        Does NOT use the description or the labels for hashing.
        """

        _hash = sha512()

        file_hashes = []

        for k, vdict in self.fields.items():
            _hash.update(k.encode('utf-8'))

            try:
                hashval =  np.round_(
                    np.array(vdict['source-value']), decimals=12
                ).data.tobytes()
            except:
                try:
                    hashval = np.array(
                        vdict['source-value'], dtype=STRING_DTYPE_SPECIFIER
                    ).data.tobytes()
                except:
                    raise PropertySettingsHashError(
                        "Could not hash key {}: {}".format(k, vdict)
                    )

            _hash.update(hashval)
            _hash.update(str(vdict['source-unit']).encode('utf-8'))


        for (fname, contents) in self.files:
            file_hashes.append(_hash.update(contents.encode('utf-8')))

        _hash.update(self.method.encode('utf-8'))
        # _hash.update(self.description.encode('utf-8'))

        return int(str(int(_hash.hexdigest(), 16)-HASH_SHIFT)[:HASH_LENGTH])


    def __eq__(self, other):
        """Equality check compares hashes"""
        return hash(self) == hash(other)


    def __str__(self):
        return "PropertySettings(method='{}', description='{}', "\
            "labels={})".format(
                self.method,
                self.description,
                self.labels,
            )


    def __repr__(self):
        return str(self)


class PropertySettingsHashError(Exception):
    pass
