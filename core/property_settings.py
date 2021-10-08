class PropertySettings:
    """
    This class is used to store information useful for reproducing a Property.

    Attributes:

        method (str):
            A short string describing the method used for computing the
            properties (e.g., 'VASP', 'QuantumEspresso', 'experiment', ...)

        description (str):
            A human-readable description of the settings.

        files (list):
            A list of strings, where each entry is the text of a file that may
            be useful for computing one or more of the properties.

        labels (list):
            A list of strings; generated by parsing `files`, `description`, and
            `method`.
    """

    def __init__(
        self,
        method='',
        description='',
        files=None,
        labels=None,
    ):

        if files is None: files = []
        if labels is None: labels = []

        self.method         = method
        self.files          = files
        self.description    = description
        self.labels         = labels

        self.parse_labels_from_files()


    def parse_labels_from_files(self):
        pass


    def __str__(self):
        return "PropertySettings(method='{}', description='{}')".format(
            self.method,
            self.description,
        )


    def __repr__(self):
        return str(self)