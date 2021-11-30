from html.parser import HTMLParser


class InvalidHeaderError(Exception):
    pass


class MarkdownFormatError(Exception):
    pass


class BadTableFormatting(Exception):
    pass


class DatasetParser(HTMLParser):

    KNOWN_HEADERS = [
        'Name',
        'Authors',
        'Links',
        'Description',
        'Data',
        'Summary',
        'Properties',
        'Property settings',
        'Property labels',
        'Configuration sets',
        'Configuration labels',
    ]

    def __init__(self):
        super().__init__()

        self.data           = {}
        self._t             = None
        self._table         = None
        self._loading_table = False
        self._loading_row   = False
        self._header        = None
        self._href          = None

    def get_data(self, k):
        try:
            return self.data[k]
        except:
            raise MarkdownFormatError(
                f"An error occurred trying to access DatasetParser.data['{k}']"
            )


    def handle_starttag(self, tag, attrs):
        self._t = tag

        if tag == 'table':
            self._table = []
            self._loading_table = True
        elif (tag == 'thead') or ('tag' == 'tbody'):
            pass
        elif tag == 'tr':
            self._loading_row = True
            self._table.append([])

        for att in attrs:
            if att[0] == 'href':
                self._href = att[1]


    def handle_endtag(self, tag):
        if tag == 'table':
            self.data[self._header] = self._table
            self._table = None
            self._loading_table = False
        elif tag == 'tr':
            self._loading_row = False

    def handle_data(self, data):
        data = data.rstrip('\n')

        if data:
            if self._t == 'h1':
                # Begin reading new block
                if data not in self.KNOWN_HEADERS:
                    raise InvalidHeaderError(
                        f"Header '{data}' not in {self.KNOWN_HEADERS}"
                    )
                self._header = data
                self.data[self._header] = []

            else:
                # Add data to current block
                if not self._loading_table:
                    # Adding basic text
                    self.data[self._header] += [_.strip() for _ in data.split('\n')]
                else:
                    # Specifically adding to a table
                    if self._loading_row:
                        if self._href is not None:
                            self._table[-1].append((data, self._href))
                            self._href = None
                        else:
                            self._table[-1].append(data)
