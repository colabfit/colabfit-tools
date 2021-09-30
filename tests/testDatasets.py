import markdown
import unittest

from core.dataset import Dataset

class TestMDConstructor(unittest.TestCase):
    def test_from_html(self):
        with open('tests/files/test.md', 'r') as f:
            dataset = Dataset.from_markdown(
                markdown.markdown(f.read(), extensions=['tables'])
            )
        self.assertTrue(False)