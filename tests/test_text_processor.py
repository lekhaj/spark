import unittest
from src.pipeline.text_processor import TextProcessor

class TestTextProcessor(unittest.TestCase):

    def setUp(self):
        self.processor = TextProcessor()

    def test_convert_to_image_valid_prompt(self):
        prompt = "A beautiful sunset over the mountains"
        image = self.processor.convert_to_image(prompt)
        self.assertIsNotNone(image)
        self.assertTrue(hasattr(image, 'save'))  # Assuming the image object has a save method

    def test_convert_to_image_empty_prompt(self):
        prompt = ""
        with self.assertRaises(ValueError):
            self.processor.convert_to_image(prompt)

    def test_convert_to_image_invalid_prompt(self):
        prompt = None
        with self.assertRaises(ValueError):
            self.processor.convert_to_image(prompt)

if __name__ == '__main__':
    unittest.main()