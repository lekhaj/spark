import unittest
from unittest.mock import Mock
from src.pipeline.pipeline import Pipeline

class TestPipeline(unittest.TestCase):

    def setUp(self):
        # Mock the required processors
        self.mock_text_processor = Mock()
        self.mock_grid_processor = Mock()
        self.pipeline = Pipeline(self.mock_text_processor, self.mock_grid_processor)

    def test_process_text(self):
        text_prompt = "A beautiful sunset over the mountains."
        self.mock_text_processor.convert_to_image.return_value = "mock_image_path"

        image = self.pipeline.process_text(text_prompt)

        self.mock_text_processor.convert_to_image.assert_called_once_with(text_prompt)
        self.assertEqual(image, "mock_image_path")

    def test_process_grid(self):
        # Setup the return value for the grid processor
        expected_result = "path/to/generated/grid_image.png"
        self.mock_grid_processor.convert_grid_to_image.return_value = expected_result

        # Test data
        test_grid = [
            ["mountain", "forest"],
            ["lake", "desert"]
        ]

        # Call the method being tested
        result = self.pipeline.process_grid(test_grid)

        # Assert the grid processor was called with the correct argument
        self.mock_grid_processor.convert_grid_to_image.assert_called_once_with(test_grid)

        # Assert the result matches what's expected
        self.assertEqual(result, expected_result)

if __name__ == '__main__':
    unittest.main()