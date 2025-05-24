import unittest
from src.pipeline.grid_processor import GridProcessor

class TestGridProcessor(unittest.TestCase):
    def setUp(self):
        self.grid_processor = GridProcessor()

    def test_convert_grid_to_image(self):
        grid = [
            [0, 1, 2],
            [1, 2, 3],
            [2, 3, 4]
        ]
        image = self.grid_processor.convert_grid_to_image(grid)
        self.assertIsNotNone(image)
        # Additional assertions can be added to verify the image content

    def test_empty_grid(self):
        grid = []
        image = self.grid_processor.convert_grid_to_image(grid)
        self.assertIsNone(image)  # Assuming the method returns None for empty grids

    def test_invalid_grid(self):
        grid = [
            [0, 1, 'a'],  # Invalid entry
            [1, 2, 3],
            [2, 3, 4]
        ]
        with self.assertRaises(ValueError):
            self.grid_processor.convert_grid_to_image(grid)

if __name__ == '__main__':
    unittest.main()