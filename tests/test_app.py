import unittest

import numpy as np
from fastapi.testclient import TestClient

from app import app, prepare_pixels


class AppTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_site_and_assets(self):
        for path in ['/', '/static/app.js', '/static/style.css', '/health']:
            self.assertEqual(self.client.get(path).status_code, 200)

    def test_prediction(self):
        pixels = np.zeros((28, 28))
        pixels[4:24, 13:16] = 1
        response = self.client.post('/api/predict', json={'pixels': pixels.flatten().tolist()})
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertIn(result['digit'], range(10))
        self.assertEqual(len(result['scores']), 10)
        self.assertTrue(all(0 <= score <= 1 for score in result['scores']))
        self.assertEqual(len(result['processed_pixels']), 784)

    def test_invalid_drawings(self):
        for pixels in [[], [0] * 783, [0] * 785, [0] * 784, [-1] * 784, [2] * 784]:
            with self.subTest(length=len(pixels), first=pixels[:1]):
                self.assertEqual(self.client.post('/api/predict', json={'pixels': pixels}).status_code, 422)

    def test_nonfinite_pixels(self):
        for value in [float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                prepare_pixels([value] * 784)

    def test_centering_preserves_translated_shape(self):
        left = np.zeros((28, 28))
        left[2:19, 3:6] = 1
        right = np.zeros((28, 28))
        right[8:25, 20:23] = 1
        np.testing.assert_array_equal(prepare_pixels(left.flatten()), prepare_pixels(right.flatten()))


if __name__ == '__main__':
    unittest.main()
