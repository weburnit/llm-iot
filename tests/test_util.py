from aitomic.util import few_shot, complete, get_json
import json
from aitomic.iot import SensorData
import unittest


class TestGetJson(unittest.TestCase):
    def test_few_shot(self):
        assert json.dumps({'failure': True, 'anomaly': True}) in few_shot()
        assert 'assistant should return a JSON' in few_shot()

    def test_complete(self):
        data = SensorData.model_validate(
            {"timestamp": 1, "volt": 1, "pressure": 2, "vibration": 3, "rotate": 4, "anomaly": False, "failure": False})
        assert 'assistant:' in complete(data)
        assert "volt: 1.0" in complete(data)
        assert "pressure: 2.0" in complete(data)
        assert "vibration: 3.0" in complete(data)
        assert "rotate: 4.0" in complete(data)

    def test_valid_json(self):
        input_str = 'Hello {"key1": "value1"} world'
        expected = {"key1": "value1"}
        actual = get_json(input_str)
        self.assertEqual(actual, expected)

    def test_invalid_json(self):
        input_str = 'Hello {invalid} world'
        actual = get_json(input_str)
        self.assertIsNone(actual)

    def test_no_json(self):
        input_str = 'Hello world'
        actual = get_json(input_str)
        self.assertIsNone(actual)

    def test_multiple_json(self):
        input_str = 'Hello {"key1":"value1"} ignore {"key2":"value2"}'
        expected = {"key1": "value1"}
        actual = get_json(input_str)
        self.assertEqual(actual, expected)
