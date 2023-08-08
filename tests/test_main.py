from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
from aitomic.iot import SensorData

import pytest


@pytest.mark.asyncio
def test_sensor_ws():
    from aitomic.main import app
    client = TestClient(app)
    with client.websocket_connect("/ws/sensor") as websocket:
        data = {}
        websocket.send_json(data)
        response = websocket.receive_json()
        assert response == {}


@pytest.mark.asyncio
def test_mqtt_ws():
    with patch('aitomic.iot.connection.ConnectionManager') as mock_connection:
        from aitomic.main import app

        client = TestClient(app)
        instance = mock_connection.return_value
        instance.broadcast = AsyncMock()
        with client.websocket_connect("/ws/mqtt") as websocket:
            data = SensorData(
                timestamp=111111.0,
                vibration=11111.0,
                rotate=222,
                pressure=233.0,
                volt=111.1,
                anomaly=False,
                failure=False
            )
            websocket.send_json(data.dict())

            response = websocket.receive_json()
            assert response == {'received': True}

