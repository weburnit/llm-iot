from collections import deque
from pydantic import BaseModel
import pandas as pd
import asyncio
import websockets
from datetime import datetime
from aitomic.config import logger


class SensorData(BaseModel):
    timestamp: float
    vibration: float
    rotate: float
    pressure: float
    volt: float
    anomaly: bool
    failure: bool


async def mock_mqtt(iot_data: pd.DataFrame, SOCKET_PORT):
    """Simulates incoming IoT data write to socket"""
    mqtt_endpoint = f"ws://localhost:{SOCKET_PORT}/ws/mqtt"
    print(mqtt_endpoint)
    async with websockets.connect(mqtt_endpoint) as websocket:
        for _, row in iot_data.iterrows():
            data = SensorData(
                # timestamp=row['datetime'].timestamp(),
                timestamp=datetime.now().timestamp(),
                vibration=row['vibration'],
                rotate=row['rotate'],
                pressure=row['pressure'],
                volt=row['volt'],
                anomaly=False,
                failure=False
            )
            await websocket.send(data.json())

            logger.debug(f"> Writing {data.json()}")

            await asyncio.sleep(1)
