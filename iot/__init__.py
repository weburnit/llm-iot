from collections import deque
from pydantic import BaseModel
from datetime import datetime
import pandas as pd
import asyncio
import datetime
import websockets


class SensorData(BaseModel):
    timestamp: float
    vibration: float
    rotate: float
    pressure: float
    volt: float


class SensorDataDetection(SensorData):
    anomaly: str
    failure: str


class IoTDataBuffer:

    def __init__(self, max_size=100):
        self.buffer = deque(maxlen=max_size)
        self.lock = asyncio.Lock()

    async def add_data(self, data):
        async with self.lock:
            self.buffer.append(data)

    async def get_next_data(self) -> SensorData:
        async with self.lock:
            if len(self.buffer) > 0:
                return self.buffer.popleft()
            else:
                return None


async def mock_mqtt(iot_data: pd.DataFrame, SOCKET_PORT):
    """Simulates incoming IoT data write to socket"""
    mqtt_endpoint = "ws://localhost:PORT/ws/mqtt".replace("PORT", "{0}".format(SOCKET_PORT))
    print(mqtt_endpoint)
    async with websockets.connect(mqtt_endpoint) as websocket:
        for _, row in iot_data.iterrows():
            data = SensorData(
                timestamp=datetime.datetime.now().timestamp(),
                vibration=row['vibration'],
                rotate=row['rotate'],
                pressure=row['pressure'],
                volt=row['volt']
            )
            await websocket.send(data.json())

            print(f"> Writing {data.json()}")

            await asyncio.sleep(1)

