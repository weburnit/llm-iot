import json
from aitomic.iot import SensorData
import re


def get_json(input_str):
    match = re.findall(r'{[^}]*}', input_str)
    if match:
        for json_str in match:
            try:
                data = json.loads(json_str)
                return data
            except json.JSONDecodeError:
                return None
    return None


def few_shot():
    prompt = ('You are a service that detect failure and anomaly of iot device. assistant should return a JSON. for '
              'example: {"failure": true, "anomaly": true}')
    prompt += "user: Detect iot device with volt: 1 vibration: 2 pressure: 2 rotate: 3 \n\n assistant:" + json.dumps(
        {"failure": True, "anomaly": True}) + "\n\n\n"

    return prompt


def complete(row: SensorData):
    return f"Detect iot device with volt: {float(row.volt)} vibration: {float(row.vibration)} pressure: {float(row.pressure)} rotate: {float(row.rotate)}.\n\nassistant:"
