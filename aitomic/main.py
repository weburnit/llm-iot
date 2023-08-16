import datetime

from fastapi import FastAPI, WebSocket, Request, WebSocketDisconnect
from aitomic.iot import SensorData, mock_mqtt
import argparse
import asyncio
from fastapi.responses import HTMLResponse
import os
from aitomic.model import IoTModel, load_dataframe
from aitomic.config import logger, HAS_CUDA
from aitomic.iot.connection import ConnectionManager
from aitomic.util import few_shot, complete, get_json

script_dir = os.path.dirname(__file__)

PORT = 8080
app = FastAPI(version="0.1")
trainer = IoTModel()

iot_data = None  # mock_iot_mqtt data source

manager = ConnectionManager()

parser = argparse.ArgumentParser(description='IOT application')


@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    with open(f"{script_dir}/index.html", "r") as f:
        html = f.read()
    html = html.replace("PORT", "{0}".format(PORT))
    return HTMLResponse(content=html)


@app.websocket("/ws/sensor")
async def sensor_endpoint(*, websocket: WebSocket):
    await manager.connect(websocket)
    try:
        await websocket.receive_text()
        await websocket.send_json({})
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.websocket("/ws/mqtt")
async def websocket_mqtt_endpoint(*, websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            sensor_data = SensorData.parse_raw(data)

            if HAS_CUDA:  # Detect anomaly and failure
                detection = await trainer.generate(few_shot() + complete(sensor_data))

                json_detect = get_json(detection)
                if json_detect is not None:
                    sensor_data.anomaly = json_detect['failure']
                    sensor_data.failure = json_detect['failure']

            else:  # Mock anomaly detection
                if datetime.datetime.now().timestamp() % 3 <= 1:
                    sensor_data.anomaly = True
                    sensor_data.failure = True
            await manager.broadcast(sensor_data.dict())
            await websocket.send_json({'received': True})
    except WebSocketDisconnect:
        pass


@app.get("/")
def get_app_version():
    return {"version": app.version}


@app.get("/fake_data")
async def get_app_version():
    iot_data = load_dataframe(args.train_files, args.metadata_file)
    await asyncio.run(mock_mqtt(iot_data, PORT))
    return {"version": app.version}


if __name__ == "__main__":

    parser.add_argument('--train-files', type=str, help='Path to the data and labels training files(feather)')
    parser.add_argument('--metadata-file', type=str, help='Path to the metadata file(metadata.json)')
    parser.add_argument('--train', type=bool, help='Required train')
    parser.add_argument('--train-base-model', type=str, help='Base Machine Learning model')
    parser.add_argument('--trained-new-name', type=str, help='Trained model name')
    parser.add_argument('--mqtt', type=bool, help='Enable Mqtt')
    parser.add_argument('--port', type=int, default=8080, help='PORT')

    args = parser.parse_args()
    if args.port > 0:
        PORT = args.port

    if args.mqtt:
        iot_data = load_dataframe(args.train_files, args.metadata_file)
        asyncio.run(mock_mqtt(iot_data, PORT))
        pass

    trainer.load_model(model_name=args.train_base_model)  # google/flan-t5-base
    if args.train:
        logger.info(f'Training files: {args.train_files}')
        logger.info(f'Metadata file: {args.metadata_file}')
        logger.info(f'Start training model {args.train_base_model} with adapter name {args.trained_new_name}')
        trainer.train(args.train_files.split(','),
                      args.metadata_file,
                      new_peft_model_name=args.trained_new_name)
        pass
    else:
        lora_model = '{0}_{1}'.format(args.train_base_model, args.trained_new_name).replace('/', '_')
        logger.info(f'Loading existing model: {lora_model}')
        trainer.load_lora('lora/{0}'.format(lora_model))
        logger.info(f'Loaded existing model success')
    import uvicorn

    uvicorn.run(app, host='0.0.0.0', port=PORT)
