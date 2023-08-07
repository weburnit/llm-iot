from fastapi import FastAPI, WebSocket, Request
from model import IoTModel, load_dataframe
import argparse
from config import FAILURE_PROMPT, ANOMALY_PROMPT
from iot import SensorData, SensorDataDetection, IoTDataBuffer, mock_mqtt
import asyncio
from fastapi.responses import HTMLResponse

PORT = 8083
app = FastAPI(version="0.1")
trainer = IoTModel()

iot_data = None  # mock_iot_mqtt data source
data_buffer = IoTDataBuffer()
anomaly_buffer = IoTDataBuffer()


@app.get("/index", response_class=HTMLResponse)
async def get_index(request: Request):
    with open("index.html", "r") as f:
        html = f.read()
    html = html.replace("PORT", "{0}".format(PORT))
    return HTMLResponse(content=html)


@app.websocket("/ws/sensor")
async def sensor_endpoint(*, websocket: WebSocket):
    await websocket.accept()
    while True:
        new_data = await data_buffer.get_next_data()
        if new_data is not None:
            print(f"Received and Push to buffer: {new_data}")
            await anomaly_buffer.add_data(new_data)
            await websocket.send_json(new_data.dict())
        await websocket.send_json({})


@app.websocket("/ws/mqtt")
async def websocket_mqtt_endpoint(*, websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        sensor_data = SensorData.parse_raw(data)

        print(f"PUSSHING and Push to buffer: {data}")
        await data_buffer.add_data(sensor_data)
        await websocket.send_json({})


@app.websocket("/ws/anomaly")
async def anomaly(websocket: WebSocket):
    await websocket.accept()
    while True:
        detect_data = await anomaly_buffer.get_next_data()
        if detect_data is not None:
            result = await trainer.generate(ANOMALY_PROMPT.format(detect_data.volt, detect_data.vibration,
                                                                  detect_data.pressure,
                                                                  detect_data.rotate))
            print("Detect anomaly: ", result)
            await websocket.send_json(SensorDataDetection(vibration=detect_data.vibration,
                                                          rotate=detect_data.rotate,
                                                          volt=detect_data.volt,
                                                          pressure=detect_data.pressure,
                                                          anomaly=result).dict())
        await websocket.send_json({})


@app.get("/")
def get_app_version():
    return {"version": app.version}


@app.get("/fake_data")
def get_app_version():
    iot_data = load_dataframe(args.train_files, args.metadata_file)
    asyncio.run(mock_mqtt(iot_data, PORT))
    return {"version": app.version}


@app.post("/failure")
async def model_failure_detection(prompt: SensorData):
    return trainer.generate(FAILURE_PROMPT.format(prompt.volt, prompt.vibration, prompt.pressure, prompt.rotate))


@app.post("/anomaly")
async def model_anomalous_detection(prompt: SensorData):
    return trainer.generate(ANOMALY_PROMPT.format(prompt.volt, prompt.vibration, prompt.pressure, prompt.rotate))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='IOT application')
    parser.add_argument('--train-files', type=str, help='Path to the data and labels training files(feather)')
    parser.add_argument('--metadata-file', type=str, help='Path to the metadata file(metadata.json)')
    parser.add_argument('--train', type=bool, help='Required train')
    parser.add_argument('--train-base-model', type=str, help='Base Machine Learning model')
    parser.add_argument('--trained-new-name', type=str, help='Trained model name')

    parser.add_argument('--mqtt', type=bool, help='Enable Mqtt')

    args = parser.parse_args()
    trainer.load_model(model_name=args.train_base_model)  # google/flan-t5-base
    if args.train:
        print(f'Training files: {args.train_files}')
        print(f'Metadata file: {args.metadata_file}')
        print(f'Start training model {args.train_base_model} with adapter name {args.trained_new_name}')
        trainer.train(args.train_files.split(','),
                      args.metadata_file,
                      new_peft_model_name=args.trained_new_name)
        pass
    else:
        lora_model = '{0}_{1}'.format(args.train_base_model, args.trained_new_name).replace('/', '_')
        print(f'Loading existing model: ', lora_model)
        trainer.load_lora('lora/{0}'.format(lora_model))
        print(f'Loaded existing model success')
    import uvicorn

    uvicorn.run(app, host='0.0.0.0', port=PORT)
