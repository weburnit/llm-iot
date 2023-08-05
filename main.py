from fastapi import FastAPI
from pydantic import BaseModel
from model import Trainer

app = FastAPI(version="0.1")
trainer = Trainer()

import argparse


class Prompt(BaseModel):
    vibration: float
    rotate: float
    volt: float
    pressure: float
    max_tokens: int = 28


@app.get("/")
def get_app_version():
    return {"version": app.version}


@app.post("/device")
async def interact_model(prompt: Prompt):
    return trainer.generate(
        f"Detect iot device with volt: {prompt.volt} vibration: {prompt.vibration}, pressure: {prompt.pressure}, rotate: {prompt.rotate} and failure is True or False?")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FastAPI application')
    parser.add_argument('--train-files', type=str, help='Path to the training files')
    parser.add_argument('--metadata-file', type=str, help='Path to the metadata file')
    parser.add_argument('--train', type=bool, help='Required train')
    parser.add_argument('--train-base-model', type=str, help='Base Machine Learning model')
    parser.add_argument('--trained-new-name', type=str, help='Trained model name')

    args = parser.parse_args()

    trainer.load_model(model_name=args.train_base_model)  # google/flan-t5-base
    if args.train:
        trainer.train(args.train_files.split(','),
                      args.metadata_file,
                      new_peft_model_name='train-iot-device-full-train')
    # Do something with the arguments
    print(f'Training files: {args.train_files}')
    print(f'Metadata file: {args.metadata_file}')

    import uvicorn

    uvicorn.run(app, host='0.0.0.0', port=8000)
