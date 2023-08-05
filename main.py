from fastapi import FastAPI
from pydantic import BaseModel
from model import Trainer
import argparse
from config import FAILURE_PROMPT, ANOMALY_PROMPT

app = FastAPI(version="0.1")
trainer = Trainer()


class Prompt(BaseModel):
    vibration: float
    rotate: float
    volt: float
    pressure: float
    max_tokens: int = 28


@app.get("/")
def get_app_version():
    return {"version": app.version}


@app.post("/failure")
async def model_failure_detection(prompt: Prompt):
    return trainer.generate(FAILURE_PROMPT.format(prompt.volt, prompt.vibration, prompt.pressure, prompt.rotate))


@app.post("/anomaly")
async def model_anomalous_detection(prompt: Prompt):
    return trainer.generate(ANOMALY_PROMPT.format(prompt.volt, prompt.vibration, prompt.pressure, prompt.rotate))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FastAPI application')
    parser.add_argument('--train-files', type=str, help='Path to the data and labels training files(feather)')
    parser.add_argument('--metadata-file', type=str, help='Path to the metadata file(metadata.json)')
    parser.add_argument('--train', type=bool, help='Required train')
    parser.add_argument('--train-base-model', type=str, help='Base Machine Learning model')
    parser.add_argument('--trained-new-name', type=str, help='Trained model name')

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
        print(f'Loading existing model')
        trainer.load_model(args.trained_new_name)
        print(f'Loaded existing model success')
    import uvicorn

    uvicorn.run(app, host='0.0.0.0', port=8080)
