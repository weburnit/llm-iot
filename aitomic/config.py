import torch
import logging

# Create and configure logger
logging.basicConfig(filename="newfile.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')

# Creating an object
logger = logging.getLogger()

# Setting the threshold of logger to DEBUG
logger.setLevel(logging.DEBUG)
HAS_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if HAS_CUDA else 'cpu')

DEVICE_MAP = 'auto'
MODEL = 'google/flan-t5-base'

TRAINING_PARAMS = {
    'max_seq_length': 256,
    'micro_batch_size': 12,
    'gradient_accumulation_steps': 8,
    'epochs': 1,
    'learning_rate': 3e-4,
}

LORA_TRAINING_PARAMS = {
    'lora_r': 8,
    'lora_alpha': 32,
    'lora_dropout': 0.01,
}

GENERATION_PARAMS = {
    'max_new_tokens': 80,
    'temperature': 0.0001,
    'top_k': 40,
    'top_p': 0.3,
    'repetition_penalty': 0.1,
    'do_sample': 'store_true',
    'num_beams': 1,
}

SHARE = 'store_true'

FAILURE_PROMPT = "Detect iot device with volt: {0} vibration: {1} pressure: {2} rotate: {3} \n\n detect failure is True or False"
ANOMALY_PROMPT = "Detect iot device with volt: {0} vibration: {1} pressure: {2} rotate: {3} \n\n detect anomaly is True or False"
