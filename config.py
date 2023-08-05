import torch

HAS_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if HAS_CUDA else 'cpu')

MODELS = [
    'google/flan-t5-xl'
]

DEVICE_MAP = 'auto'
MODEL = 'cerebras/Cerebras-GPT-2.7B'

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
    'temperature': 0.1,
    'top_k': 40,
    'top_p': 0.3,
    'repetition_penalty': 1.5,
    'do_sample': 'store_true',
    'num_beams': 1,
}

SHARE = 'store_true'
