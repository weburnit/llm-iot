import os
import pandas as pd
import gc
import transformers
import peft
import datasets
from contextlib import nullcontext
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from aitomic.config import *


class IoTModel(object):
    def __init__(self):
        self.model = None
        self.model_name = None
        self.lora_name = None
        self.loras = {}
        self.dataset = None

        self.tokenizer = None
        self.trainer = None

        self.feather_files = None  # ['sample_data/iot_pmfp_data.feather', 'sample_data/iot_pmfp_labels.feather']
        self.metadata_file = None  # 'sample_data/metadata.json'
        self.label_col = 'failure'

    @property
    def data(self):
        return self.dataset

    def set_data(self, data):
        self.dataset = data

    def load_data(self):
        if self.dataset is None:
            self.dataset = self.prepare_dataset(load_data(self.feather_files, self.metadata_file, self.label_col))

    def train(self, feather_files, metadata_file, new_peft_model_name, **kwargs):
        """Train new model"""
        assert self.model is not None
        assert self.model_name is not None
        assert self.tokenizer is not None
        self.feather_files = feather_files
        self.metadata_file = metadata_file
        self.load_data()

        kwargs = {**TRAINING_PARAMS, **LORA_TRAINING_PARAMS, **kwargs}

        self.lora_name = None
        self.loras = {}

        if hasattr(self.model, 'disable_adapter'):
            self.load_model(self.model_name, force=True)

        self.model = peft.prepare_model_for_kbit_training(self.model)
        self.model = peft.get_peft_model(self.model, peft.LoraConfig(
            r=kwargs['lora_r'],
            lora_alpha=kwargs['lora_alpha'],
            lora_dropout=kwargs['lora_dropout'],
            bias="none",
            task_type="CAUSAL_LM",
        ))

        if not os.path.exists('../lora'):
            os.makedirs('../lora')

        sanitized_model_name = self.model_name.replace('/', '_').replace('.', '_')
        output_dir = f"aitomic/lora/{sanitized_model_name}_{new_peft_model_name}"

        training_args = transformers.TrainingArguments(
            per_device_train_batch_size=kwargs['micro_batch_size'],
            gradient_accumulation_steps=kwargs['gradient_accumulation_steps'],
            num_train_epochs=kwargs['epochs'],
            learning_rate=kwargs['learning_rate'],
            fp16=False,
            optim='adamw_torch',
            logging_steps=20,
            save_total_limit=3,
            output_dir=output_dir,
        )

        self.trainer = transformers.Trainer(
            model=self.model,
            args=training_args,
            data_collator=transformers.DataCollatorForLanguageModeling(
                self.tokenizer,
                mlm=False,
            ),
            train_dataset=self.dataset['train'],
            eval_dataset=self.dataset['validation']
        )

        self.model.config.use_cache = False
        result = self.trainer.train(resume_from_checkpoint=False)

        return result

    def generate(self, prompt, **kwargs):
        """Generate prompt by model"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded before generation.")

        kwargs = {**GENERATION_PARAMS, **kwargs}
        inputs = self._prepare_inputs_for_generation(prompt)
        kwargs = self._prepare_kwargs_for_generation(kwargs)

        disable_lora = self._get_lora_context()
        with torch.no_grad(), disable_lora:
            output = self._generate_output(inputs, kwargs)

        return self.tokenizer.decode(output, skip_special_tokens=True).strip()

    def unload_model(self):
        self._reset_model_and_tokenizer()
        self._clear_memory()

    def load_lora(self, lora_name, replace_model=True):
        if self.model is None or lora_name is None:
            raise ValueError("Model and LORA name must be specified.")
        if lora_name == self.lora_name:
            return
        if lora_name in self.loras:
            self._set_current_lora(lora_name)
            return

        peft_config = peft.PeftConfig.from_pretrained(lora_name)
        if not replace_model:
            assert peft_config.base_model_name_or_path == self.model_name

        if peft_config.base_model_name_or_path != self.model_name:
            self.load_model(peft_config.base_model_name_or_path)

        self._load_adapter_from_config(lora_name)
        self._set_current_lora(lora_name)

    def unload_lora(self):
        self.lora_name = None

    def tokenize_sample(self, item, max_seq_length, add_eos_token=True):
        assert self.tokenizer is not None
        result = self.tokenizer(
            item["text"],
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
        )

        result = {
            "input_ids": result["input_ids"][:-1],
            "attention_mask": result["attention_mask"][:-1],
        }

        if (
                result["input_ids"][-1] != self.tokenizer.eos_token_id
                and len(result["input_ids"]) < max_seq_length
                and add_eos_token
        ):
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        return result

    def tokenize_training_text(self, training_text, max_seq_length, separator="\n\n\n", **kwargs):
        samples = training_text.split(separator)
        samples = [x.strip() for x in samples]

        def to_dict(text):
            return {'text': text}

        samples = [to_dict(x) for x in samples]

        training_dataset = datasets.Dataset.from_list(samples)
        training_dataset = training_dataset.shuffle().map(
            lambda x: self.tokenize_sample(x, max_seq_length),
            batched=False
        )

        return training_dataset

    def prepare_dataset(self, dataframe):
        train, test = train_test_split(dataframe, test_size=0.2, random_state=42)  # 80% for training
        test, val = train_test_split(test, test_size=0.5, random_state=42)  # 10% for test and validation

        dataset = DatasetDict(
            {
                "train": Dataset.from_pandas(train),  # 80%
                "test": Dataset.from_pandas(test),  # 10%
                "validation": Dataset.from_pandas(val)  # 10%
            }
        )

        self.dataset = dataset

        # Define a function to encode the data
        def encode(examples):
            # Tokenize the data
            examples['input_ids'] = self.tokenizer(examples['signal'], truncation=True, padding="max_length",
                                                   return_tensors="pt").input_ids

            examples['labels'] = self.tokenizer(examples['failure'], truncation=True, padding="max_length",
                                                return_tensors="pt").input_ids

            return examples

        # Encode the dataset
        dataset = dataset.map(encode, batched=True)
        dataset = dataset.remove_columns(['signal', 'failure', ])

        # Format the dataset to PyTorch tensors
        dataset.set_format(type='torch', columns=['input_ids', 'labels'])

        return dataset

    def _reset_model_and_tokenizer(self):
        self.model = None
        self.model_name = None
        self.lora_name = None
        self.tokenizer = None

    def _clear_memory(self):
        if HAS_CUDA:
            with torch.no_grad():
                torch.cuda.empty_cache()
        gc.collect()

    def load_model(self, model_name, force=False, **kwargs):
        if not force and model_name == self.model_name:
            return

        self.unload_model()
        self._initialize_model_and_tokenizer(model_name)

    def _initialize_model_and_tokenizer(self, model_name):
        self.model = self._create_model_from_pretrained(model_name)
        self.tokenizer = self._create_tokenizer_from_pretrained(model_name)
        self.model_name = model_name
        self.loras = {}

    def _create_model_from_pretrained(self, model_name):
        return transformers.AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            device_map=DEVICE_MAP,
            load_in_8bit=True,
            torch_dtype=torch.half,
        )

    def _create_tokenizer_from_pretrained(self, model_name):
        if model_name.startswith('decapoda-research/llama'):
            tokenizer = transformers.LlamaTokenizer.from_pretrained(model_name)
        else:
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token_id = 0
        return tokenizer

    def _set_current_lora(self, lora_name):
        self.lora_name = lora_name
        self.model.set_adapter(lora_name)

    def _load_adapter_from_config(self, lora_name):
        if hasattr(self.model, 'load_adapter'):
            self.model.load_adapter(lora_name, adapter_name=lora_name)
        else:
            self.model = peft.PeftModel.from_pretrained(self.model, lora_name, adapter_name=lora_name)
        if self.model_name.startswith('cerebras'):
            self.model.half()
        self.loras[lora_name] = True

    def _prepare_inputs_for_generation(self, prompt):
        inputs = self.tokenizer(str(prompt), return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)
        return input_ids

    def _prepare_kwargs_for_generation(self, kwargs):
        if self.model.config.pad_token_id is None:
            kwargs['pad_token_id'] = self.model.config.eos_token_id
        if kwargs['do_sample']:
            del kwargs['num_beams']

        generation_config = transformers.GenerationConfig(use_cache=False, **kwargs)
        return generation_config

    def _get_lora_context(self):
        if self.lora_name is None and hasattr(self.model, 'disable_adapter'):
            return self.model.disable_adapter()
        else:
            return nullcontext()

    def _generate_output(self, inputs, kwargs):
        return self.model.generate(
            input_ids=inputs,
            attention_mask=torch.ones_like(inputs),
            generation_config=kwargs
        )[0].to(self.model.device)


def load_data(feather_files, metadata_file):
    data = load_dataframe(feather_files, metadata_file)

    data = data.assign(
        signal=data.apply(
            lambda
                row: f"Detect iot device with volt: {row['volt']} vibration: {row['vibration']} pressure: {row['pressure']} rotate: {row['rotate']} \n\n detect failure: {row['failure']} anomaly: {row['anomaly']}",
            axis=1))
    data = data.assign(failure=data.apply(lambda row: f"failure {row['failure']} and anomaly {row['anomaly']}", axis=1))
    drop_cols = [col for col in data.columns if col not in ['signal', 'failure']]
    data = data.drop(columns=drop_cols)

    return data


def load_dataframe(feather_files, metadata_file):
    # Load and merge data
    data_frames = [pd.read_feather(f) for f in feather_files.split(',')]
    data = pd.merge(data_frames[0], data_frames[1], on=["machineID", "datetime", "anomaly"])

    return data
