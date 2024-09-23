from transformers import (
    AutoTokenizer,
    AutoConfig, 
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer)
from datasets import load_dataset, DatasetDict, Dataset
from transformers import BertForSequenceClassification

from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig, PeftModelForSequenceClassification
import evaluate
import torch
import numpy as np

import pandas as pd
import numpy as np
from tensorflow.keras.layers import TextVectorization, Embedding, Bidirectional, LSTM, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras import Model, Input
from keras.callbacks import EarlyStopping,ModelCheckpoint
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import re
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from transformers import TextClassificationPipeline
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from accelerate import Accelerator
from torch.utils.data.dataloader import DataLoader
from peft import LoraConfig, TaskType
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from transformers import pipeline, AutoModel, AutoTokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline
import pandas as pd
import numpy as np
import torch.nn.functional as F
import datasets
from tqdm.auto import tqdm
import os
os.environ['CURL_CA_BUNDLE'] = ''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

warnings.filterwarnings("ignore")

#MODEL_NAME = "distilbert-base-uncased"
MODEL_NAME = "bert-base-uncased"
BATCH_SIZE = 4

train_dataset = datasets.load_from_disk("train_gpu_bert-base-uncased.hf")
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
print("Training Data loaded!!")
id2label = {0: "Human", 1: "LLM"}
label2id = {"Human": 0, "LLM": 1}
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label=id2label,
    label2id=label2id
)
print("Model loaded from HuggingFace")
peft_config = LoraConfig(target_modules='all-linear', 
                         task_type=TaskType.SEQ_CLS, 
                         inference_mode=False, 
                         r=2, 
                         lora_alpha=32, 
                         lora_dropout=0.1)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
optimizer = AdamW(model.parameters(), lr=5e-5)
print("Starting training....")
EPOCHS = 2
num_training_steps = EPOCHS * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")
model.to(device)

accelerator = Accelerator()
device = accelerator.device

model, optimizer, train_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader
)

lr_scheduler = accelerator.prepare(lr_scheduler)

num_training_steps = EPOCHS * len(train_dataloader)
progress_bar = tqdm(range(num_training_steps), disable=not accelerator.is_local_main_process)

model.train()
for epoch in range(EPOCHS):
    for batch in train_dataloader:
        # No need to move data to device

        with accelerator.autocast():
            outputs = model(**batch)
            loss = outputs.loss

        # Backward pass
        accelerator.backward(loss)

        # Optimizer step
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        # Update progress bar
        progress_bar.update(1)

print("Training Done!")
accelerator.wait_for_everyone()
unwrapped_model = accelerator.unwrap_model(model)
if accelerator.is_main_process:
    unwrapped_model.save_pretrained(f'Finetuned_Models/GPU_Complete_{MODEL_NAME}_fine_tuned_model', "lora_adapter", save_adapter=True, save_config=True, from_pt=True)

