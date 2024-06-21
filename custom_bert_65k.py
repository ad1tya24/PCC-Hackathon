from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
dataset = load_dataset("ai4privacy/pii-masking-65k")
train_data = dataset["train"]
train_df = pd.DataFrame(train_data)

# Split the data
train_df, test_df = train_test_split(train_df, test_size=0.2, random_state=42)

print("Training data size:", len(train_df))
print("Testing data size:", len(test_df))

from transformers import BertTokenizer

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

# Tokenize function
def tokenize_and_preserve_labels(sentence, text_labels):
    tokenized_sentence = []
    labels = []

    for word, label in zip(sentence, text_labels):
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)

        tokenized_sentence.extend(tokenized_word)
        labels.extend([label] * n_subwords)
    
    return tokenized_sentence, labels

# Tokenize the training data
train_tokenized_texts_and_labels = [
    tokenize_and_preserve_labels(sent, labs) 
    for sent, labs in zip(train_df['tokenised_unmasked_text'], train_df['token_entity_labels'])
]

# Tokenize the testing data
test_tokenized_texts_and_labels = [
    tokenize_and_preserve_labels(sent, labs) 
    for sent, labs in zip(test_df['tokenised_unmasked_text'], test_df['token_entity_labels'])
]

import torch
from torch.utils.data import DataLoader, TensorDataset

MAX_LEN = 128
BATCH_SIZE = 32

def pad_and_truncate(tokens, labels, max_len):
    tokens = tokens[:max_len - 2]
    labels = labels[:max_len - 2]

    tokens = ['[CLS]'] + tokens + ['[SEP]']
    labels = ['O'] + labels + ['O']

    padding_length = max_len - len(tokens)
    tokens += ['[PAD]'] * padding_length
    labels += ['O'] * padding_length

    return tokens, labels

# Convert tokens and labels to input IDs and attention masks
def convert_to_input_ids_and_attention_masks(tokenized_texts_and_labels, max_len):
    input_ids = []
    attention_masks = []
    labels = []

    for tokenized_text, label in tokenized_texts_and_labels:
        tokenized_text, label = pad_and_truncate(tokenized_text, label, max_len)
        input_id = tokenizer.convert_tokens_to_ids(tokenized_text)
        attention_mask = [int(token_id != 0) for token_id in input_id]

        input_ids.append(input_id)
        attention_masks.append(attention_mask)
        labels.append([label_map[l] for l in label])

    return torch.tensor(input_ids), torch.tensor(attention_masks), torch.tensor(labels)

# Create label map
label_list = list(set([label for labels in train_df['token_entity_labels'] for label in labels]))
label_map = {label: i for i, label in enumerate(label_list)}

# Convert train data
train_inputs, train_masks, train_labels = convert_to_input_ids_and_attention_masks(train_tokenized_texts_and_labels, MAX_LEN)
# Convert test data
test_inputs, test_masks, test_labels = convert_to_input_ids_and_attention_masks(test_tokenized_texts_and_labels, MAX_LEN)

# Create data loaders
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE)

test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)

from transformers import BertForTokenClassification, AdamW

# Define the model
model = BertForTokenClassification.from_pretrained(
    'bert-base-cased',
    num_labels=len(label_list)
)

# Move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

from transformers import get_linear_schedule_with_warmup

EPOCHS = 3
total_steps = len(train_dataloader) * EPOCHS

# Create optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=3e-5, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        model.zero_grad()
        outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_loss / len(train_dataloader)
    print(f'Epoch {epoch + 1}/{EPOCHS} - Training loss: {avg_train_loss}')

from seqeval.metrics import classification_report

# Evaluation loop
model.eval()
predictions, true_labels = [], []

for batch in test_dataloader:
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask, b_labels = batch

    with torch.no_grad():
        outputs = model(b_input_ids, attention_mask=b_input_mask)
    
    logits = outputs.logits
    logits = torch.argmax(logits, dim=2)

    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()

    for i in range(len(logits)):
        pred_labels = [list(label_map.keys())[p] for p in logits[i] if p != -100]
        true_labels_batch = [list(label_map.keys())[l] for l in label_ids[i] if l != -100]

        predictions.append(pred_labels)
        true_labels.append(true_labels_batch)

print(classification_report(true_labels, predictions))
