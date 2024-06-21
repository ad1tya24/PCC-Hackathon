from faker import Faker
import random

def generate_pii_data_with_filler_text():
    fake = Faker()
    num_of_records = 30
    data_types = data_types = ["name", "phone", "email", "ipv4", "ipv6", "mac", "url", "ssn", "zipcode", "mrn", "fax", "date", "aadhaar", "pan"]

    def create_fake_data(num_of_records, data_type):
        data = []
        for _ in range(num_of_records):
            if data_type == "name":
                data.append(fake.name())
            elif data_type == "phone":
                data.append(fake.phone_number())
            elif data_type == "email":
                data.append(fake.ascii_email())
            elif data_type == "ipv4":
                data.append(fake.ipv4())
            elif data_type == "ipv6":
                data.append(fake.ipv6())
            elif data_type == "mac":
                data.append(fake.mac_address())
            elif data_type == "url":
                data.append(fake.url())
            elif data_type == "ssn":
                data.append(fake.ssn())
            elif data_type == "zipcode":
                data.append(fake.zipcode())
            elif data_type == "date":
                data.append(fake.date())
            elif data_type == "aadhaar":
                data.append(fake.bothify('#### #### ####'))
            elif data_type == "pan":
                data.append(fake.bothify('?????####?').upper())
            elif data_type == "mrn":
                if random.choice([True, False]):
                    data.append(fake.bothify('##########'))
                elif random.choice([True, False]):
                    data.append(fake.bothify('??-########'))
                else:
                    data.append(fake.bothify('###-##-####'))
            elif data_type == "fax":
                if random.choice([True, False]):
                    data.append(fake.bothify('(###) ###-####'))
                else:
                    data.append(fake.bothify('+1-###-###-####'))
            else:
                raise ValueError("Invalid data type provided.")
        return data

    def create_filler_text(num_of_texts):
        texts = []
        for _ in range(num_of_texts):
            texts.append(fake.text())
        return texts

    all_data = []

    for data_type in data_types:
        pii_data = create_fake_data(num_of_records, data_type)
        filler_texts = create_filler_text(num_of_records)

        for pii, text in zip(pii_data, filler_texts):
            all_data.append(pii)
            all_data.append(text)

    return '\n' + '\n\n'.join(all_data)

# Usage
data_string = generate_pii_data_with_filler_text()
#print(data_string)

import torch
from torch.nn.functional import softmax
from transformers import AutoTokenizer, AutoModelForTokenClassification
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from torch.nn.functional import softmax


# Function to detect PII entities in text
def detect_pii_entities(text, tokenizer, model):
    # Tokenize the input data with truncation and convert to PyTorch tensors
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt", return_offsets_mapping=True)
    offset_mapping = inputs.pop('offset_mapping').squeeze()

    # Get the predictions from the model
    outputs = model(**inputs)

    # Apply softmax to get probabilities and decode the highest probability class for each token
    predictions = softmax(outputs.logits, dim=-1)

    # Get the predicted class for each token
    predicted_classes = predictions.argmax(dim=-1)

    # Convert the predicted classes to their corresponding labels
    labels = [model.config.id2label[label_id.item()] for label_id in predicted_classes.squeeze()]

    # Initialize variables to store PII entities and current entity information
    detected_pii = []
    current_entity_tokens = []
    current_label = None
    start_position = None

    # Iterate over each token and its corresponding label and offset
    for idx, (label, word_id) in enumerate(zip(labels, inputs['input_ids'].squeeze().tolist())):
        # Skip special tokens
        if offset_mapping[idx] is None or offset_mapping[idx] == (0, 0):
            continue

        # Check if the token is the beginning of a new entity
        if label.startswith('B-'):
            # Save the previous entity if there is one
            if current_entity_tokens:
                entity = tokenizer.decode(current_entity_tokens, skip_special_tokens=True)
                detected_pii.append({
                    'entity': entity,
                    'type': current_label,
                    'start': offset_mapping[start_position][0],
                    'end': offset_mapping[idx-1][1],
                    'score': torch.max(predictions[0][start_position:idx]).item()
                })
            # Start a new entity
            current_entity_tokens = [word_id]
            current_label = label[2:]  # Remove the 'B-' prefix
            start_position = idx
        # Check if the token is inside the current entity
        elif label.startswith('I-') and current_label == label[2:]:
            current_entity_tokens.append(word_id)
        # Handle tokens outside any entity
        else:
            # Save the previous entity if there is one
            if current_entity_tokens:
                entity = tokenizer.decode(current_entity_tokens, skip_special_tokens=True)
                detected_pii.append({
                    'entity': entity,
                    'type': current_label,
                    'start': offset_mapping[start_position][0],
                    'end': offset_mapping[idx-1][1],
                    'score': torch.max(predictions[0][start_position:idx]).item()
                })
            # Reset current entity information
            current_entity_tokens = []
            current_label = None
            start_position = None

    # Add the last entity if there is one
    if current_entity_tokens:
        entity = tokenizer.decode(current_entity_tokens, skip_special_tokens=True)
        detected_pii.append({
            'entity': entity,
            'type': current_label,
            'start': offset_mapping[start_position][0],
            'end': offset_mapping[-1][1],
            'score': torch.max(predictions[0][start_position:]).item()
        })

    # Post-process detected PII to group phone number tokens
    phone_numbers = []
    for pii in detected_pii:
        if pii['type'] == 'PHONE':
            phone_numbers.append(pii['entity'])

    # Use regular expressions to find complete phone numbers in the original text
    phone_pattern = re.compile(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}(x\d+)?')
    phone_matches = phone_pattern.finditer(text)
    for match in phone_matches:
        # Check if the matched phone number is already in the detected PII
        if match.group() not in phone_numbers:
            detected_pii.append({
                'entity': match.group(),
                'type': 'PHONE_NUMBER',
                'start': match.start(),
                'end': match.end(),
                'score': 1.0  # Assign a high confidence score for regex matches
            })

    return detected_pii

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("obi/deid_bert_i2b2")
model = AutoModelForTokenClassification.from_pretrained("obi/deid_bert_i2b2")

# Split the data into chunks that fit within the model's input length
data_chunks = data_string.split('-------------------\n')

# Detect PII entities in each chunk
all_detected_pii = []
for chunk in data_chunks:
    detected_pii = detect_pii_entities(chunk, tokenizer, model)
    all_detected_pii.extend(detected_pii)

# # Print all detected PII entities
# print(all_detected_pii)

for pii in all_detected_pii:
    if pii['score'] > 0.3:
        print(f"PII: {pii['entity']}, Type: {pii['type']}, Score: {pii['score']:.2f}")
