from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_analyzer.nlp_engine import NlpEngineProvider
from faker import Faker
import random
import re

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

# Initialize the NLP engine
nlp_engine_provider = NlpEngineProvider()
nlp_engine = nlp_engine_provider.create_engine()

analyzer = AnalyzerEngine(nlp_engine=nlp_engine)

# Define a more specific pattern for Indian PAN card format with word boundaries
indian_pan_pattern = [Pattern(name="pan_pattern", regex=r'\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b', score=1.0)]

# Define the pattern for Indian Aadhaar card format
indian_aadhaar_pattern = [Pattern(name="aadhaar_pattern", regex=r'\b[2-9]{1}[0-9]{3}\s?[0-9]{4}\s?[0-9]{4}\b', score=1.0)]

# Access the recognizer registry directly
recognizer_registry = analyzer.registry

# Create a new recognizer for Indian Aadhaar numbers
aadhaar_recognizer = PatternRecognizer(supported_entity="IN_AADHAAR", patterns=indian_aadhaar_pattern)

# Add the new recognizer to the registry
recognizer_registry.add_recognizer(aadhaar_recognizer)

# Update the patterns in the existing recognizer for IN_PAN
for rec in recognizer_registry.recognizers:
    if rec.name == 'IN_PAN':
        rec.patterns = indian_pan_pattern

def is_valid_pan(pan):
    # Define the regex pattern for a valid PAN number
    pan_pattern = re.compile(r'\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b')
    # Check if the detected PAN matches the pattern
    return bool(pan_pattern.match(pan))

# Function to check if the detected PII is a valid person name
def is_valid_person(detected_pii):
    # Define a regex pattern to match any special character or number
    pattern = re.compile('[^A-Za-z ]')
    # Search for the pattern in the detected PII
    if pattern.search(detected_pii):
        return False
    return True

results = analyzer.analyze(text=data_string, language='en')

print(len(results))
for result in results:
    detected_pii = data_string[result.start:result.end]
    if result.score < 0.4:
        continue  # Skip this PII as it's below the threshold
    # Check if the entity type is PERSON and if it's valid based on our criteria
    if result.entity_type == 'PHONE_NUMBER' and '.' in detected_pii:
        continue  # Skip this PII as it's not a valid phone number
    if result.entity_type == 'PERSON' and not is_valid_person(detected_pii):
        continue  # Skip printing this PII as it's not a valid person name
    if result.entity_type == 'IN_PAN' and not is_valid_pan(detected_pii):
        continue  # Skip this PII as it's not a valid PAN number
    # Print all detected PII information
    print(f"PII: {detected_pii}, Type: {result.entity_type}, Score: {result.score}")
