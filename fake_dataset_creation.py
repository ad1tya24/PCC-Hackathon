from faker import Faker
import random
from datetime import datetime, timedelta

fake = Faker('en_US')

'''
Currently created 30 records for each of the datatypes
- The number of the records can be set individually
- The '-------------------' have been added for testing
- Random text can also be added for later testing 
'''


def create_fake_names (num_of_names) :
    names = []

    for num in range (0, num_of_names) :
        names.append(fake.name())

    return (names)

def create_fake_phones (num_of_phones) :
    phones = []

    for num in range (0, num_of_phones) :
        phones.append(fake.phone_number())

    return (phones)    

def create_fake_dates(num_of_dates):
    date_formats = [
        '%m/%d/%Y',  # MM/DD/YYYY
        '%d-%m-%Y',  # DD-MM-YYYY
        '%Y.%m.%d',  # YYYY.MM.DD
        '%d,%b,%Y',  # DD Mon YYYY
        '%d,%B,%Y'   # DD Month YYYY
    ]
    dates = []
    for i in range(num_of_dates):
        # Generate a random date between today and 10 years ago
        random_date = datetime.today() - timedelta(days=random.randint(0, 3650))
        # Cycle through each date format
        date_format = date_formats[i % len(date_formats)]
        # Format the random date in the chosen format
        formatted_date = random_date.strftime(date_format)
        # Add commas for "DD Month YYYY" format
        if date_format == '%d %B %Y':
            day, month, year = formatted_date.split(' ')
            formatted_date = f'{day}, {month}, {year}'
        dates.append(formatted_date)

    return dates

def create_fake_mails (num_of_mails) :
    mails = []

    for num in range (0, num_of_mails) :
        mails.append(fake.ascii_email())

    return (mails)    

def create_fake_ipv4 (num_of_ipv4) :
    ipv4_addresses = []

    for num in range (0, num_of_ipv4) :
        ipv4_addresses.append(fake.ipv4())

    return (ipv4_addresses)   

def create_fake_ipv6 (num_of_ipv6) :
    ipv6_addresses = []

    for num in range (0, num_of_ipv6) :
        ipv6_addresses.append(fake.ipv6())

    return (ipv6_addresses)   

def create_fake_mac (num_of_mac) :
    mac_addresses = []

    for num in range (0, num_of_mac) :
        mac_addresses.append(fake.mac_address())

    return (mac_addresses)   

def create_fake_urls (num_of_urls) :
    urls = []

    for num in range (0, num_of_urls) :
        urls.append(fake.url())

    return (urls)   

def create_fake_ssn (num_of_ssn) :
    ssn = []

    for num in range (0, num_of_ssn) :
        ssn.append(fake.ssn())

    return (ssn)

def create_fake_health_insurance_beneficiary_numbers(num_of_beneficiary_numbers):
    beneficiary_numbers = []
    for _ in range(num_of_beneficiary_numbers):
        # This will create a random alphanumeric ID of format 'AAA-1234567890'.
        beneficiary_numbers.append(fake.bothify(text='???-##########'))

    return (beneficiary_numbers)   

def create_fake_zipcodes (num_of_zipcodes) :
    zipcodes = []

    for num in range (0, num_of_zipcodes) :
        zipcodes.append(fake.zipcode())

    return (zipcodes)   

def create_fake_mrns (num_of_mrns) :
    # Taking into consideration 3 types of Medical Record Numbers
    mrns = []

    for num in range (0, num_of_mrns) :
        if (num % 3 == 0) :
            mrns.append(fake.bothify('##########'))
        elif (num % 3 == 1) :
            mrns.append(fake.bothify('??-########'))
        else :
            mrns.append(fake.bothify('###-##-####'))

    return (mrns)

def create_fake_fax (num_of_fax) :
    # Taking into consideration North American Standard and International Format
    fax = []

    for num in range (0, num_of_fax) :
        if (num % 2 == 0) :
            fax.append(fake.bothify('(###) ###-####'))
        else :
            fax.append(fake.bothify('+1-###-###-####'))

    return (fax)

def items_to_file(items, file_name):
    with open(file_name, 'a', encoding='utf-8') as file:
        file.write('-------------------' + '\n')
        for item in items:
            file.write(item + '\n')

def print_log_file(file_name):
    with open(file_name, 'r', encoding='utf-8') as file:
        contents = file.read()
        print(contents)

def main () :
    file_name = 'test.log'
    num_of_names = 30
    num_of_phones = 30
    num_of_mails = 30
    num_of_ipv4 = 30
    num_of_ipv6 = 30
    num_of_mac = 30
    num_of_urls = 30
    num_of_ssn = 30
    num_of_zipcodes = 30
    num_of_mrns = 30
    num_of_fax = 30
    num_of_dates= 30
    num_of_insurances= 30

    names = create_fake_names(num_of_names)
    items_to_file(names, file_name)

    phones = create_fake_phones(num_of_phones)
    items_to_file(phones, file_name)

    mails = create_fake_mails(num_of_mails)
    items_to_file(mails, file_name)

    dates = create_fake_dates(num_of_dates)
    items_to_file(dates, file_name)

    beneficiary_numbers = create_fake_health_insurance_beneficiary_numbers(num_of_insurances)
    items_to_file(beneficiary_numbers, file_name)

    ipv4_addresses = create_fake_ipv4(num_of_ipv4)
    items_to_file(ipv4_addresses, file_name)

    ipv6_addresses = create_fake_ipv6(num_of_ipv6)
    items_to_file(ipv6_addresses, file_name)

    mac_addresses = create_fake_mac(num_of_mac)
    items_to_file(mac_addresses, file_name)

    urls = create_fake_urls(num_of_urls)
    items_to_file(urls, file_name)

    ssn = create_fake_ssn(num_of_ssn)
    items_to_file(ssn, file_name)

    zipcodes = create_fake_zipcodes(num_of_zipcodes)
    items_to_file(zipcodes, file_name)

    mrns = create_fake_mrns(num_of_mrns) 
    items_to_file(mrns, file_name)

    fax = create_fake_fax(num_of_fax)
    items_to_file(fax, file_name)

    # Print the log file
    print_log_file(file_name)

def regex_test() : 
    file_name = 'test.log'
    # num_of_mrns = 30

    # mrns = create_fake_mrns(num_of_mrns)
    # items_to_file(mrns, file_name)

if __name__ == "__main__" :
    main()
    # regex_test()