import re

def get_log_data(log_file):
    with open(log_file, 'r', encoding='utf-8') as file:
        log_data = file.readlines()
    return log_data

def file_processing (log_data) :
    # Assumes the EOL is '\n'
    # Assumes the in line delimeter is ' '
    #lines = log_data.split("\n")

    words = []
    for line in log_data : 
        words.extend(line.split())
    
    log_data = words

    return (log_data)

def check_phone (log_data) :
    pattern = re.compile(r"""
        (?<!\w)                             # Negative lookbehind (not preceded by a word character)
        (\+?\d{1,3}[\s-]?)?             # Optional country code (e.g., +1, 001)
        \(?(\d{3,4})\)?[\s.-]?          # Area code with optional parentheses, allowing 3 or 4 digits
        (\d{3})[\s.-]?                  # First part of the number
        (\d{4})                         # Second part of the number
        (?:[\s.-]*(x|\d+)\w{1,5})?            # Optional extension prefixed by 'x' or digits, followed by alphanumeric characters
        \b                               # Word boundary (end of the match)
        """, re.VERBOSE)

    phone_numbers = []
    
    for data in log_data:
        # Check if the line matches the phone number pattern
        if pattern.match(data):
            phone_numbers.append(data)
    
    return (phone_numbers)

def check_dates(log_data):
    date_patterns = [
        r'\b\d{2}/\d{2}/\d{4}\b',  # MM/DD/YYYY
        r'\b\d{2}-\d{2}-\d{4}\b',  # DD-MM-YYYY
        r'\b\d{4}\.\d{2}\.\d{2}\b',  # YYYY.MM.DD
        r'\b\d{2} \w{3} \d{4}\b',  # DD Mon YYYY
        r'\b\d{2} \w+ \d{4}\b',     # DD Month YYYY
        r'\b\d{1,2}\s*,\s*[A-Za-z]+\s*,\s*\d{4}\b'  # D, Month, YYYY and DD, Month, YYYY
    ]
    combined_pattern = '|'.join(date_patterns)
    date_regex = re.compile(combined_pattern, re.IGNORECASE)

    found_dates = []
    for data in log_data:
        matches = date_regex.findall(data)
        found_dates.extend(matches)
    return found_dates

def check_health_insurance_beneficiary(log_data):
    # Pattern to match the format 'AAA-1234567890'
    health_insurance_pattern = re.compile(r"""
        \b[A-Za-z]{3}-\d{10}\b
    """, re.VERBOSE)

    health_insurance_numbers = []
    for data in log_data:
        matches = health_insurance_pattern.findall(data)
        health_insurance_numbers.extend(matches)
    return health_insurance_numbers

def check_mail (log_data) :
    pattern = re.compile(r"""
      [a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+      # Username part
      @                                     # At symbol
      [a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)*     # Domain name
      (?:\.[a-zA-Z]{2,})?                   # Optional top-level domain (e.g., .com, .org)
    """, re.VERBOSE)


    emails = []
    for data in log_data:
        if pattern.search(data): 
            emails.append(data)

    return (emails)

def check_ip (log_data) :
    ipv4_pattern = re.compile(r"(?:[0-9]{1,3}\.){3}[0-9]{1,3}")
    ipv6_pattern = re.compile(r"(?:(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}|(?:[0-9a-fA-F]{1,4}:){1,7}:|:(?::[0-9a-fA-F]{1,4}){1,7}|(?:[0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|(?:[0-9a-fA-F]{1,4}:){1,5}(?::[0-9a-fA-F]{1,4}){1,2}|(?:[0-9a-fA-F]{1,4}:){1,4}(?::[0-9a-fA-F]{1,4}){1,3}|(?:[0-9a-fA-F]{1,4}:){1,3}(?::[0-9a-fA-F]{1,4}){1,4}|(?:[0-9a-fA-F]{1,4}:){1,2}(?::[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:(?::[0-9a-fA-F]{1,4}){1,6}|:(?::[0-9a-fA-F]{1,4}){1,7})")
    mac_pattern = re.compile(r"(?:(?:[0-9a-fA-F]{2}[:-]){5}[0-9a-fA-F]{2})")

    ip_mac = []
    for data in log_data:
        for pattern in [ipv4_pattern, ipv6_pattern, mac_pattern]:
            ip_mac.extend(pattern.findall(data))
    return (ip_mac)

def check_url(log_data):
    pattern = re.compile(r"""
        (https?://)?                         # Optional http or https
        (www\.)?                             # Optional www
        [a-zA-Z0-9.-]+\.[a-zA-Z]{2,}         # Domain name
        (:\d+)?                              # Optional port number
        (/[^\s]*)?                           # Optional path
    """, re.VERBOSE)
    
    urls = []
    for data in log_data:
        if pattern.search(data):
            urls.append(data)
    return (urls)

def check_ssn (log_data) :
    pattern = re.compile(
        r"(?<!\d)(?!666|000|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}(?!\d)"
    )

    ssn_list = []
    for data in log_data:
        if pattern.search(data):
            ssn_list.append(data)
    return (ssn_list)

def check_zipcode (log_data) :
    zip_pattern = re.compile(r"\b(\d{5}(?:-\d{4})?)\b")
    zip_codes = []
    for data in log_data:
        if zip_pattern.search(data):
            zip_codes.append(data)
    return (zip_codes)

def check_mrn (log_data) :
    pattern = re.compile(
        r"\b\d{10}\b|"  # Matches ##########
        r"\b[A-Za-z]{2}-\d{8}\b|"  # Matches ??-########
        r"\b\d{3}-\d{2}-\d{4}\b"  # Matches ###-##-####
    )
    
    mrn_list = []
    for data in log_data:
        matches = pattern.findall(data)
        mrn_list.extend(matches)
    return (mrn_list)

def check_fax (log_data) :
    # Fax number checks for one set of combinations
    # Todo: Improve for the second set of combinations
    fax_pattern = re.compile(
        r"(?<!\d)"                        # Negative lookbehind to ensure the match is not preceded by a digit
        r"(\(\d{3}\)\s\d{3}-\d{4}"        # Matches (###) ###-####
        r"|\+1-\d{3}-\d{3}-\d{4})"        # Matches +1-###-###-####
        r"(?!\d)"                         # Negative lookahead to ensure the match is not followed by a digit
    )

    fax_numbers = []
    for data in log_data:
        if fax_pattern.search(data):
            fax_numbers.append(data)
    return (fax_numbers)

def regex_check_pii (log_data) :
    phone_numbers = check_phone(log_data)
    print(phone_numbers)
    print(len(phone_numbers))

    mails = check_mail(log_data)
    print("Emails:", mails)
    print("Number of emails:", len(mails))

    ip = check_ip(log_data)
    print("IP addresses and MAC addresses:", ip)
    print("Number of IP addresses and MAC addresses:", len(ip))

    urls = check_url(log_data)
    print("URLs:", urls)
    print("Number of URLs:", len(urls))

    ssn = check_ssn(log_data)
    print("Social Security Numbers:", ssn)
    print("Number of Social Security Numbers:", len(ssn))

    zipcode = check_zipcode(log_data)
    print("Zip codes:", zipcode)
    print("Number of zip codes:", len(zipcode))

    mrn = check_mrn(log_data)
    print("Medical Record Numbers:", mrn)
    print("Number of Medical Record Numbers:", len(mrn))

    fax_numbers = check_fax(log_data)
    print("Fax numbers:", fax_numbers)
    print("Number of fax numbers:", len(fax_numbers))

    dates = check_dates(log_data)
    print("Dates:", dates)
    print("Number of dates:", len(dates))

    health_insurance_numbers = check_health_insurance_beneficiary(log_data)
    print("Health Insurance Beneficiary Numbers:", health_insurance_numbers)
    print("Number of Health Insurance Beneficiary Numbers:", len(health_insurance_numbers))
        

def main () :
    log_file = "test.log"
    log_data = get_log_data(log_file)
    log_data = file_processing(log_data)
    print("Log data:", log_data)
    print("Number of log data:", len(log_data))

    regex_check_pii(log_data)

if __name__ == "__main__" :
    main()

