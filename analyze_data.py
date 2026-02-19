import json
import PyPDF2

# Read PDF instructions
pdf_path = "ultimate_data_science_challenge (1).pdf"
try:
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        print("=== PDF INSTRUCTIONS ===")
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            print(f"\n--- PAGE {page_num + 1} ---")
            print(text)
except Exception as e:
    print(f"Error reading PDF: {e}")

# Read and analyze JSON files
print("\n\n=== LOGINS.JSON STRUCTURE ===")
try:
    with open('logins.json', 'r') as f:
        logins_data = json.load(f)
        print(f"Type: {type(logins_data)}")
        if isinstance(logins_data, list):
            print(f"Number of records: {len(logins_data)}")
            if logins_data:
                print(f"First record keys: {logins_data[0].keys() if isinstance(logins_data[0], dict) else 'N/A'}")
                print(f"Sample first record:\n{json.dumps(logins_data[0], indent=2)}")
        elif isinstance(logins_data, dict):
            print(f"Keys: {logins_data.keys()}")
            sample_key = list(logins_data.keys())[0]
            print(f"Sample entry: {json.dumps({sample_key: logins_data[sample_key]}, indent=2)[:500]}")
except Exception as e:
    print(f"Error reading logins.json: {e}")

print("\n=== ULTIMATE_DATA_CHALLENGE.JSON STRUCTURE ===")
try:
    with open('ultimate_data_challenge.json', 'r') as f:
        challenge_data = json.load(f)
        print(f"Type: {type(challenge_data)}")
        if isinstance(challenge_data, list):
            print(f"Number of records: {len(challenge_data)}")
            if challenge_data:
                print(f"First record keys: {challenge_data[0].keys() if isinstance(challenge_data[0], dict) else 'N/A'}")
                print(f"Sample first record:\n{json.dumps(challenge_data[0], indent=2)[:500]}")
        elif isinstance(challenge_data, dict):
            print(f"Keys: {challenge_data.keys()}")
            for key in list(challenge_data.keys())[:3]:
                print(f"\nSample for key '{key}':")
                val = challenge_data[key]
                if isinstance(val, list):
                    print(f"  List with {len(val)} items")
                    if val and isinstance(val[0], dict):
                        print(f"  First item keys: {val[0].keys()}")
                elif isinstance(val, dict):
                    print(f"  Dict with keys: {val.keys()}")
except Exception as e:
    print(f"Error reading ultimate_data_challenge.json: {e}")
