from flask import Flask, request, jsonify
from bs4 import BeautifulSoup
import re
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import openai
import os
import requests

app = Flask(__name__)
openai.api_type = "azure"
openai.api_version = "2023-08-01-preview"
datasets = ["CPTDS1.csv", "CPTDS2.csv", "CPTDS3.csv", "CPTDS4.csv"]

# Azure OpenAI setup
openai.api_base = "https://openai-anazim.openai.azure.com/"
openai.api_key = "f55bb1a1587041fa9cab2da8e4b42e3f"
deployment_id = "chatgpt"

# Azure AI Search setup
search_endpoint = "https://testllmservicev1.search.windows.net"
search_key = "D2RQ7F6UJfmMjgWvR5Mz3pQcA3zqgHb0WLWobavIjzAzSeBNlGOO"
search_index_name = "cptcodess"

script_url = "https://expertsonsite.com/File/script.js"


def get_cpt_codes_from_script(url):
    try:
        response = requests.get(url)
        response.raise_for_status()

        # Extract CPT codes from the JavaScript file using regex
        pattern = r'"([^"]+)":\s*"\s*([A-Za-z\d]+)\s*"'
        matches = re.findall(pattern, response.text)

        cpt_codes = {name: code for name, code in matches}

        return cpt_codes

    except requests.exceptions.RequestException as e:
        print(f"Error fetching CPT codes: {e}")

    return None


def setup_byod(deployment_id: str) -> None:
    class BringYourOwnDataAdapter(requests.adapters.HTTPAdapter):
        def send(self, request, **kwargs):
            request.url = f"{openai.api_base}/openai/deployments/{deployment_id}/extensions/chat/completions?api-version={openai.api_version}"
            return super().send(request, **kwargs)

    session = requests.Session()

    session.mount(prefix=f"{openai.api_base}/openai/deployments/{deployment_id}", adapter=BringYourOwnDataAdapter())

    openai.requestssession = session


setup_byod(deployment_id)


def store_user_feedback(user_input, corrected_cpt_code):
    feedback_file = "user_feedback.csv"
    feedback_df = pd.DataFrame({"Disease": [user_input], "Code": [corrected_cpt_code]})

    if os.path.exists(feedback_file):
        existing_df = pd.read_csv(feedback_file)
        feedback_df = pd.concat([existing_df, feedback_df], ignore_index=True)

    feedback_df.to_csv(feedback_file, index=False)


def check_user_feedback(user_input):
    feedback_file = "user_feedback.csv"
    if not os.path.exists(feedback_file):
        return None

    feedback_df = pd.read_csv(feedback_file)
    matching_row = feedback_df[feedback_df["Disease"].str.lower() == user_input.lower()]

    if not matching_row.empty:
        return matching_row.iloc[0]["Code"]

    return None


def get_cpt_code_from_aapc(disease_name, sources_to_check=None):
    url = "https://www.aapc.com/codes/ajax_search/search"
    params = {
        "keyword": disease_name,
        "idcategory": "",
        "searchtype": "tabcode",
        "search_and": 1,
        "sort": "undefined",
        "searchYear": 2023,  # You can customize the search parameters as needed
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()

        # Extract CPT codes from the JSON response
        cpt_codes = []
        for item in data.get("treeview", []):
            for child in item.get("children", []):
                for html_item in child.get("children", []):
                    cpt_code_element = html_item.get("html", "")
                    cpt_code_match = re.search(r'(\d{5}[A-Z]?)', cpt_code_element)
                    if cpt_code_match:
                        cpt_code = cpt_code_match.group(0)
                        cpt_codes.append(cpt_code)

        if cpt_codes:
            # Check if there are multiple matches from other sources
            if sources_to_check:
                for source, code_to_check in sources_to_check:
                    if code_to_check in cpt_codes:
                        return code_to_check

            # Check if Rule Based code exists in the list
            if "RuleBased" in [source for source, _ in sources_to_check]:
                rule_based_code = next(
                    (code_to_check for source, code_to_check in sources_to_check if source == "RuleBased"), None)
                if rule_based_code in cpt_codes:
                    return rule_based_code

            # Return the first CPT code found (you can adjust this logic if needed)
            return cpt_codes[0]

    print(f"CPT Code not found on AAPC API")
    return None


def get_cpt_code(disease_name):
    system_message = {
        "role": "system",
        "content": "You will strictly be stick to following rules being a medical CPT code expert: "
                   "1- Only answer with CPT code when provided disease name CPT code "
                   "2- Do not provide any other information if CPT code is not there "
                   "3- When the question is out of context, just reply with CPT code not found "
                   "4- Search the CPT code in all files properly and completely "
                   "5- If CPT code is not found suggest the best CPT code from your knowledge",
    }

    user_message = {"role": "user", "content": f"CPT code for {disease_name}?"}

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[system_message, user_message],
        deployment_id=deployment_id,
        dataSources=[
            {
                "type": "AzureCognitiveSearch",
                "parameters": {
                    "endpoint": search_endpoint,
                    "key": search_key,
                    "indexName": search_index_name,
                },
            }
        ],
    )

    cpt_code_result = response.choices[0].message.content

    cpt_code_pattern = r'\b(?:\d{5}|[A-Za-z]\d{4}|\d{4})\b'

    cpt_code_match = re.search(cpt_code_pattern, cpt_code_result)

    if cpt_code_match:
        extracted_cpt_code = cpt_code_match.group(0)
    else:
        extracted_cpt_code = "CPT code not found"

    return extracted_cpt_code


def get_cpt_code_from_google(disease_name):
    url = f"https://www.google.com/search?q=CPT+code+for+{disease_name}"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        # Search bolded text within tables on the page
        table_bold_elements = soup.select('table b')
        for table_bold_element in table_bold_elements:
            table_bold_text = table_bold_element.text.strip()
            cpt_code_match = re.search(r'\b\d{5}\b|\b[A-Za-z]\d{4}\b', table_bold_text)
            if cpt_code_match:
                cpt_code = cpt_code_match.group()
                return cpt_code

        # Search the first link's content, table, or highlighted text
        first_link = soup.select_one('a')
        if first_link:
            # Check highlighted text within the first link
            highlighted_elements = first_link.select('[style*="background-color:yellow"]')
            if highlighted_elements:
                highlighted_text = highlighted_elements[0].text.strip()
                cpt_code_match = re.search(r'\b\d{5}\b|\b[A-Za-z]\d{4}\b', highlighted_text)
                if cpt_code_match:
                    cpt_code = cpt_code_match.group()
                    return cpt_code

            # Check the content and tables within the first link
            link_content = first_link.text.strip()
            link_tables = first_link.select('table')
            for table in link_tables:
                table_text = table.text.strip()
                link_content += f"\n{table_text}"

            cpt_code_match = re.search(r'\b\d{5}\b|\b[A-Za-z]\d{4}\b', link_content)
            if cpt_code_match:
                cpt_code = cpt_code_match.group()
                return cpt_code

        # If no match is found in the first link or tables, proceed to the rest of the page
        result_elements = soup.select('div.tF2Cxc')

        for result_element in result_elements:
            bold_elements = result_element.select('b')
            result_text = bold_elements[0].text.strip() if bold_elements else result_element.text.strip()

            cpt_code_match = re.search(r'\b\d{5}\b|\b[A-Za-z]\d{4}\b', result_text)

            if cpt_code_match:
                cpt_code = cpt_code_match.group()
                return cpt_code

    except requests.exceptions.RequestException as e:
        print(f"CPT Code not found on Google: {e}")

    return None


def get_cpt_code_from_medicallienmgt(disease_name, script_url):
    cpt_codes_from_script = get_cpt_codes_from_script(script_url)

    if cpt_codes_from_script:
        normalized_search_text = disease_name.lower()

        for name, code in cpt_codes_from_script.items():
            if normalized_search_text in name.lower():
                return code

        print(f"CPT Code not found for '{disease_name}' in the script")
    else:
        print("CPT Codes not found from the script")

    return None


def find_best_matching_disease_code(user_input, datasets, tokenizer, model):
    user_feedback_cpt = check_user_feedback(user_input)
    if user_feedback_cpt:
        return user_feedback_cpt

    max_matched_words = 0
    best_matching_disease_code = None

    user_input_tokens = set(tokenizer.tokenize(user_input.lower()))

    feedback_file = "user_feedback.csv"
    if os.path.exists(feedback_file):
        feedback_df = pd.read_csv(feedback_file)
        for _, row in feedback_df.iterrows():
            disease_name, disease_code = row["Disease"], row["Code"]
            if isinstance(disease_name, str):  # Check if it's a string
                disease_tokens = set(tokenizer.tokenize(disease_name.lower()))
                matched_words = len(user_input_tokens.intersection(disease_tokens))

                if matched_words > max_matched_words:
                    max_matched_words = matched_words
                    best_matching_disease_code = disease_code

    if not best_matching_disease_code:
        for dataset in datasets:
            df = pd.read_csv(dataset)
            disease_name_to_code = dict(zip(df["Disease"], df["Code"]))

            for disease_name, disease_code in disease_name_to_code.items():
                if isinstance(disease_name, str):  # Check if it's a string
                    disease_tokens = set(tokenizer.tokenize(disease_name.lower()))
                    matched_words = len(user_input_tokens.intersection(disease_tokens))

                    if matched_words > max_matched_words:
                        max_matched_words = matched_words
                        best_matching_disease_code = disease_code

    return best_matching_disease_code


def get_best_matching_cpt_code_with_feedback(user_input, datasets, tokenizer, model):
    cpt_code_azure_openai = get_cpt_code(user_input)
    cpt_code_google = get_cpt_code_from_google(user_input)
    cpt_code_rule_based = find_best_matching_disease_code(user_input, datasets, tokenizer, model)
    cpt_code_aapc = get_cpt_code_from_aapc(user_input, sources_to_check=[
        ("AzureOpenAI", cpt_code_azure_openai),
        ("Google", cpt_code_google),
        ("RuleBased", cpt_code_rule_based),
    ])
    cpt_code_medicallienmgt = get_cpt_code_from_medicallienmgt(user_input, script_url)

    # Store results in a dictionary
    results = {
        "AzureOpenAI": cpt_code_azure_openai,
        "Google": cpt_code_google,
        "RuleBased": cpt_code_rule_based,
        "AAPC": cpt_code_aapc,
        "Medicallienmgt": cpt_code_medicallienmgt,
    }

    # Filter out null values and empty strings, excluding Google if it's None
    valid_results = {source: cpt_code for source, cpt_code in results.items() if
                     cpt_code is not None and (cpt_code != "" or source != "Google")}

    if not valid_results:
        print(f"No valid CPT codes found for {user_input}")
        return "CPT code not found"

    # Check if more than one result matches
    matching_results = [cpt_code for _, cpt_code in valid_results.items()]
    final_result = max(set(matching_results), key=matching_results.count)

    print("All Results:")
    for source, cpt_code in valid_results.items():
        print(f"{source}: {cpt_code}")

    if final_result == "CPT code not found" or valid_results["RuleBased"] == final_result:
        # If none of the sources match or the rule-based approach is the final result, set final_result to the rule-based code
        final_result = valid_results["RuleBased"]

    print(f"\nThe final CPT code for {user_input} is: {final_result}")

    return final_result


model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


@app.route('/')
def home():
    return 'CPT CODE SUGGESTOR SERVICE - MUHAMMAD UMAR KHAN'


@app.route('/predict_cpt_code', methods=['POST'])
def predict_cpt_code():
    try:
        # Get user input from the request
        user_input = request.json['user_input']

        # Call your updated function to get the best matching CPT code with feedback
        final_result = get_best_matching_cpt_code_with_feedback(user_input, datasets, tokenizer, model)

        # Prepare the response
        response = {"final_result": final_result}

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
