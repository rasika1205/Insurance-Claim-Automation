import os, re
from flask import Flask, render_template, request
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from PyPDF2 import PdfReader
import google.generativeai as genai
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.document_loaders import DirectoryLoader
from langchain.chains import LLMChain
import json
api_key="your key"
genai.configure(api_key=api_key)
if api_key is None or api_key == "":
    print("OpenAI API key not set or empty. Please set the environment variable.")
    exit()  # Terminate the program if the API key is not set.
FAISS_PATH = "/faiss"
# Flask App
app = Flask(__name__)

vectorstore = None
conversation_chain = None
chat_history = []
general_exclusion_list = ["HIV/AIDS", "Parkinson's disease", "Alzheimer's disease","pregnancy", "substance abuse", "self-inflicted injuries", "sexually transmitted diseases(std)", "pre-existing conditions"]
def get_document_loader():
    loader = DirectoryLoader('documents', glob="**/*.pdf", show_progress=True, loader_cls=PyPDFLoader)
    docs = loader.load()
    return docs

def get_text_chunks(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(

        chunk_size=1000,
        chunk_overlap=200,
        length_function=len

    )
    chunks = text_splitter.split_documents(documents)
    return chunks
def get_embeddings():
    documents = get_document_loader()
    chunks = get_text_chunks(documents)
    db = FAISS.from_documents(
        chunks, HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    )
    return db
def get_retriever():
    db = get_embeddings()
    retriever = db.as_retriever()
    return retriever

def get_claim_approval_context():
    return (
        "Claims must include: patient name, address, claim reason, medical bill. "
        "Patient ID is not mandatory."
        "The claimed amount cannot exceed the bill amount. "
        "Diagnosis must match claim reason. "
        "Prescription is optional so don't reject the claim if the presciption is not provided."
        "If a phone number is present in the medical bill, it is sufficient for verification. "
    )

def get_general_exclusion_context():
    return (
        "Excluded conditions: HIV/AIDS, Parkinson's disease, Alzheimer's disease, pregnancy, "
        "substance abuse, self-inflicted injuries, sexually transmitted diseases (STD), pre-existing conditions."
    )

def get_file_content(file):
    text = ""
    if file.filename.endswith(".pdf"):
        pdf = PdfReader(file)
        for page_num in range(len(pdf.pages)):
            page = pdf.pages[page_num]
            text += page.extract_text()

    return text


model = genai.GenerativeModel("gemini-1.5-flash")


def get_bill_info(data):
    """
    Extracts 'disease' and 'expense amount' from medical invoice text using Gemini API.
    """
    prompt = (
        "Act as an expert in extracting information from medical invoices.\n"
        "You are given the invoice details of a patient.\n"
        "Go through the given document carefully and extract the 'disease' and the 'expense amount' from the data.\n"
        "Return the data in this JSON format exactly:\n"
        "{'disease': '', 'expense': ''}\n\n"
        f"INVOICE DETAILS:\n{data}"
    )

    # Generate response
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.4,
            max_output_tokens=500,
        )
    )

    # Gemini returns a `text` field with the output string
    text_response = response.text.strip()
    # Remove backticks and 'json' label if present
    text_response = text_response.strip("`").strip()
    if text_response.startswith("json"):
        text_response = text_response[len("json"):].strip()

    # Replace single quotes with double quotes
    json_str = text_response.replace("'", '"')
    # Attempt to parse the JSON
    try:
        # Sometimes Gemini may use single quotes, so replace with double quotes
        extracted_data = json.loads(json_str)
    except Exception as e:
        raise ValueError(f"Failed to parse JSON from Gemini response:\n{text_response}\nError: {e}")

    return extracted_data
PROMPT = """You are an AI assistant for verifying health insurance claims. You are given with the references for approving the claim and the patient details. Analyse the given data and predict if the claim should be accepted or not. Use the following guidelines for your analysis.

1.Verify if the patient has provided all necessary information and all necessary documents
and if you find any incomplete information or required documents are not provided then set INFORMATION criteria as FALSE and REJECT the claim.
if patient has provided all required documents then set INFORMATION criteria as TRUE. 

2. If any disease mentioned in the medical bill of the patient is in the general exclusions list, set EXCLUSION criteria as FALSE and REJECT the claim.

Use this information to verify if the application is valid and to accept or reject the application.

DOCUMENTS FOR CLAIM APPROVAL: {claim_approval_context}
EXCLUSION LIST : {general_exclusion_context}
PATIENT INFO : {patient_info}
MEDICAL BILL : {medical_bill_info}

Use the above information to verify if the application is valid and decide if the application has to be accepted or rejected keeping the guidelines into consideration. 

Generate a detailed report about the claim and procedures you followed for accepting or rejecting the claim and the write the information you used for creating the report. 
Create a report in the following format

Write whether INFORMATION AND EXCLUSION are TRUE or FALSE 
Reject the claim if any of them is FALSE.
Write whether claim is accepted or not. If the claim has been accepted, the maximum amount which can be approved will be {max_amount}

Executive Summary
[Provide a Summary of the report.]

Introduction
[Write a paragraph about the aim of this report, and the state of the approval.]

Claim Details
[Provide details about the submitted claim]

Claim Description
[Write a short description about claim]

Document Verification
[Mentions which documents are submitted and if they are verified.] 

Document Summary
[Give a summary of everything here including the medical reports of the patient]

Please verify for any signs of fraud in the submitted claim if you find the documents required for accepting the claim for the medical treatment.
"""


prompt = PromptTemplate(input_variables=["claim_approval_context", "general_exclusion_context", "patient_info","max_amount"], template=PROMPT)


def check_claim_rejection(claim_reason, general_exclusion_list, prompt_template, threshold=0.4):
    vectorizer = CountVectorizer()
    patient_info_vector = vectorizer.fit_transform([claim_reason])

    for disease in general_exclusion_list:
        disease_vector = vectorizer.transform([disease])
        similarity = cosine_similarity(patient_info_vector, disease_vector)[0][0]
        if float(similarity) > float(threshold):
            prompt_template = """You are an AI assistant for verifying health insurance claims. You are given with the references for approving the claim and the patient details. Analyse the given data and give a good rejection. You the following guidelines for your analysis.
            PATIENT INFO : {patient_info}

            Executive Summary
                [Provide a Summary of the report.]

                Introduction
                [Write a paragraph about the aim of this report, and the state of the approval.]

                Claim Details
                [Provide details about the submitted claim]

                Claim Description
                [Write a short description about claim]

                Document Verification
                [Mentions which documents are submitted and if they are verified.] 

                Document Summary
                [Give a summary of everything here including the medical reports of the patient]

            CLAIM MUST BE REJECTED: Patient has {disease} which is present in the general exclusion list."""
            return prompt_template

    return prompt_template
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=['GET', 'POST'])
def msg():
    claim_validation_message = ""

    # Get form data
    name = request.form['name']
    address = request.form['address']
    claim_type = request.form['claim_type']
    claim_reason = request.form['claim_reason']
    date = request.form['date']
    medical_facility = request.form['medical_facility']
    medical_bill = request.files['medical_bill']
    total_claim_amount = request.form['total_claim_amount']
    description = request.form['description']
    phone_no = request.form['phone_no']

    # Read medical bill text (assumes a get_file_content function exists)
    bill = get_file_content(medical_bill)

    # Extract info using Gemini
    bill_info = get_bill_info(bill)

    # Case 1: Claim amount is more than bill amount → Reject
    if bill_info['expense'] and int(bill_info['expense']) < int(total_claim_amount):
        claim_validation_message = (
            "The amount mentioned for claiming is more than the billed amount. Claim Rejected."
        )
        return render_template("result.html",
                               name=name, address=address, claim_type=claim_type,
                               claim_reason=claim_reason, date=date,
                               medical_facility=medical_facility,
                               total_claim_amount=total_claim_amount,
                               description=description, phone_no=phone_no, output=claim_validation_message
                               )

    # Case 2: Bill amount >= claim → Perform exclusion check
    elif bill_info['expense'] and int(bill_info['expense']) >= int(total_claim_amount):
        patient_info = (
            f"Name: {name}\nAddress: {address}\nClaim type: {claim_type}\n"
            f"Claim reason: {claim_reason}\nMedical facility: {medical_facility}\n"
            f"Phone Number:{phone_no}\n"
            f"Date: {date}\nTotal claim amount: {total_claim_amount}\nDescription: {description}"
            f"Note: No separate prescription file was provided. The medical bill includes phone number but no ID."
        )
        medical_bill_info = f"Medical Bill: {bill}"

        validated_prompt = check_claim_rejection(bill_info["disease"], general_exclusion_list, PROMPT)

        # Use Gemini to generate claim decision
        full_prompt = (
            f"{validated_prompt}\n\n"
            f"PATIENT INFO:\n{patient_info}\n\n"
            f"DOCUMENTS FOR CLAIM APPROVAL: {get_claim_approval_context()}\n"
            f"EXCLUSION LIST: {get_general_exclusion_context()}\n"
            f"{medical_bill_info}\n"
            f"Maximum claimable amount: {total_claim_amount}\n"
        )

        response = model.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.4, max_output_tokens=500)
        )
        output = response.text.strip()
        output = re.sub(r'\n', '<br>', output)

        return render_template("result.html",
                               name=name, address=address, claim_type=claim_type,
                               claim_reason=claim_reason, date=date,
                               medical_facility=medical_facility,
                               total_claim_amount=total_claim_amount,
                               description=description,phone_no=phone_no, output=output
                               )

    # Case 3: No valid expense extracted
    else:
        output = "Please enter a valid Consultation Receipt."
        return render_template("result.html",
                               name=name, address=address, claim_type=claim_type,
                               claim_reason=claim_reason, date=date,
                               medical_facility=medical_facility,
                               total_claim_amount=total_claim_amount,
                               description=description,phone_no=phone_no, output=output
                               )
app.run(debug=True)