![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![Build Passing](https://img.shields.io/badge/build-passing-brightgreen.svg)
![License](https://img.shields.io/badge/license-proprietary-lightgrey.svg)

---

# Insurance Claim Automation

This project automates the verification and approval process for health insurance claims using AI and Natural Language Processing (NLP). It provides a web interface for users to submit their claims, automatically extracts and analyzes claim data from uploaded medical bills, checks for exclusion criteria, and generates a detailed acceptance or rejection report.

## Features

- **Automated Information Extraction:** Uses Google Gemini AI to extract disease and expense amount from uploaded PDF medical bills.
- **Document Verification:** Checks for the presence of required documents and patient information.
- **Exclusion List Checking:** Compares claim reason against a configurable list of general exclusions (e.g., pre-existing conditions, certain diseases).
- **Claim Validation:** Automatically rejects claims where the claimed amount exceeds the billed amount or if the disease is listed in the general exclusions.
- **Detailed Reports:** Generates a comprehensive report outlining the decision and the reasoning steps, including checks for missing information, document summaries, and potential fraud.
- **Web Interface:** Flask-based web app for submitting insurance claims.

## How It Works

1. **User Submission:** The user fills out a form with personal, claim, and contact details and uploads a PDF medical bill.
2. **AI-Powered Extraction:** The system extracts critical information (disease, expense) from the bill using Gemini AI.
3. **Validation:** The application automatically checks:
   - If all required documents and information are present.
   - If the disease is part of the general exclusions.
   - If the claimed amount does not exceed the billed amount.
4. **Decision Report:** The system renders a detailed acceptance or rejection report for the user.

## Technologies Used

- **Python, Flask:** Backend framework for the web application.
- **LangChain, HuggingFace, FAISS:** For document loading, text splitting, embedding, and retrieval.
- **Google Gemini (generativeai):** For extracting and interpreting information from medical documents.
- **scikit-learn:** Used for text similarity checks.
- **PyPDF2:** For reading PDF files.
- **HTML:** Frontend templates.

## Getting Started

### Prerequisites

- Python 3.8+
- Google Gemini API key
- Required Python packages (see below)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/rasika1205/Insurance-Claim-Automation.git
   cd Insurance-Claim-Automation
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Create `requirements.txt` if it doesn't exist, based on the imports in `app.py`)*

3.  **API Key Setup:**
   - Create a `.env` file in the project root:
     ```bash
     touch .env
     ```
   - Add your Gemini API key to `.env`:
     ```env
     GEMINI_API_KEY=your_key_here
     ```

4. **Run the application:**
   ```bash
   python app.py
   ```

5. **Open in browser:**
   - Go to `http://127.0.0.1:5000/`

### Usage

1. Fill in the claim submission form.
2. Upload the medical bill as a PDF.
3. Submit the form.
4. View the automated decision and detailed report.
   
## Demo
<img width="1868" height="816" alt="Screenshot 2025-07-13 132946" src="https://github.com/user-attachments/assets/b2fc0b14-870e-4618-927c-75db452c3798" />


## Project Structure

- `app.py` - Main application logic and Flask routes.
- `templates/` - HTML templates for web pages.

## Notes

- The exclusion list and claim approval criteria can be customized in `app.py`.
- Make sure to keep your API keys secure.
- Only PDF documents are supported for medical bills.

## License

This project is **proprietary** and protected by copyright © 2025 Rasika Gautam.

You are welcome to view the code for educational or evaluation purposes (e.g., portfolio review by recruiters).  
However, you may **not copy, modify, redistribute, or claim this project as your own** under any circumstances — including in interviews or job applications — without written permission.

---

Feel free to explore the code.

_Developed with 💡 by [Rasika Gautam](https://github.com/rasika1205)_
