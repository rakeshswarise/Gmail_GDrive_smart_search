
markdown
Copy
Edit
# 📬 Gmail Smart QA Assistant (Powered by Gemini)

This project is a Streamlit-based application that allows users to ask natural language questions about their Gmail messages. It uses **Google Gemini 1.5 Flash** for keyword extraction and Q&A, and **Gmail API** to fetch email data in real time.

![Gemini QA App](https://img.shields.io/badge/built_with-Streamlit-orange?logo=streamlit)  
![Gemini Model](https://img.shields.io/badge/AI-Gemini_1.5_Flash-blue?logo=google)  
![License](https://img.shields.io/badge/license-MIT-green)

---

## 🚀 Features

- 🔐 Google OAuth2 login via `credentials.json`
- ✨ Natural language keyword extraction using Gemini
- 📥 Gmail message search with semantic understanding
- 🤖 Gemini-powered question answering using only email content
- 🧠 Session-based chat history
- 🪵 In-app log viewer for debugging

---

## 📦 Folder Structure

.
├── .env
├── credentials.json
├── token.json
├── gmail_qa_enhanced.log
├── app.py # Main Streamlit app
└── README.md

yaml
Copy
Edit

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/rakeshswarise/Gmail_GDrive_smart_search.git
cd Gmail_GDrive_smart_search
2. Create and Activate Virtual Environment (Optional but Recommended)
bash
Copy
Edit
python -m venv venv
source venv/bin/activate      # On macOS/Linux
venv\Scripts\activate         # On Windows
3. Install Requirements
bash
Copy
Edit
pip install -r requirements.txt
Create requirements.txt using:
pip freeze > requirements.txt (after all modules are installed)

4. Setup Your Environment
Create a .env file in the root directory with:

env
Copy
Edit
GEMINI_API_KEY=your_google_gemini_api_key
Download your credentials.json from Google Cloud Console and place it in the root folder.
Enable Gmail API and OAuth Consent Screen first.

🧪 Run the App
bash
Copy
Edit
streamlit run app.py
🧰 Gmail API Setup Guide
Go to Google Cloud Console

Enable the Gmail API

Create an OAuth 2.0 Client ID (Desktop)

Download credentials.json

First run will open a browser to authenticate

📌 Example Usage
Enter a question like:

"What did my manager say about the budget in the last 2 weeks?"

App:

Extracts keywords using Gemini

Searches your Gmail

Passes email content to Gemini for answering

Shows clear answers with logs and message previews

⚠️ Notes & Best Practices
.env, .log, token.json, and credentials.json should be added to .gitignore

All Gemini responses are based strictly on email context

You can adjust how many emails are retrieved using the slider

🧾 License
This project is licensed under the MIT License

🙌 Acknowledgements
Google Gemini API

Streamlit

Google Gmail API

Made with ❤️ by Rakesh using Gemini + Gmail

yaml
Copy
Edit

---

Would you also like the `.gitignore` file? Let me know — I can generate that too.