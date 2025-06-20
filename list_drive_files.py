import os
import re
import base64
import logging
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import google.generativeai as genai
from google.api_core.exceptions import GoogleAPIError # Import for specific API errors

# --- Configuration and Initialization ---

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("❌ GEMINI_API_KEY not found in .env. Please set it to proceed.")
    st.stop()

# Configure Generative AI
genai.configure(api_key=GEMINI_API_KEY)

# Define Gmail API scopes
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

# --- Logging Setup ---
logger = logging.getLogger(__name__) # Use __name__ for better log differentiation
logger.setLevel(logging.INFO)

# Create a file handler for logging
log_file_path = "gmail_qa_enhanced.log"
fh = logging.FileHandler(log_file_path)
fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(fh)

# --- Gmail Authentication ---
@st.cache_resource
def authenticate():
    """Authenticates with Gmail API and returns credentials."""
    creds = None
    if os.path.exists('token.json'):
        try:
            creds = Credentials.from_authorized_user_file('token.json', SCOPES)
        except Exception as e:
            logger.error(f"Error loading token.json: {e}")
            creds = None # Reset creds if file is corrupted

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as e:
                logger.error(f"Error refreshing credentials: {e}")
                st.error("Failed to refresh Gmail credentials. Please re-authenticate.")
                return None
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            try:
                # This opens a browser window for authentication
                creds = flow.run_local_server(port=0)
            except Exception as e:
                logger.error(f"Error running local authentication server: {e}")
                st.error("Failed to authenticate with Gmail. Ensure 'credentials.json' is valid and your browser can open.")
                return None
        if creds:
            with open('token.json', 'w') as token:
                token.write(creds.to_json())
    return creds

# --- Gemini Keyword Extraction ---
def semantic_keywords_gemini(query: str) -> list[str]:
    """
    Extracts important search keywords from a natural language query using Gemini.
    These keywords are then used to search Gmail.

    Args:
        query (str): The user's question.

    Returns:
        list[str]: A list of extracted keywords.
    """
    # Using a fast model for keyword extraction
    model = genai.GenerativeModel("gemini-1.5-flash") # Often sufficient for keyword extraction
    prompt = f'''
    You are an NLP assistant. Extract up to 15 important search keywords or short phrases from this question.
    Focus only on terms that would directly help in searching for relevant emails.
    Do not include stop words or overly general terms unless they are crucial for the query.

    Question: "{query}"

    Return the keywords as a Python list of strings. Example: ["keyword1", "keyword2", "specific phrase"]
    '''
    try:
        response = model.generate_content(prompt)
        # Safely extract the list from the response text using regex and eval
        match = re.search(r'\[.*?\]', response.text, re.DOTALL)
        if match:
            try:
                keywords = eval(match.group(0)) # eval is used here because Gemini is constrained to output a Python list literal
                if isinstance(keywords, list) and all(isinstance(k, str) for k in keywords):
                    logger.info(f"Extracted keywords: {keywords}")
                    return keywords
                else:
                    logger.warning(f"Gemini returned a malformed list or non-string keywords: {keywords}. Raw response: {response.text}")
                    return []
            except (SyntaxError, NameError, TypeError) as e:
                logger.error(f"Failed to parse keywords from Gemini response '{match.group(0)}': {e}")
                return []
        else:
            logger.warning(f"No list found in Gemini keyword extraction response for query: '{query}'. Raw response: {response.text}")
            return []
    except GoogleAPIError as e:
        logger.error(f"Gemini API error during keyword extraction for query '{query}': {e}")
        st.error(f"Error extracting keywords: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error during keyword extraction for query '{query}': {e}")
        return []

# --- Gmail Search Functionality ---
def search_gmail(service, keywords: list[str], full_query: str = "", max_results: int = 20) -> list[str]:
    """
    Searches Gmail for messages matching keywords and the full query.
    Constructs a robust Gmail API query string.

    Args:
        service: Authenticated Gmail API service object.
        keywords (list[str]): List of keywords for search.
        full_query (str): The original full user query to include in search.
        max_results (int): Maximum number of email messages to retrieve.

    Returns:
        list[str]: A list of formatted email texts.
    """
    try:
        query_parts = []
        # Add keywords with OR operator, quoting for exact phrases
        if keywords:
            query_parts.append(" OR ".join(f'"{k.strip()}"' for k in keywords if k.strip()))
        
        # Add the full query as a fallback, ensuring it's not redundant if already in keywords
        if full_query and all(full_query.lower() not in k.lower() for k in keywords):
            query_parts.append(f'"{full_query.strip()}"')

        gmail_query = " ".join(query_parts) if query_parts else ""

        if not gmail_query:
            logger.warning("No effective Gmail search query could be constructed from keywords or full query.")
            return []

        logger.info(f"Gmail search query executed: {gmail_query}")

        # Fetch message IDs
        results = service.users().messages().list(userId='me', q=gmail_query, maxResults=max_results).execute()
        messages = results.get('messages', [])
        logger.info(f"Found {len(messages)} Gmail messages for query.")

        emails = []
        for m in messages:
            try:
                # Fetch full message details
                msg = service.users().messages().get(userId='me', id=m['id'], format='full').execute()
                headers = msg.get('payload', {}).get('headers', [])

                # Extract common headers
                sender = next((h['value'] for h in headers if h['name'] == 'From'), "N/A")
                subject = next((h['value'] for h in headers if h['name'] == 'Subject'), "No Subject")
                
                # Attempt to parse and format date for consistency
                date_str = next((h['value'] for h in headers if h['name'] == 'Date'), "Unknown Date")
                try:
                    # Parse various date formats, then format to a standard one
                    parsed_date = datetime.strptime(date_str, "%a, %d %b %Y %H:%M:%S %z") # Example: Thu, 20 Jun 2025 12:00:00 +0530
                    formatted_date = parsed_date.strftime("%Y-%m-%d %H:%M:%S")
                except ValueError:
                    formatted_date = date_str # Fallback to original if parsing fails

                def extract_body(payload_part):
                    """Recursively extracts plain text body from email payload."""
                    if payload_part.get("mimeType") == "text/plain":
                        data = payload_part['body'].get('data')
                        if data:
                            try:
                                return base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
                            except Exception as decode_e:
                                logger.warning(f"Failed to decode text/plain body part: {decode_e}")
                                return ""
                    elif "parts" in payload_part:
                        for part in payload_part.get("parts", []):
                            nested_body = extract_body(part)
                            if nested_body:
                                return nested_body
                    # Fallback for simple single-part messages without 'parts' key but with data
                    elif payload_part.get("body", {}).get("data") and not payload_part.get("parts"):
                        data = payload_part["body"]["data"]
                        try:
                            return base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
                        except Exception as decode_e:
                            logger.warning(f"Failed to decode fallback body part: {decode_e}")
                            return ""
                    return ""

                body = extract_body(msg['payload'])
                # Concatenate email details into a single string for context
                email_text = f"From: {sender}\nSubject: {subject}\nDate: {formatted_date}\n\n{body}"
                emails.append(email_text)
            except Exception as e:
                logger.error(f"Error processing individual email (ID: {m.get('id', 'N/A')}): {e}")
                # Continue to next email even if one fails
        return emails
    except GoogleAPIError as e:
        logger.error(f"Gmail API error during search operation: {e}")
        st.error(f"Error searching Gmail: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error during Gmail search operation: {e}")
        return []

# --- Gemini Answering Function ---
def ask_gemini(question: str, context: str) -> str:
    """
    Asks Gemini to answer a question based *only* on provided email context.
    The prompt is carefully crafted to limit Gemini's scope.

    Args:
        question (str): The user's question.
        context (str): The concatenated text of relevant Gmail messages.
                       This is the ONLY source of information for Gemini.

    Returns:
        str: Gemini's answer to the question, or a "not found" message.
    """
    model = genai.GenerativeModel("gemini-1.5-flash") # Balances quality and speed
    prompt = f'''
    You are an AI email assistant. Your sole purpose is to answer the user's question
    **ONLY using the information explicitly found within the provided Gmail messages.**
    Do NOT use any outside information, general knowledge, or make assumptions.
    Keep the answer concise and to the point. Aim for a summary if multiple emails are relevant.
    If the answer or any part of the answer cannot be directly and unequivocally
    found in the provided "Gmail Messages" context, you MUST state:
    "Not found in the provided Gmail messages."

    Question: "{question}"

    Gmail Messages:
    \"\"\"
    {context}
    \"\"\"

    Please provide a concise and direct answer based strictly on the context.
    '''
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except GoogleAPIError as e:
        logger.error(f"Gemini API error during answering request: {e}")
        st.error(f"Error generating answer: {e}")
        return "Gemini processing error due to API issue. Please try again."
    except Exception as e:
        logger.error(f"Unexpected error during Gemini answering: {e}")
        return "An unexpected error occurred while processing your request. Please check logs."

# --- Streamlit UI ---
# Set page configuration. Using an emoji directly for icon compatibility.
st.set_page_config(page_title="📧 Gmail Smart QA", layout="centered")
st.title("📬 Gmail Smart QA Assistant (Gemini)")
st.markdown("Ask questions about your Gmail messages and get answers powered by Gemini!")

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input for the query
user_query = st.text_input("Ask any Gmail-related question:", key="user_input_query")

# --- New: Slider for Max Emails ---
# Allow user to set the maximum number of emails to retrieve
max_emails_to_fetch = st.slider(
    "Max emails to search:",
    min_value=1,
    max_value=30, # Increased max for flexibility
    value=15,    # Default to 15, as you mentioned a range around 10-15
    step=1,
    help="Determines how many top relevant emails are fetched from Gmail for analysis."
)
st.info(f"Will fetch up to **{max_emails_to_fetch}** emails for your query.")


# Handle the "Ask" button click
if st.button("Ask", key="ask_button_submit") and user_query:
    if not user_query.strip():
        st.warning("Please enter a question before clicking 'Ask'.")
    else:
        # Step 1: Authenticate and get Gmail service
        with st.spinner("Authenticating with Gmail..."):
            creds = authenticate()
            if not creds:
                st.error("Authentication failed. Please check your `credentials.json` and try again.")
                st.stop() # Stop execution if authentication fails
            gmail_service = build('gmail', 'v1', credentials=creds)

        # Step 2: Extract keywords and search Gmail
        with st.spinner(f"Extracting keywords and fetching up to {max_emails_to_fetch} relevant emails..."):
            keywords = semantic_keywords_gemini(user_query)
            if not keywords:
                keywords = [user_query] # Fallback: use the original query if no keywords are extracted
                logger.info(f"Falling back to full query as keyword due to no semantic keywords found: '{user_query}'")

            # Pass the user-selected max_emails_to_fetch to the search_gmail function
            emails = search_gmail(gmail_service, keywords, full_query=user_query, max_results=max_emails_to_fetch)

            # Concatenate all retrieved email content into a single context string
            # IMPORTANT: For very large email sets, consider summarizing or selecting
            # only the most relevant parts to stay within Gemini's token limits.
            full_context = "\n\n---\n\n".join(emails)

            # For display purposes in Streamlit, we might truncate long contexts
            display_context = full_context[:8000] if len(full_context) > 8000 else full_context

        if emails:
            st.success(f"📧 Loaded {len(emails)} relevant emails based on your query.")
            with st.expander("📝 Raw Email Context Preview (First 8000 characters)"):
                st.text_area("Email Context Provided to Gemini", display_context, height=300, key="email_context_display")
        else:
            st.info("🚫 No Gmail messages matched your query or extracted keywords. Try a different phrasing.")
            # If no emails are found, pass a clear message to Gemini as context
            full_context = "No relevant emails were found for this query in the user's Gmail inbox."

        # Step 3: Ask Gemini to answer based on the context
        with st.spinner("Generating answer with Gemini (using only email context)..."):
            answer = ask_gemini(user_query, full_context)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Update chat history
        st.session_state.chat_history.insert(0, {"role": "ai", "content": f"**Gemini Answer ({timestamp})**:\n{answer}"})
        st.session_state.chat_history.insert(0, {"role": "user", "content": f"**Query**: {user_query}"})

# Display chat history
st.subheader("Conversation History")
if not st.session_state.chat_history:
    st.info("Your conversation history will appear here.")
else:
    for chat_entry in st.session_state.chat_history:
        with st.chat_message(chat_entry["role"]):
            st.markdown(chat_entry["content"])

# Option to clear chat history
if st.session_state.chat_history and st.button("Clear Chat History", key="clear_chat_button"):
    st.session_state.chat_history = []
    st.rerun() # Rerun to clear the displayed history instantly

# --- Debugging and Logging ---
with st.expander("📄 View Logs"):
    if os.path.exists(log_file_path):
        try:
            with open(log_file_path, "r") as f:
                log_content = f.read()
            st.text_area("Log Output", log_content, height=250, key="log_output_area")
        except Exception as e:
            st.error(f"Could not read log file: {e}")
    else:
        st.info("Log file not found yet. Perform some actions to generate logs.")

st.markdown("""
---
*Powered by Google Gemini and Gmail API*
*Ensure `credentials.json` is in the same directory and `GEMINI_API_KEY` is set in `.env`.*
""")