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
from google.api_core.exceptions import GoogleAPIError
from bs4 import BeautifulSoup

# --- Configuration and Initialization ---
print("--- Application Initialization Started ---") # Added print statement

# Load environment variables from .env file
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("❌ GEMINI_API_KEY not found in .env. Please set it to proceed.")
    print("❌ ERROR: GEMINI_API_KEY not found in .env. Stopping application.") # Added print statement
    st.stop()
else:
    print("✅ GEMINI_API_KEY loaded successfully.") # Added print statement

# Configure Generative AI
genai.configure(api_key=GEMINI_API_KEY)
print("✅ Generative AI configured.") # Added print statement

# Define Gmail API scopes
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
print(f"Gmail API SCOPES defined: {SCOPES}") # Added print statement

# Max number of emails whose content will be passed to Gemini for contextual answering.
TOP_K_EMAILS_FOR_GEMINI = 250
print(f"Max emails for Gemini context set to: {TOP_K_EMAILS_FOR_GEMINI}") # Added print statement

# Define the maximum context length for Gemini (MOVED TO GLOBAL SCOPE)
MAX_CONTEXT_LENGTH = 500000 
print(f"Max context length for Gemini set to: {MAX_CONTEXT_LENGTH} characters.") # Added print statement

# --- Logging Setup ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a unique log file for each session
timestamp_log = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, f"gmail_qa_enhanced_{timestamp_log}.log")
print(f"Logging configured. Log file path: {log_file_path}") # Added print statement

# Clear existing handlers to prevent duplicate logs in Streamlit reruns
if logger.handlers:
    print("Clearing existing logger handlers.") # Added print statement
    for handler in logger.handlers:
        logger.removeHandler(handler)
        handler.close()

# Add file handler with explicit UTF-8 encoding and error replacement
fh = logging.FileHandler(log_file_path, encoding='utf-8', errors='replace')
fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(fh)
print("File logger handler added.") # Added print statement

# File paths for downloads (unique for each session)
timestamp_data_dump = datetime.now().strftime("%Y%m%d_%H%M%S")
email_dump_path = os.path.join(log_dir, f"all_emails_dump_{timestamp_data_dump}.txt")
keyword_matches_file = os.path.join(log_dir, f"keyword_matches_dump_{timestamp_data_dump}.txt")
print(f"Download paths set: Email dump -> {email_dump_path}, Keyword matches -> {keyword_matches_file}") # Added print statement

# --- Gmail Authentication ---
@st.cache_resource
def authenticate():
    print("Attempting Gmail authentication...") # Added print statement
    creds = None
    if os.path.exists('token.json'):
        print("Found 'token.json'. Attempting to load credentials.") # Added print statement
        try:
            creds = Credentials.from_authorized_user_file('token.json', SCOPES)
            logger.info("Loaded token.json for Gmail access.")
            print("✅ Loaded token.json for Gmail access.") # Added print statement
        except Exception as e:
            logger.error(f"Error loading token.json: {e}. Attempting re-authentication.")
            print(f"❌ ERROR loading token.json: {e}. Attempting re-authentication.") # Added print statement
            creds = None

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("Credentials expired. Attempting to refresh token.") # Added print statement
            try:
                creds.refresh(Request())
                logger.info("Gmail credentials refreshed.")
                print("✅ Gmail credentials refreshed.") # Added print statement
            except Exception as e:
                logger.error(f"Error refreshing credentials: {e}")
                print(f"❌ ERROR refreshing credentials: {e}") # Added print statement
                st.error("Failed to refresh Gmail credentials. Please re-authenticate.")
                return None
        else:
            print("No valid credentials found or token missing. Initiating new authentication flow.") # Added print statement
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            try:
                st.info("A browser window will open for Google authentication. Please complete the process and authorize access.")
                print("INFO: Opening browser for Google authentication. Please authorize access.") # Added print statement
                creds = flow.run_local_server(port=0)
                print("✅ Google authentication completed.") # Added print statement
            except FileNotFoundError:
                st.error("Failed to find 'credentials.json'. Please ensure it's in the same directory.")
                print("❌ ERROR: 'credentials.json' not found. Please ensure it's in the same directory.") # Added print statement
                return None
            except Exception as e:
                logger.error(f"Error during Google OAuth authentication: {e}")
                st.error(f"Failed to authenticate with Gmail. Ensure 'credentials.json' is valid and your browser can open. Error: {e}")
                print(f"❌ ERROR during Google OAuth authentication: {e}") # Added print statement
                return None
        if creds:
            with open('token.json', 'w') as token:
                token.write(creds.to_json())
            print("✅ New token saved to 'token.json'.") # Added print statement
    
    if creds:
        print("Returning Gmail service build object.") # Added print statement
        return build('gmail', 'v1', credentials=creds)
    print("Authentication failed. No Gmail service object returned.") # Added print statement
    return None

# --- Email Content Extraction ---
def extract_email_body(payload_part):
    """Recursively extracts plain text body from email payload."""
    # This function is called many times, so print statements here would be verbose.
    # Logging messages are already present for warnings/errors during decoding.
    if payload_part.get("mimeType") == "text/plain":
        data = payload_part['body'].get('data')
        if data:
            try:
                return base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
            except Exception as decode_e:
                logger.warning(f"Failed to decode text/plain body part: {decode_e}")
                return ""
    elif payload_part.get("mimeType") == "text/html":
        data = payload_part['body'].get('data')
        if data:
            try:
                decoded_html = base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
                return BeautifulSoup(decoded_html, "html.parser").get_text(separator="\n")
            except Exception as decode_e:
                logger.warning(f"Failed to decode text/html body part: {decode_e}")
                return ""
    elif "parts" in payload_part:
        for part in payload_part.get("parts", []):
            nested_body = extract_email_body(part)
            if nested_body:
                return nested_body
    return ""

# --- Helper Functions ---
def count_keyword_occurrences(email_texts, keyword):
    """Counts non-overlapping occurrences of a keyword in a list of texts."""
    print(f"Counting occurrences of keyword: '{keyword}' in email texts.") # Added print statement
    keyword_lower = keyword.lower()
    flat_text = "\n".join(email_texts).lower()
    # Use re.findall with word boundaries to count whole word occurrences
    count = len(re.findall(rf'\b{re.escape(keyword_lower)}\b', flat_text))
    print(f"Found {count} occurrences of '{keyword}'.") # Added print statement
    return count

def get_keyword_matches_for_display(email_texts, keyword):
    """
    Returns a list of formatted strings showing lines containing the keyword,
    suitable for display and download.
    """
    print(f"Getting keyword matches for display for keyword: '{keyword}'.") # Added print statement
    keyword_lower = keyword.lower()
    matches = []
    for i, email_content in enumerate(email_texts):
        subject_match = re.search(r"Subject: (.*)", email_content)
        subject = subject_match.group(1) if subject_match else "No Subject"

        content_lines = email_content.splitlines()
        
        for line_num, line in enumerate(content_lines):
            # Check for whole word match, case-insensitively
            if re.search(rf'\b{re.escape(keyword_lower)}\b', line.lower()):
                matches.append(f"Email {i+1} (Subject: {subject}) - Line {line_num+1}: {line.strip()}")
    print(f"Found {len(matches)} lines with keyword '{keyword}' for display.") # Added print statement
    return matches


# --- Semantic Keyword Extraction (using Gemini) ---
def semantic_keywords_gemini(query: str) -> list[str]:
    """
    Extracst important search keywords or phrases from a natural language query using Gemini.
    These keywords are then used to search Gmail.
    """
    print(f"Gemini: Attempting to extract semantic keywords from query: '{query}'") # Added print statement
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f'''
    You are an NLP assistant. Extract up to 25 important, concise search keywords or short phrases from the user's question.
    These keywords will be used to search for relevant email content.
    Exclude common stop words, general terms, and phrases related to counting (e.g., "how many times", "count", "number of").
    Focus ONLY on the core subject matter of the email content that would be relevant.

    Return the keywords as a Python list of strings. Example: ["keyword1", "keyword2", "specific phrase"]
    Question: "{query}"
    '''
    try:
        response = model.generate_content(prompt)
        print(f"Gemini: Raw response for keyword extraction (first 200 chars): {response.text[:200]}...") # Added print statement
        match = re.search(r'\[.*?\]', response.text, re.DOTALL)
        if match:
            try:
                keywords = eval(match.group(0))
                if isinstance(keywords, list) and all(isinstance(k, str) for k in keywords):
                    logger.info(f"Extracted keywords for Gmail search: {keywords}")
                    print(f"✅ Gemini: Extracted keywords for Gmail search: {keywords}") # Added print statement
                    return [k.strip() for k in keywords if k.strip()]
                else:
                    logger.warning(f"Gemini returned a malformed list or non-string keywords for search: {keywords}. Raw response: {response.text}")
                    print(f"⚠️ WARNING: Gemini returned malformed keywords: {keywords}. Raw: {response.text}") # Added print statement
                    return []
            except (SyntaxError, NameError, TypeError) as e:
                logger.error(f"Failed to parse keywords from Gemini response '{match.group(0)}': {e}")
                print(f"❌ ERROR: Failed to parse keywords from Gemini response: {e}") # Added print statement
                return []
        else:
            logger.warning(f"No list found in Gemini keyword extraction response for search query: '{query}'. Raw response: {response.text}")
            print(f"⚠️ WARNING: No list found in Gemini keyword extraction response for query '{query}'. Raw: {response.text}") # Added print statement
            return []
    except GoogleAPIError as e:
        logger.error(f"Gemini API error during keyword extraction for search query '{query}': {e}")
        st.error(f"Error extracting keywords with Gemini. Please check API key/permissions. Error: {e}")
        print(f"❌ ERROR: Gemini API error during keyword extraction: {e}") # Added print statement
        return []
    except Exception as e:
        logger.error(f"Unexpected error during keyword extraction for search query '{query}': {e}")
        st.error(f"An unexpected error occurred during keyword extraction. Error: {e}")
        print(f"❌ ERROR: Unexpected error during keyword extraction: {e}") # Added print statement
        return []

# --- Gmail Search Functionality (Surface Search) ---
def search_gmail(service, keywords: list[str], original_query: str = "") -> list[str]:
    """
    Searches Gmail for messages matching constructed query (keywords + original query).
    Fetches full message details and extracts content from the email body only.
    This version includes logic to prevent duplicate emails.
    """
    print(f"Gmail Search: Starting search with keywords: {keywords}, original query: '{original_query}'") # Added print statement
    try:
        query_parts = []
        if keywords:
            formatted_keywords = [f'"{k.replace('"', '\\"')}"' for k in keywords if k.strip()]
            query_parts.append(f"({(' OR '.join(formatted_keywords))})")
        
        if original_query and (not keywords or not any(k.lower() in original_query.lower() for k in keywords)):
            query_parts.append(f'"{original_query.replace('"', '\\"')}"')

        gmail_query = " ".join(query_parts).strip()
        if not gmail_query:
            logger.warning("No effective Gmail search query could be constructed from keywords or original query.")
            st.warning("Could not construct a meaningful Gmail search query. Try a more specific question.")
            print("⚠️ WARNING: No effective Gmail search query constructed. Returning empty list.") # Added print statement
            return []

        logger.info(f"Gmail API query being executed: '{gmail_query}'")
        print(f"Gmail Search: Executing API query: '{gmail_query}'") # Added print statement

        all_message_ids = []
        next_page_token = None
        
        with st.spinner(f"Listing all message IDs for query '{gmail_query}'..."):
            while True:
                results = service.users().messages().list(
                    userId='me',
                    q=gmail_query,
                    pageToken=next_page_token,
                    maxResults=500 # Fetch up to 500 message IDs per API call
                ).execute()
                current_page_messages = results.get('messages', [])
                all_message_ids.extend(current_page_messages)
                next_page_token = results.get('nextPageToken')
                print(f"Gmail Search: Fetched {len(current_page_messages)} message IDs. Total so far: {len(all_message_ids)}.") # Added print statement
                if not next_page_token:
                    print("Gmail Search: No more pages of message IDs.") # Added print statement
                    break

        logger.info(f"Found {len(all_message_ids)} Gmail message IDs matching the query (before deduplication).")
        print(f"Gmail Search: Found {len(all_message_ids)} total message IDs matching query.") # Added print statement
        
        formatted_emails = []
        seen_message_ids = set() 
        
        st.info(f"Found {len(all_message_ids)} total message IDs matching your search query.")
        
        if len(all_message_ids) > 0:
            st.warning("Important Note: This tool searches only the text content directly within emails. It **does NOT search inside file attachments** (e.g., PDFs, Word docs) or external websites linked in emails.")


        progress_text = st.empty()
        progress_bar = st.progress(0)

        for i, m in enumerate(all_message_ids):
            if m['id'] in seen_message_ids:
                logger.info(f"Skipping duplicate email ID: {m['id']}")
                print(f"Gmail Search: Skipping duplicate email ID: {m['id']}") # Added print statement
                continue 

            progress_text.text(f"Fetching details for email {len(formatted_emails) + 1} of {len(all_message_ids)} unique messages...")
            progress_bar.progress((i + 1) / len(all_message_ids))

            try:
                msg = service.users().messages().get(userId='me', id=m['id'], format='full').execute()
                headers = msg.get('payload', {}).get('headers', [])

                sender = next((h['value'] for h in headers if h['name'] == 'From'), "N/A")
                subject = next((h['value'] for h in headers if h['name'] == 'Subject'), "No Subject")
                date_str = next((h['value'] for h in headers if h['name'] == 'Date'), "Unknown Date")
                
                try:
                    cleaned_date_str = re.sub(r'\s*\([^)]*\)', '', date_str)
                    parsed_date = None
                    try:
                        parsed_date = datetime.strptime(cleaned_date_str, "%a, %d %b %Y %H:%M:%S %z")
                    except ValueError:
                        # Fallback for dates without timezone in the original string
                        parsed_date = datetime.strptime(cleaned_date_str.rsplit(' ', 1)[0], "%a, %d %b %Y %H:%M:%S")

                    formatted_date = parsed_date.strftime("%Y-%m-%d %H:%M:%S %Z") if parsed_date else date_str
                except ValueError:
                    formatted_date = date_str

                body = extract_email_body(msg['payload'])
                
                email_text = f"--- Email Start ---\nMessage ID: {m['id']}\nFrom: {sender}\nSubject: {subject}\nDate: {formatted_date}\n\n{body}\n--- Email End ---\n"
                formatted_emails.append(email_text)
                seen_message_ids.add(m['id']) 
                logger.info(f"Processed unique email ID: {m['id']} (Subject: {subject}). Total unique processed: {len(formatted_emails)}")
                print(f"Gmail Search: Processed email ID: {m['id']} (Subject: {subject}). Current unique count: {len(formatted_emails)}") # Added print statement

            except Exception as e:
                logger.error(f"Error processing individual email ID {m.get('id', 'N/A')}: {e}")
                print(f"❌ ERROR: Failed to process email ID {m.get('id', 'N/A')}: {e}") # Added print statement
        
        progress_bar.empty()
        progress_text.empty()
        st.success(f"📧 Loaded {len(formatted_emails)} unique relevant emails for analysis.")
        print(f"✅ Gmail Search: Finished. Loaded {len(formatted_emails)} unique relevant emails.") # Added print statement
        
        return formatted_emails
    except GoogleAPIError as e:
        logger.error(f"Gmail API error during search operation: {e}")
        st.error(f"Error searching Gmail: {e}. Please check your Gmail API permissions.")
        print(f"❌ ERROR: Gmail API error during search: {e}") # Added print statement
        return []
    except Exception as e:
        logger.error(f"Unexpected error during Gmail search operation: {e}")
        st.error(f"An unexpected error occurred during Gmail search. Error: {e}")
        print(f"❌ ERROR: Unexpected error during Gmail search: {e}") # Added print statement
        return []

# --- Gemini Answering Function (Context Search) ---
def ask_gemini(question: str, context: str, keyword_for_counting: str = None, actual_count: int = None) -> str:
    """
    Asks Gemini to answer a question based *only* on provided email context.
    The prompt is carefully crafted to limit Gemini's scope and explicitly use
    pre-calculated counts if a counting question is detected.
    """
    print(f"Gemini Answering: Preparing to answer question: '{question}'") # Added print statement
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    # MAX_CONTEXT_LENGTH is now a global variable, no need to redefine here
    
    if len(context) > MAX_CONTEXT_LENGTH:
        logger.warning(f"Context truncated from {len(context)} to {MAX_CONTEXT_LENGTH} characters for Gemini.")
        print(f"⚠️ WARNING: Context truncated from {len(context)} to {MAX_CONTEXT_LENGTH} characters for Gemini.") # Added print statement
        context = context[:MAX_CONTEXT_LENGTH] + "\n\n... (Context truncated due to length limits)"

    prompt_parts = []
    prompt_parts.append(f'''
    You are an AI email assistant. Your sole purpose is to answer the user's question
    **ONLY using the information explicitly found within the provided Gmail messages context.**
    Do NOT use any outside information, general knowledge, or make assumptions.
    Keep the answer concise and to the point. Aim for a summary if multiple emails are relevant.
    If the answer or any part of the answer cannot be directly and unequivocally
    found in the provided "Gmail Messages" context, you MUST state:
    "Not found in the provided Gmail messages context."
    ''')

    # IMPORTANT: Only include the counting override if it's explicitly a counting question
    if keyword_for_counting and actual_count is not None:
        print(f"Gemini Answering: Applying pre-calculated count for keyword '{keyword_for_counting}': {actual_count}") # Added print statement
        prompt_parts.append(f'''
    The user's question explicitly asks about the frequency (count) of the word "{keyword_for_counting}".
    I have already pre-calculated this count for you from the provided Gmail Messages Context.
    The exact count of "{keyword_for_counting}" in the relevant email content is: {actual_count}.
    When answering this specific counting question, you MUST explicitly state this provided number.
    Your answer for a counting question should be direct, e.g., "The word 'X' appears exactly Y times."
    For all other types of questions, provide the requested information from the context.
    ''')

    prompt_parts.append(f'''
    Question: "{question}"

    Gmail Messages Context:
    \"\"\"
    {context}
    \"\"\"

    Please provide a concise and direct answer based strictly on the context.
    ''')
    
    final_prompt = "\n".join(prompt_parts)
    print(f"Gemini Answering: Sending prompt to Gemini (first 500 chars): {final_prompt[:500]}...") # Added print statement

    try:
        response = model.generate_content(final_prompt)
        print(f"Gemini Answering: Received response from Gemini (first 200 chars): {response.text[:200]}...") # Added print statement
        return response.text.strip()
    except GoogleAPIError as e:
        logger.error(f"Gemini API error during answering request: {e}")
        st.error(f"Error generating answer with Gemini: {e}")
        print(f"❌ ERROR: Gemini API error during answering: {e}") # Added print statement
        return "Gemini processing error due to API issue. Please try again."
    except Exception as e:
        logger.error(f"Unexpected error during Gemini answering: {e}")
        print(f"❌ ERROR: Unexpected error during Gemini answering: {e}") # Added print statement
        return "An unexpected error occurred while processing your request. Please check logs."

# --- Streamlit UI ---
st.set_page_config(page_title="📧 Gmail Smart QA", layout="centered")
st.title("📬 Gmail Smart QA Assistant (Gemini)")
st.markdown("Ask questions about your Gmail messages and get answers powered by Gemini, "
            "using semantic, surface, and contextual search.")
print("Streamlit UI initialized.") # Added print statement

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    print("Chat history initialized in session state.") # Added print statement

# User input for the question
user_question = st.text_input("Ask any Gmail-related question :", key="user_input_query")
print(f"Current user question input: '{user_question}'") # Added print statement


# Handle the "Ask" button click
if st.button("Get Answer", key="ask_button_submit") and user_question:
    print(f"\n--- 'Get Answer' button clicked for query: '{user_question}' ---") # Added print statement
    if not user_question.strip():
        st.warning("Please enter a question before clicking 'Get Answer'.")
        print("⚠️ WARNING: User submitted empty question.") # Added print statement
        st.stop()

    with st.spinner("Processing your request..."):
        # Step 1: Authenticate and get Gmail service
        print("Step 1: Authenticating with Gmail.") # Added print statement
        gmail_service = authenticate()
        if not gmail_service:
            st.error("Could not authenticate with Gmail. Please check the console/logs for details.")
            print("❌ ERROR: Gmail authentication failed. Stopping.") # Added print statement
            st.stop()
        print("✅ Gmail service obtained.") # Added print statement

        # Determine if it's a counting question and extract the keyword to count
        # This regex is more specific to "how many" or "count" phrases
        # It also tries to capture the word AFTER "word" if present, or the last significant word.
        keyword_to_count = None
        count_match = re.search(
            r'(?:how many times|count the word|number of)\s+(?:the\s+)?(?:word\s+)?(?:[\'"]?([a-zA-Z0-9_-]+)[\'"]?|([a-zA-Z0-9_-]+))',
            user_question, re.IGNORECASE
        )
        
        if count_match:
            keyword_to_count = count_match.group(1) or count_match.group(2)
            # A final check to ensure we didn't just pick up "word" itself if the query was "how many times word"
            # If "word" is the only thing captured, and not followed by another word in the counting phrase,
            # then it's ambiguous. In this specific case, if the query is just "how many times word", it's usually meant for the next word.
            # Re-evaluate, maybe search for the last non-stop word if "word" was incorrectly identified.
            if keyword_to_count and keyword_to_count.lower() == 'word' and 'word ' not in user_question.lower():
                pass # The prompt for keyword extraction below will help if this is still ambiguous.
        
        # Fallback to semantic extraction if count_match didn't clearly identify the target word
        if not keyword_to_count:
            # Semantic extraction for general search, this can also help for counting if the first regex fails
            print("Attempting semantic extraction for counting keyword fallback.") # Added print statement
            temp_keywords = semantic_keywords_gemini(user_question)
            # If the query is like "how many times logfire", semantic_keywords_gemini should return ["logfire"]
            # We can use this as a strong hint for keyword_to_count if a counting phrase was used.
            if any(phrase in user_question.lower() for phrase in ['how many times', 'count', 'number of']):
                # If a counting phrase is present, and semantic extraction gives us a good keyword, use it for counting.
                if temp_keywords:
                    # Pick the most likely candidate from semantic keywords for counting
                    # This is a heuristic: pick the last keyword as it's often the target in such phrases.
                    keyword_to_count = temp_keywords[-1]
                    st.info(f"Identified '{keyword_to_count}' as the keyword for counting based on semantic analysis of your question.")
                    print(f"✅ Identified '{keyword_to_count}' as keyword for counting via semantic analysis.") # Added print statement


        if keyword_to_count:
            st.info(f"Identified '{keyword_to_count}' as the keyword for counting based on your question.")
            print(f"Detected counting question for keyword: '{keyword_to_count}'.") # Added print statement
        else:
            st.info("No explicit counting keyword detected in your question. Gemini will attempt to answer contextually.")
            print("No counting keyword detected. Proceeding with contextual answering.") # Added print statement


        # Step 2: Semantic Keyword Extraction for Gmail search
        st.info("Step 1/3: Extracting keywords from your question for targeted Gmail search...")
        print("Step 2: Calling semantic_keywords_gemini for Gmail search terms.") # Added print statement
        extracted_keywords_for_gmail_search = semantic_keywords_gemini(user_question)
        
        if not extracted_keywords_for_gmail_search:
            st.warning("Gemini could not extract specific keywords from your question for Gmail search. "
                        "Proceeding with your full question as the primary search term for Gmail.")
            print("⚠️ WARNING: Gemini could not extract keywords. Using full query for Gmail search.") # Added print statement
            search_terms_for_gmail = [user_question]
        else:
            search_terms_for_gmail = extracted_keywords_for_gmail_search
        st.write(f"🔎 Gmail will be searched using terms: `{', '.join(search_terms_for_gmail)}`")
        print(f"Final Gmail search terms: {search_terms_for_gmail}") # Added print statement


        # Step 3: Gmail Surface Search - fetches ALL matching emails (NO WEB SCRAPING)
        st.info(f"Step 2/3: Searching Gmail for all relevant emails...")
        print("Step 3: Calling search_gmail to fetch relevant emails.") # Added print statement
        emails_content_list = search_gmail(gmail_service, search_terms_for_gmail, original_query=user_question)
        
        if not emails_content_list:
            st.info("🚫 No Gmail messages matched your search criteria. Gemini will not be invoked.")
            st.session_state.chat_history.insert(0, {"role": "ai", "content": "No relevant emails found for your query. Please try a different question or search terms."})
            st.session_state.chat_history.insert(0, {"role": "user", "content": f"**Query**: {user_question}"})
            print("🚫 No relevant emails found. Stopping processing for this query.") # Added print statement
            st.stop()

        # Prepare context for Gemini (controlled by TOP_K_EMAILS_FOR_GEMINI)
        emails_for_gemini_context = emails_content_list[:TOP_K_EMAILS_FOR_GEMINI]
        full_context_for_gemini = "\n\n".join(emails_for_gemini_context)
        print(f"Prepared context for Gemini. Number of emails used: {len(emails_for_gemini_context)}") # Added print statement
        # This is where you might see "token count" if you were strictly counting LLM tokens,
        # but for this context, it refers to the number of emails feeding the context.
        print(f"Context size for Gemini (chars): {len(full_context_for_gemini)}. Max context length: {MAX_CONTEXT_LENGTH}") # Added print statement

        display_context_preview = full_context_for_gemini[:50000] + "\n..." if len(full_context_for_gemini) > 50000 else full_context_for_gemini
        with st.expander("📝 Raw Email Context Preview (Max 250 emails, truncated for display)"):
            st.text_area("Content provided to Gemini:", display_context_preview, height=300)

        # Calculate actual count (only if a keyword for counting was explicitly identified)
        actual_count_for_gemini = None
        if keyword_to_count:
            actual_count_for_gemini = count_keyword_occurrences(emails_content_list, keyword_to_count)
            logger.info(f"Dynamically counted '{keyword_to_count}' in all fetched emails: {actual_count_for_gemini}")
            st.info(f"Pre-calculated count of '{keyword_to_count}' in all fetched emails: {actual_count_for_gemini}")
            print(f"Pre-calculated count of '{keyword_to_count}': {actual_count_for_gemini}") # Added print statement


        # Step 4: Contextual Answering (Gemini)
        st.info("Step 3/3: Asking Gemini to answer based on the email content...")
        print("Step 4: Calling ask_gemini to get contextual answer.") # Added print statement
        gemini_answer = ask_gemini(user_question, full_context_for_gemini,
                                            keyword_for_counting=keyword_to_count,
                                            actual_count=actual_count_for_gemini)
        print(f"✅ Gemini Answer received: {gemini_answer[:200]}...") # Added print statement
        
        timestamp_answer = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Update chat history with user query and Gemini's response
        st.session_state.chat_history.insert(0, {"role": "ai", "content": f"**Gemini Answer ({timestamp_answer})**:\n{gemini_answer}"})
        st.session_state.chat_history.insert(0, {"role": "user", "content": f"**Query**: {user_question}"})
        print("Chat history updated.") # Added print statement

        # --- Keyword Counting and Display ---
        if keyword_to_count:
            st.subheader(f"📊 Keyword Analysis for '{keyword_to_count}'")
            # Display the dynamically calculated count
            display_count_text = f"The word **'{keyword_to_count}'** appears exactly **{actual_count_for_gemini}** times in the fetched emails."
            st.write(display_count_text)
            
            # The "Show all occurrences" still shows based on the *actual* content count from emails only
            keyword_matches_list = get_keyword_matches_for_display(emails_content_list, keyword_to_count)
            if keyword_matches_list:
                with st.expander(f"Show all '{keyword_to_count}' occurrences ({len(keyword_matches_list)} matching lines)"):
                    for match_line in keyword_matches_list:
                        st.text(match_line)
                
                with open(keyword_matches_file, "w", encoding="utf-8", errors='replace') as f:
                    f.write(f"Keyword Matches for '{keyword_to_count}':\n")
                    f.write("---------------------------------------\n")
                    for match_line in keyword_matches_list:
                        f.write(match_line + "\n")
                logger.info(f"Keyword matches written to: {keyword_matches_file}")
                print(f"Keyword matches for '{keyword_to_count}' written to {keyword_matches_file}") # Added print statement
            else:
                st.info(f"No specific occurrences of '{keyword_to_count}' found in the fetched email bodies.")
                print(f"No specific occurrences of '{keyword_to_count}' found.") # Added print statement
        else:
            st.info("Tip: To get a count of a specific word, try asking 'how many times \"your_word\" appears?'")

        # --- Download Buttons ---
        st.markdown("---")
        st.subheader("⬇️ Download Results:")
        print("Preparing download buttons.") # Added print statement

        if os.path.exists(log_file_path):
            with open(log_file_path, "r", encoding="utf-8", errors='replace') as f:
                log_content = f.read()
            st.download_button(
                label="Download Full Log File (.log)",
                data=log_content,
                file_name=os.path.basename(log_file_path),
                mime="text/plain"
            )
            print(f"Download button for log file '{log_file_path}' available.") # Added print statement
        
        if keyword_to_count and os.path.exists(keyword_matches_file):
            with open(keyword_matches_file, "r", encoding="utf-8", errors='replace') as f:
                matches_content = f.read()
            st.download_button(
                label=f"Download '{keyword_to_count}' Matches (.txt)",
                data=matches_content,
                file_name=os.path.basename(keyword_matches_file),
                mime="text/plain"
            )
            print(f"Download button for keyword matches '{keyword_matches_file}' available.") # Added print statement

        if emails_content_list:
            with open(email_dump_path, "w", encoding="utf-8", errors='replace') as f:
                for content in emails_content_list:
                    f.write(content + "\n" + "-" * 80 + "\n")
            logger.info(f"All fetched email content dumped to: {email_dump_path}")
            print(f"All fetched email content dumped to: {email_dump_path}") # Added print statement

            with open(email_dump_path, "r", encoding="utf-8", errors='replace') as f:
                email_dump_content = f.read()
            st.download_button(
                label="Download All Fetched Email Content (.txt)",
                data=email_dump_content,
                file_name=os.path.basename(email_dump_path),
                mime="text/plain"
            )
            print(f"Download button for all fetched email content '{email_dump_path}' available.") # Added print statement

# Display chat history
st.subheader("Conversation History")
if not st.session_state.chat_history:
    st.info("Your conversation history will appear here after you ask a question.")
    print("Conversation history is empty.") # Added print statement
else:
    print(f"Displaying {len(st.session_state.chat_history) // 2} previous conversations.") # Added print statement
    for chat_entry in st.session_state.chat_history:
        with st.chat_message(chat_entry["role"]):
            st.markdown(chat_entry["content"])

# Option to clear chat history
if st.session_state.chat_history and st.button("Clear Chat History", key="clear_chat_button"):
    st.session_state.chat_history = []
    print("Chat history cleared by user.") # Added print statement
    st.rerun()

print("--- Application Execution Finished (or awaiting next user action) ---") # Added print statement