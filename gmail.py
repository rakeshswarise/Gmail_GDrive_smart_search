import os
import base64
import re
import logging
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import google.generativeai as genai

# Load Gemini API Key
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

@st.cache_resource
def gmail_authenticate():
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
        creds = flow.run_local_server(port=0)
        with open("token.json", "w") as token:
            token.write(creds.to_json())
    return build("gmail", "v1", credentials=creds)

# Recursive MIME decoder
def extract_text_from_parts(parts):
    texts = []
    for part in parts:
        mime_type = part.get("mimeType", "")
        body = part.get("body", {})
        data = body.get("data")
        if data:
            try:
                text = base64.urlsafe_b64decode(data.encode("UTF-8")).decode("utf-8", errors="ignore")
                texts.append(text)
            except:
                pass
        elif "parts" in part:
            texts.extend(extract_text_from_parts(part["parts"]))
    return texts

# Extract full email text
def extract_email_text(service, msg_id):
    msg = service.users().messages().get(userId="me", id=msg_id, format="full").execute()
    payload = msg.get("payload", {})
    headers = payload.get("headers", [])
    subject = next((h["value"] for h in headers if h["name"] == "Subject"), "(No Subject)")
    link = f"https://mail.google.com/mail/u/0/#inbox/{msg_id}"

    all_texts = []

    if "parts" in payload:
        all_texts = extract_text_from_parts(payload["parts"])
    else:
        body = payload.get("body", {}).get("data")
        if body:
            try:
                text = base64.urlsafe_b64decode(body.encode("UTF-8")).decode("utf-8", errors="ignore")
                all_texts.append(text)
            except:
                pass

    return subject, link, "\n".join(all_texts)

# Surface search
def surface_search(texts, keyword):
    count = 0
    matches = []
    for subject, link, text in texts:
        word_count = len(re.findall(rf"\b{re.escape(keyword)}\b", text, flags=re.IGNORECASE))
        if word_count > 0:
            matches.append((subject, link, word_count))
        count += word_count
    return count, matches

# Gemini call
def gemini_search(prompt):
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text.strip()

# Streamlit UI
st.title("📬 Gmail Smart QA (Surface + Semantic Search)")
st.markdown("**Ask something about your Gmail messages**")

query = st.text_input("🔍 Enter your question:", placeholder="e.g., how many times word logfire available")

if st.button("Search") and query:
    service = gmail_authenticate()

    # Get all messages (Paginate beyond 100 if needed)
    messages = []
    response = service.users().messages().list(userId="me", maxResults=500).execute()
    messages.extend(response.get("messages", []))
    while "nextPageToken" in response:
        response = service.users().messages().list(userId="me", maxResults=500, pageToken=response["nextPageToken"]).execute()
        messages.extend(response.get("messages", []))

    if not messages:
        st.warning("🚫 No Gmail messages found.")
    else:
        texts = []
        for msg in messages:
            try:
                subject, link, body = extract_email_text(service, msg["id"])
                texts.append((subject, link, body))
            except Exception as e:
                logging.error(f"Error reading message: {e}")

        st.success(f"✅ Found {len(texts)} email(s) for your query!")

        # Extract keyword
        keyword_match = re.search(r"word\s+(\w+)", query.lower())
        keyword = keyword_match.group(1) if keyword_match else None

        if keyword:
            surface_count, matches = surface_search(texts, keyword)
            st.markdown(f"**🔎 The word '{keyword}' appears `{surface_count}` times across all fetched emails.**")

            if matches:
                st.markdown("---")
                st.subheader("📌 Matching Emails:")
                for subject, link, count in matches:
                    st.markdown(f"- [{subject}]({link}) — `{count}` times")
        else:
            st.info("❓ No keyword found. Please include 'word <keyword>' in your question.")

        # Gemini Response
        full_context = "\n\n".join([body for _, _, body in texts])
        final_prompt = f"{query}\n\nHere is the email content:\n{full_context}"
        try:
            gemini_response = gemini_search(final_prompt)
            st.markdown("### 🤖 Gemini Answer:")
            st.info(gemini_response)
        except Exception as e:
            st.error(f"Gemini API Error: {e}")
