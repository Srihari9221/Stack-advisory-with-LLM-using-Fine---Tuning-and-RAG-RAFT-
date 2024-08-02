import os
import json
import sqlite3
import pandas as pd
import google.auth
import google_auth_oauthlib.flow
import google.auth.transport.requests
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import streamlit as st

st.set_page_config(page_title="CSV Analysis", layout="wide")
client_secret_path = 'client_secret.json'
token_path = 'token.json'
prompt_file = 'prompt2.txt' 
api_key = 'AIzaSyDUzTjmUlea7ZHea8Bd_yIx3FLLNU0TjRg' 

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = client_secret_path

credentials = None

# If token.exists, load credentials
if os.path.exists(token_path):
    with open(token_path, 'r') as token_file:
        credentials_info = json.load(token_file)
        credentials = google.oauth2.credentials.Credentials.from_authorized_user_info(credentials_info)

# If credentials are None, run the OAuth flow to get the new credentials
if not credentials or not credentials.valid:
    flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(
        client_secret_path,
        scopes=['https://www.googleapis.com/auth/cloud-platform',
                'https://www.googleapis.com/auth/generative-language.tuning']
    )
    credentials = flow.run_local_server(port=0)

    # Save the credentials for next run
    with open(token_path, 'w') as token_file:
        token_file.write(credentials.to_json())

# Refresh the credentials to ensure they are valid
credentials.refresh(google.auth.transport.requests.Request())
st.success('Authenticated successfully.')

# Load prompt from text file
with open(prompt_file, 'r') as file:
    prompt = [file.read()]

# Configure the generative AI API
genai.configure(credentials=credentials)

# Function To Load Google Gemini Model and provide queries as response
def get_gemini_response(question, prompt):
    # Use the fine-tuned model here
    model = genai.GenerativeModel('tunedModels/test-zy42bbktz6ig')
    response = model.generate_content([prompt[0], question])
    return response.text

#  retrieve relevant details
def retrieve_documents(question, conn):
    df = pd.read_sql_query("SELECT * FROM data", conn)
    
    # Combine relevant columns for retrieval
    df['combined'] = df['FunctionalDomain'] + ' ' + df['FunctionalArea'] + ' ' + df['TechnologyStack'] + ' ' + df['License'] + ' ' + df['Tool'] + ' ' + df['LastUpdated'] + ' ' + df['Advisory'] + ' ' + df['SupportedVersion']

    # Vectorize the text using TF-IDF
    tfidf = TfidfVectorizer().fit_transform(df['combined'])
    
    # Vectorize the question
    question_tfidf = TfidfVectorizer().fit(df['combined']).transform([question])
    # cosine simularity and return 3 relavant  documents
    cosine_similarities = linear_kernel(question_tfidf, tfidf).flatten()
    relevant_indices = cosine_similarities.argsort()[-3:][::-1]
    relevant_documents = df.iloc[relevant_indices]
    
    return relevant_documents

# additional info
def display_additional_info(question,combined_relavant_text):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content([question,combined_relevant_text])
        st.write("Additional Information:")
        st.write(response.text)
    
    except Exception as e:
        st.error(f"Error fetching additional information: {e}")

# Streamlit 
st.title(' Stack Advisory using RAFT')

question = st.text_input('Enter your question:')
if st.button('Submit'):
    if question:
        try:
            # Connect to SQLite database
            conn = sqlite3.connect('stack.db')
            
            # Retrieve relevant documents
            relevant_docs = retrieve_documents(question, conn)
            combined_relevant_text = ' '.join(relevant_docs['combined'].tolist())
            print(combined_relevant_text)
            
            # Generate SQL query
            sql_query = get_gemini_response(question, [prompt[0], combined_relevant_text])
            with st.expander("Generated SQL Query"):
                st.write(sql_query)
            
            # Fetch results into DataFrame
            df = pd.read_sql_query(sql_query, conn)
            
            # Display results
            if not df.empty:
                st.write("Query Results:")
                st.dataframe(df, use_container_width=True)
            else:
                st.write("No results found.")
            
            display_additional_info(question,combined_relevant_text)
            # Close Connection
            conn.close()
            
        except sqlite3.Error as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning('Please enter a question before submitting.')
