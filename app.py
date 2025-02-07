import os
import io
import PyPDF2  # Library to read PDF files
import pytesseract  # Library for optical character recognition (OCR)
from PIL import Image  # Library for image processing
import streamlit as st  # Streamlit library for building web applications
from dotenv import load_dotenv  # Library to load environment variables
import google.generativeai as gen_ai  # Google Generative AI library
from langchain_community.vectorstores import FAISS  # Vector store for efficient similarity search
from langchain.prompts import PromptTemplate  # Library for creating prompts
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Library for splitting text into chunks
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # Embeddings for Google Generative AI
from langchain_google_genai import ChatGoogleGenerativeAI  # Google Generative AI model for chatbot
from langchain.chains.question_answering import load_qa_chain  # Library for question-answering chains

# Load environment variables and configure the API key
load_dotenv()
os.getenv("GOOGLE_API_KEY")
gen_ai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Initialize session state variables
if 'vector_store' not in st.session_state:
    st.session_state['vector_store'] = None
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Set the Tesseract executable path (update as needed)
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Harsh\Desktop\Slack_Bot\Tesseract-OCR\tesseract.exe'

def extract_text_from_files(file_bytes_list, file_extension_list, ocr=False):
    text = ""
    for file_bytes, file_extension in zip(file_bytes_list, file_extension_list):
        if file_extension == ".pdf":
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            for page in pdf_reader.pages:
                if ocr:
                    # If OCR is enabled, try to extract text from images on the page
                    images = getattr(page, "images", None)
                    if images:
                        for image in images:
                            image_bytes = image.data
                            image_file = io.BytesIO(image_bytes)
                            img = Image.open(image_file)
                            text += pytesseract.image_to_string(img) + "\n"
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text
        elif file_extension in [".png", ".jpg", ".jpeg", ".webp"]:
            image_file = io.BytesIO(file_bytes)
            image = Image.open(image_file)
            text += pytesseract.image_to_string(image) + "\n"
    return text

def split_text_into_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def create_vector_store(chunks):
    if not chunks:
        st.warning("No text content found in the uploaded file.")
        return
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
        st.session_state['vector_store'] = vector_store
    except Exception as e:
        st.error(f"An error occurred while creating the vector store: {str(e)}")

def setup_conversational_chain():
    prompt_template = """
    Provide a detailed, concise, and user-friendly response based on the given context.
    
    Context: {context}
    
    Question: {question}
    
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", client=gen_ai, temperature=0.7)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain

def get_response(user_question):
    relevant_docs = st.session_state['vector_store'].similarity_search(user_question)
    conversational_chain = setup_conversational_chain()
    response = conversational_chain(
        {"input_documents": relevant_docs, "question": user_question},
        return_only_outputs=True
    )
    return response

def generate_summary():
    if st.session_state['vector_store'] is None:
        st.warning("No content available to summarize.")
        return
    with st.spinner("Generating summary..."):
        conversational_chain = setup_conversational_chain()
        docs = st.session_state['vector_store'].similarity_search("", k=10)
        summary_prompt = "Summarize the key points from this content:"
        summary_response = conversational_chain(
            {"input_documents": docs, "question": summary_prompt},
            return_only_outputs=True
        )
        return summary_response['output_text']

def is_greeting(text):
    # Recognize several greeting variations (e.g., "hii", "hello", etc.)
    greetings = {"hi", "hii", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening"}
    words = set(text.lower().split())
    return bool(greetings.intersection(words))

def chat_ui():
    # Display the chat conversation
    for message in st.session_state['messages']:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Get user input from the chat
    user_input = st.chat_input("Type your message here")
    if user_input:
        # Append and display the user's message
        st.session_state['messages'].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)
        
        # Determine the assistant's response:
        # If the user's message is a greeting, respond with a friendly greeting.
        if is_greeting(user_input):
            # If it's the first user message, use a default greeting.
            if len(st.session_state['messages']) == 1:
                bot_response = "Hello! How can I help you with your PDFs today?"
            else:
                bot_response = "Hello again! How can I assist you further with your PDFs?"
        else:
            response = get_response(user_input)
            bot_response = (response['output_text']
                            if isinstance(response['output_text'], str)
                            else ''.join(response['output_text']))
        st.session_state['messages'].append({"role": "assistant", "content": bot_response})
        with st.chat_message("assistant"):
            st.write(bot_response)

def main():
    st.set_page_config(page_title="Talk to PDF Bot", page_icon=":book:", layout="wide")

    # Sidebar for file uploads
    st.sidebar.title("Upload Files")
    uploaded_files = st.sidebar.file_uploader(
        "Upload your PDF or Image files", 
        type=["pdf", "png", "jpg", "jpeg", "webp"], 
        accept_multiple_files=True
    )
    ocr = st.sidebar.checkbox("Enable OCR for PDF files")

    if st.sidebar.button("Clear Conversation"):
        st.session_state['messages'].clear()
        st.session_state['vector_store'] = None
        st.experimental_rerun()

    # Process uploaded files only once (if not already processed)
    if uploaded_files and st.session_state['vector_store'] is None:
        file_bytes_list = [uploaded_file.read() for uploaded_file in uploaded_files]
        file_extension_list = [os.path.splitext(uploaded_file.name)[1].lower() for uploaded_file in uploaded_files]
        # Optionally display images for non-PDF files
        for uploaded_file in uploaded_files:
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            if file_extension in [".png", ".jpg", ".jpeg", ".webp"]:
                with st.expander("Display Image"):
                    st.image(uploaded_file)
        status_placeholder = st.empty()
        status_placeholder.write("Processing your files...")
        file_text = extract_text_from_files(file_bytes_list, file_extension_list, ocr)
        text_chunks = split_text_into_chunks(file_text)
        create_vector_store(text_chunks)
        status_placeholder.empty()  # Clear the processing message

    st.title("Talk to PDF Bot")

    # Button to generate a summary of uploaded files
    if st.button("Generate Summary"):
        summary = generate_summary()
        if summary:
            st.subheader("Summary of Uploaded Files:")
            st.write(summary)

    # If no messages exist, add an initial greeting from the assistant
    if not st.session_state['messages']:
        initial_greeting = "Hello! How can I help you with your PDFs today?"
        st.session_state['messages'].append({"role": "assistant", "content": initial_greeting})
        with st.chat_message("assistant"):
            st.write(initial_greeting)

    # Display the chat UI for conversation
    chat_ui()

if __name__ == "__main__":
    main()
