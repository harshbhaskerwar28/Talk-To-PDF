import os
import io
import PyPDF2
import pytesseract
from PIL import Image
import streamlit as st
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

# Initialize session state variables
if 'vector_store' not in st.session_state:
    st.session_state['vector_store'] = None
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Tesseract File Path
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Harsh\Desktop\Slack_Bot\Tesseract-OCR\tesseract.exe'

model_path = r"path/to/llama-2-7b-chat.ggmlv3.q8_0.bin"

# Function to extract text from PDF and image files (with optional OCR for PDF)
def extract_text_from_files(file_bytes_list, file_extension_list, ocr=False):
    # ... (same as before)

# Function to split text into chunks for vector store
def split_text_into_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    text_chunks = text_splitter.split_text(text)
    return text_chunks

# Function to create a vector store from text chunks
def create_vector_store(chunks):
    if not chunks:
        st.warning("No text content found in the uploaded file.")
        return

    try:
        embeddings = CTransformers(model_path=model_path, model_type="llama", config={'max_new_tokens': 256, 'temperature': 0.01})
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
        st.session_state['vector_store'] = vector_store
    except Exception as e:
        st.error(f"An error occurred while creating the vector store: {str(e)}")

# Function to set up the conversational chain for question-answering
def setup_conversational_chain():
    prompt_template = """
    Answer the question in a short, precise, detailed, friendly and engaging way, drawing from the provided context if possible. If the question is not directly related to the context, provide a thoughtful and relevant response based on your general knowledge.
    Make sure the response is in common English and appears human-like, avoiding any patterns or language that could be identified as AI-generated by AI text detectors.

    Context:{context}

    Question:{question}

    Answer:
    """

    model = CTransformers(model_path=model_path, model_type="llama", config={'max_new_tokens': 256, 'temperature': 0.01})
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain

# Function to get a response to a user question
def get_response(user_question):
    relevant_docs = st.session_state['vector_store'].similarity_search(user_question)
    conversational_chain = setup_conversational_chain()
    response = conversational_chain({"input_documents": relevant_docs, "question": user_question}, return_only_outputs=True)
    return response

# Main function
def main():
    # Set Streamlit app configuration
    st.set_page_config(page_title="Talk to PDF Bot", page_icon=":book:", layout="wide")

    # Create a container for the file uploader
    upload_container = st.container()
    with upload_container:
        st.title("Talk to PDF Bot")
        st.write("Upload your PDF or Image files, and I'll extract the text. You can then ask me questions about the files' content.")

        # File uploader
        uploaded_files = st.file_uploader("Upload your PDF or Image files", type=["pdf", "png", "jpg", "jpeg", "webp"], accept_multiple_files=True)
        ocr = st.checkbox("Enable OCR for PDF files")

        if st.button("Clear Conversation"):
            st.session_state['messages'].clear()
            st.session_state['vector_store'] = None
            st.experimental_rerun()

        if uploaded_files:
            file_bytes_list = [uploaded_file.read() for uploaded_file in uploaded_files]
            file_extension_list = [os.path.splitext(uploaded_file.name)[1].lower() for uploaded_file in uploaded_files]

            for uploaded_file in uploaded_files:
                file_extension = os.path.splitext(uploaded_file.name)[1].lower()
                if file_extension in [".png", ".jpg", ".jpeg", ".webp"]:
                    with st.expander("Display Image"):
                        st.image(uploaded_file)

            st.write("Processing your files...")
            file_text = extract_text_from_files(file_bytes_list, file_extension_list, ocr)
            text_chunks = split_text_into_chunks(file_text)
            create_vector_store(text_chunks)
            st.write("Files processed successfully!")

    # Create a container for the chatbot
    chatbot_container = st.container()
    with chatbot_container:
        if st.session_state['vector_store'] is not None:
            st.title("Ask me anything about the file's content")

            # Display previous messages
            for message in st.session_state['messages']:
                with st.chat_message(message["role"]):
                    st.write(message["content"])

            # Get user input
            if prompt := st.chat_input("", key="chat_input"):
                st.session_state['messages'].append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.write(prompt)

                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = get_response(prompt)
                        full_response = ''.join(response['output_text'])
                        st.write(full_response)
                        message = {"role": "assistant", "content": full_response}
                        st.session_state['messages'].append(message)

if __name__ == "__main__":
    main()
