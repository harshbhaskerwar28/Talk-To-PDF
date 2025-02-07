import os
import io
import PyPDF2
import pytesseract
from PIL import Image
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as gen_ai
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain

load_dotenv()
gen_ai.configure(api_key=os.environ["GOOGLE_API_KEY"])

if 'vector_store' not in st.session_state:
    st.session_state['vector_store'] = None
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

pytesseract.pytesseract.tesseract_cmd = r'C:\\Users\\Harsh\\Desktop\\Slack_Bot\\Tesseract-OCR\\tesseract.exe'

# Extract text from PDF or image files
def extract_text_from_files(file_bytes_list, file_extension_list, ocr=False):
    text = ""
    for file_bytes, file_extension in zip(file_bytes_list, file_extension_list):
        if file_extension == ".pdf":
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            for page_num in range(len(pdf_reader.pages)):
                page_obj = pdf_reader.pages[page_num]
                text += page_obj.extract_text() or ""
        elif file_extension in [".png", ".jpg", ".jpeg", ".webp"]:
            image_file = io.BytesIO(file_bytes)
            image = Image.open(image_file)
            text += pytesseract.image_to_string(image) + "\n"
    return text

# Split text into chunks
def split_text_into_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

# Create vector store
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

# Setup conversational chain
def setup_conversational_chain():
    prompt_template = """
    Provide a detailed, concise, and user-friendly response based on the given context.
    Context: {context}
    Question: {question}
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", client=gen_ai, temperature=0.7)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)

# Get response for questions
def get_response(user_question):
    relevant_docs = st.session_state['vector_store'].similarity_search(user_question)
    conversational_chain = setup_conversational_chain()
    response = conversational_chain({"input_documents": relevant_docs, "question": user_question}, return_only_outputs=True)
    return response

# Get summary of all content
def generate_summary():
    if st.session_state['vector_store'] is None:
        st.warning("No content available to summarize.")
        return
    with st.spinner("Generating summary..."):
        conversational_chain = setup_conversational_chain()
        context_text = "\n".join([doc.page_content for doc in st.session_state['vector_store'].similarity_search("", k=10)])
        summary_prompt = "Summarize the key points from this content:"
        summary_response = conversational_chain({"input_documents": [context_text], "question": summary_prompt}, return_only_outputs=True)
        return summary_response['output_text']

# Main function
def main():
    st.set_page_config(page_title="Talk to PDF Bot", page_icon=":book:", layout="wide")

    # Sidebar for file uploads
    st.sidebar.title("Upload Files")
    uploaded_files = st.sidebar.file_uploader("Upload your PDF or Image files", type=["pdf", "png", "jpg", "jpeg", "webp"], accept_multiple_files=True)
    ocr = st.sidebar.checkbox("Enable OCR for PDF files")

    if st.sidebar.button("Clear Conversation"):
        st.session_state['messages'].clear()
        st.session_state['vector_store'] = None
        st.experimental_rerun()

    # Process uploaded files
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

    # Main content area
    st.title("Talk to PDF Bot")

    # Add Summary Button
    if st.button("Generate Summary"):
        summary = generate_summary()
        if summary:
            st.subheader("Summary of Uploaded Files:")
            st.write(summary)

    if st.session_state['vector_store'] is not None:
        st.write("Ask me anything about the file's content.")

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
