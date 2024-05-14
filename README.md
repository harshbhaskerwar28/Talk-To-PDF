# Streamlit PDF and Image Processing Chatbot Application

## Introduction
This project is a Streamlit web application that allows users to interact with PDF and image files through a conversational interface. The application leverages various libraries and APIs to provide a seamless experience for users to extract text content from their files and ask questions related to that content.

## Overview
The application follows a structured workflow to process the user's files and generate relevant responses to their queries. Here's a high-level overview of the application's functionality:

### File Upload
Users can upload one or multiple PDF and image files (PNG, JPG, JPEG, WEBP) through the Streamlit interface.

### Text Extraction
The application processes the uploaded files to extract text content:
- For PDF files, the PyPDF2 library is used to extract text from each page.
- If the OCR (Optical Character Recognition) option is enabled for PDFs, pytesseract is used to perform OCR on any images embedded in the PDF pages.
- For image files, pytesseract is used to perform OCR and extract text.

### Text Processing
The extracted text is split into smaller chunks using the RecursiveCharacterTextSplitter from the LangChain library. These text chunks are then converted into embeddings using the GoogleGenerativeAIEmbeddings model, also from LangChain.

### Vector Store
The generated embeddings are stored in a FAISS vector store, which allows for efficient similarity search and retrieval of relevant document chunks.

### Conversational Chain Setup
A conversational chain is set up using the PromptTemplate and ChatGoogleGenerativeAI model from LangChain. This chain is responsible for generating responses based on the context and the user's question.

### User Interaction
The application displays a chat interface where users can ask questions related to the content of the uploaded files.

### Similarity Search
When a user asks a question, the FAISS vector store performs a similarity search to retrieve the most relevant document chunks based on the question's embedding.

### Response Generation
The relevant document chunks and the user's question are passed to the conversational chain. The ChatGoogleGenerativeAI model generates a response by considering the context and its general knowledge.

### Response Display
The generated response is displayed in the chat interface, and the conversation history is maintained in the application's session state.

### Conversation Management
Users can clear the conversation history, which resets the session state and vector store, allowing them to upload new files and start a new conversation.

## Technical Implementation
The application leverages several libraries and technologies to achieve its functionality:
- Streamlit
- PyPDF2
- pytesseract
- Pillow
- LangChain
- Google
User
give only that part i will add it to my befor code
ChatGPT
markdown
Copy code
## Set Environment Variables

Set the `FORCE_CMAKE` environment variable to force the use of CMake:
```bash
set FORCE_CMAKE=1


If you have an NVIDIA GPU, set DLLAMA_CUBLAS to ON:
```bash
set DLLAMA_CUBLAS=ON

##Compile and Install

Navigate into the llama-cpp-python directory:
```bash
cd llama-cpp-python


Install the package using pip with the --upgrade and --force-reinstall options to ensure a clean installation:
```bash
python -m pip install --upgrade --force-reinstall llama-cpp-python



## Set Environment Variables

Set the `FORCE_CMAKE` environment variable to force the use of CMake:
```bash
set FORCE_CMAKE=1

If you have an NVIDIA GPU, set DLLAMA_CUBLAS to ON:
``bash
set DLLAMA_CUBLAS=ON


##Compile and Install

Navigate into the llama-cpp-python directory:
``bash
cd llama-cpp-python
Install the package using pip with the --upgrade and --force-reinstall options to ensure a clean installation:

``bash
python -m pip install --upgrade --force-reinstall llama-cpp-python

If you encounter issues with the installation, you can try installing with the --no-cache-dir option:
``bash
python -m pip install --upgrade --force-reinstall --no-cache-dir llama-cpp-python

##Verify Installation

After the installation, verify that llama-cpp-python is installed correctly by running:
``bash
python -c "import llama; print(llama.__version__)"

This should print the version of llama-cpp-python installed on your system.
