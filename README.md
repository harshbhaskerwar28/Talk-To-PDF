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
- Google Generative AI
- FAISS

The application follows a modular design, with separate functions and components responsible for different tasks, such as file processing, text extraction, embedding generation, vector storage, and response generation.

## Setup and Usage
To set up and run the application locally, follow these steps:
1. Clone the repository from the provided source.
2. Install the required dependencies by running `pip install -r requirements.txt`.
3. Set up the Google Generative AI API key as an environment variable.
4. Update the path for the Tesseract OCR executable in the code.
5. Install Tesseract OCR by downloading and running the setup file from [this link](https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w64-setup-5.3.3.20231005.exe).
6. Run the Streamlit application using the command `streamlit run app.py`.
7. Upload your PDF and image files through the web interface.
8. Enable the OCR option for PDF files if needed.
9. Ask questions related to the content of the uploaded files in the chat interface.
10. Responses will be generated and displayed based on the relevant context and general knowledge.
11. You can clear the conversation history and upload new files as needed.


## Conclusion
This Streamlit application provides a user-friendly interface for processing PDF and image files, extracting text content, and engaging in a conversational experience. By leveraging various libraries and APIs, the application offers a comprehensive solution for text extraction, embedding generation, similarity search, and contextual response generation.

With its modular design and clear documentation, this project can serve as a starting point for further enhancements or integration into larger applications that require text processing and conversational AI capabilities.
