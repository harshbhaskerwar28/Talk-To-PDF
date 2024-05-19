def main():
    st.set_page_config(
        page_title="Talk to PDF Bot",
        page_icon=":book:",
        layout="wide"
    )

    # Sidebar for file uploads
    st.sidebar.title("Upload Files")
    uploaded_files = st.sidebar.file_uploader("Upload your PDF or Image files", type=["pdf", "png", "jpg", "jpeg", ".webp"], accept_multiple_files=True)
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
