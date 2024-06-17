import os
import streamlit as st
import google.generativeai as genai
import pdfplumber
from PIL import Image
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import pytesseract

# Function to extract text from images using OCR
def extract_text_from_image(image):
    try:
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        st.warning(f"Error extracting text from image: {e}")
        return ""

# Function to extract text from a single PDF
def get_single_pdf_chunks(pdf, text_splitter):
    pdf_chunks = []
    try:
        with pdfplumber.open(pdf) as pdf_reader:
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    page_chunks = text_splitter.split_text(page_text)
                    pdf_chunks.extend(page_chunks)
                else:
                    # Use OCR to extract text from the page image if text is not found
                    page_image = page.to_image()
                    image_text = extract_text_from_image(page_image.original)
                    if image_text.strip():
                        page_chunks = text_splitter.split_text(image_text)
                        pdf_chunks.extend(page_chunks)
    except Exception as e:
        st.warning(f"Error processing PDF: {e}")
    return pdf_chunks

# Function to process all uploaded PDFs
def get_all_pdfs_chunks(pdf_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2500,
        chunk_overlap=500
    )
    all_chunks = []
    for pdf in pdf_docs:
        pdf_chunks = get_single_pdf_chunks(pdf, text_splitter)
        all_chunks.extend(pdf_chunks)
    return all_chunks

# Function to process uploaded images
def get_images_chunks(images):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2500,
        chunk_overlap=500
    )
    image_chunks = []
    for image_file in images:
        try:
            image = Image.open(image_file)
            image_text = extract_text_from_image(image)
            if image_text.strip():
                chunks = text_splitter.split_text(image_text)
                image_chunks.extend(chunks)
        except Exception as e:
            st.warning(f"Error processing image {image_file.name}: {e}")
    return image_chunks

# Function to create the vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        return vectorstore
    except Exception as e:
        st.warning("Issue with creating vector store. Your file might be scanned so there will be nothing in chunks for embeddings to work on.")
        st.error(str(e))
        return None

# Function to get response from the model
def get_response(context, question, model):
    if "chat_session" not in st.session_state:
        st.session_state.chat_session = model.start_chat(history=[])

    prompt_template = f"""
    You are a helpful and informative bot that answers questions using text from the reference context included below.\
    Be sure to respond in a complete sentence, providing in-depth, detailed information and including all relevant background information. \
    However, you are talking to a non-technical audience, so be sure to break down complicated concepts and \
    strike a friendly and conversational tone.and provide atleast 20 lines\
    If the passage is irrelevant to the answer, you may ignore it. \
    Additionally, suggest queries based on the context given before answering the question.
    
    Context: {context}\n
    Question: {question}\n
    """

    try:
        response = st.session_state.chat_session.send_message(prompt_template)
        return response.text
    except Exception as e:
        st.warning(e)
        return "Error generating response"

# Function to handle the working process
def working_process(generation_config, doc_type):
    system_instruction = "You are a helpful document answering assistant. You care about user and user experience. You always make sure to fulfill user request."

    model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest", generation_config=generation_config, system_instruction=system_instruction)

    vectorstore_key = 'vectorstore_pdfs' if doc_type == 'pdf' else 'vectorstore_images'
    vectorstore = st.session_state[vectorstore_key]

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello! I'm a PDF Assistant. Ask me anything about your PDFs or Documents.")
        ]
    
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.markdown(message.content)

    user_query = st.chat_input("Enter Your Query....")
    if user_query is not None and user_query.strip() != "":
        st.session_state.chat_history.append(HumanMessage(content=user_query))

        with st.chat_message("Human"):
            st.markdown(user_query)
        
        with st.chat_message("AI"):
            try:
                relevant_content = vectorstore.similarity_search(user_query, k=10)
                result = get_response(relevant_content, user_query, model)
                st.markdown(result)
                st.session_state.chat_history.append(AIMessage(content=result))
            except Exception as e:
                st.warning(e)

# Main function
def main():
    load_dotenv()

    st.set_page_config(page_title="", page_icon=":books:")
    st.header("JITS PDF QUERY SYSTEM")
    
    genai_api_key = os.getenv("GOOGLE_API_KEY")
    if not genai_api_key:
        st.error("API key for Google Generative AI is missing.")
        return

    genai.configure(api_key=genai_api_key)

    generation_config = {
        "temperature": 0.2,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": 10000,
    }

    if "vectorstore_pdfs" not in st.session_state:
        st.session_state.vectorstore_pdfs = None
    if "vectorstore_images" not in st.session_state:
        st.session_state.vectorstore_images = None

    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and Click on 'Submit PDFs'", accept_multiple_files=True, type=['pdf'])
        if st.button("Submit PDFs"):
            with st.spinner("Processing PDFs"):
                text_chunks = get_all_pdfs_chunks(pdf_docs)
                vectorstore_pdfs = get_vector_store(text_chunks)
                st.session_state.vectorstore_pdfs = vectorstore_pdfs

        image_files = st.file_uploader("Upload your Images here and Click on 'Submit Images'", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])
        if st.button("Submit Images"):
            with st.spinner("Processing Images"):
                image_chunks = get_images_chunks(image_files)
                vectorstore_images = get_vector_store(image_chunks)
                st.session_state.vectorstore_images = vectorstore_images

    st.write("Select the type of document to query:")
    doc_type = st.selectbox("Document Type", ["PDF", "Image"])

    if doc_type == "PDF" and st.session_state.vectorstore_pdfs is not None:
        working_process(generation_config, 'pdf')
    elif doc_type == "Image" and st.session_state.vectorstore_images is not None:
        working_process(generation_config, 'image')

if __name__ == "__main__":
    main()
