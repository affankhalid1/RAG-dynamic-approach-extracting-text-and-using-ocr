from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
# from langchain.chains.question_answering import load_qa_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from pypdf import PdfReader
import streamlit as st
from dotenv import load_dotenv
import os 
import sys
from pdf2image import convert_from_bytes
from  io import BytesIO
from google.cloud import vision


load_dotenv()

groq_key = os.getenv('GROQ_API_KEY')
google_key = os.getenv('GOGGLE_API_KEY')


llm  = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key = groq_key 
)


uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
raw_text= ''

if uploaded_file:
    pdf_bytes = uploaded_file.read()
    pdfReader = PdfReader(BytesIO(pdf_bytes))
    images = convert_from_bytes(pdf_bytes)


    # read text from pdf 
    pdf_reader_calls = 0
    ocr_calls = 0
    for i, page in enumerate(pdfReader.pages):
        content = page.extract_text()
        if content and "" not in content:   
                raw_text += content
                pdf_reader_calls += 1
                # st.write(raw_text)
        else:
            # Use the corresponding image for that page from pdf2image output
            image_pil = images[i]
            buffered = BytesIO()
            image_pil.save(buffered, format="PNG")
            img_bytes = buffered.getvalue()

            client = vision.ImageAnnotatorClient()
            image = vision.Image(content = img_bytes)

            response = client.text_detection(image = image)
            texts = response.text_annotations

            if texts: 
                raw_text += texts[0].description
                ocr_calls += 1
                # st.write(raw_text)


    # st.write("pdf calls "+str(pdf_reader_calls))
    # st.write("ocr calls " +str(ocr_calls))
    st.write(raw_text)

# uploaded_Image = st.file_uploader("Upload an Image file", type=["jpeg", "jpg", "png"])


# We need to split the text using Character text split such that it should not increase token size 
text_splitter = CharacterTextSplitter(
     separator="\n",
     chunk_size = 800,
     chunk_overlap = 200,
     length_function = len
) 
texts = text_splitter.split_text(raw_text)

# Download embeddings from openai 
st.session_state.embeddings = OllamaEmbeddings(model = "nomic-embed-text:latest")
if texts:
    st.session_state.vectorstore = FAISS.from_texts(texts, st.session_state.embeddings)

    # st.write(document_search)
    # prompt = PromptTemplate.from_template("Summarize the following text:\n\n{context}")


    query = st.text_input("Write you query...")
    prompt = ChatPromptTemplate.from_template("""
    Acts as an expert Chatbot Assistant. Your job is to provide the best and accurate answer according to the qustion using the text given below.
    Question: {input}
    Text:{context}
    """)
    if query:
        document_chain = create_stuff_documents_chain(llm, prompt)
        st.session_state.retriever = st.session_state.vectorstore.as_retriever()
        retrieval_chain = create_retrieval_chain(st.session_state.retriever , document_chain)
        result = retrieval_chain.invoke({"input": query})
        content = result["answer"]
        st.write("Summary: ")
        st.write(content)