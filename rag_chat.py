import getpass
import os
from constants import API_KEY
os.environ["GOOGLE_API_KEY"] = API_KEY

from langchain_google_vertexai import ChatVertexAI

llm = ChatVertexAI(model="gemini-pro")
#import bs4
from langchain import hub
from langchain_chroma import Chroma
#from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document

# In this project we are goint to implement a QA App over Mardown data
# We will use pandoc to convert txt files to Markdown and the proceed with processing the data.

markdown_path = "./data.md"
loader = UnstructuredMarkdownLoader(markdown_path)
data = loader.load()
readme_content = data[0].page_content

docs = loader.load()

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print(rag_chain.invoke("Was sind untergeordnete Bauteile?"))


# cleanup
vectorstore.delete_collection()