import getpass
import os
from constants import API_KEY
from langchain_google_vertexai import ChatVertexAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain import hub
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document
from langchain.prompts import HumanMessagePromptTemplate 
from langchain_core.prompts.prompt import PromptTemplate

os.environ["GOOGLE_API_KEY"] = API_KEY

## We are going to use the gemini-pro model from Google for this project.
llm = ChatVertexAI(model="gemini-pro")
# In this project we are going to implement a QA App over Mardown data
# We will use pandoc to convert txt files to Markdown and the proceed with processing the data.
markdown_path = "./data.md"

# We are going to use the Markdown loader to load the file into a editable format.
loader = UnstructuredMarkdownLoader(markdown_path)

    
docs = loader.load()

print(len(docs[0].page_content))

# We are going to use the Google AI embedding to embed the data into a Vector Space.
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
# We will use the recursive Character Splitter to split the data into chunks of 1000 
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
splits = text_splitter.split_documents(docs)
    
# We will use Chroma as a Vector store to store the chunks of data and embed them in there.
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    
system_prompt = (
    "Vi ste asistent za zadatke odgovaranja na pitanja."
    "Koristite sledeće delove dobijenog konteksta da odgovorite na pitanje."
    "Ako ne znate odgovor, samo recite da ne znate."
    "Koristite najviše tri rečenice i neka odgovor bude sažet. "
    "Sve odgovore koje generišete, generišite molim Vas na srpskom jeziku."
    "\n\n"
    "{context}"
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def rag_chat(question):

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{question}"),
        ]
    )


    chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
    )

    output = ""
    for chunk in chain.stream(question):
        output += chunk
    return output

