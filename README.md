# Creating a QA Application using LangChain

In this project, we will be implementing a simple application that enables us to perform Question-Answering (QA) on a given document using Retrieval Augmented Generation (RAG). Specifically, we will be using a subset of the Serbian construction law as our dataset for performing QA tasks.

## What is Retrieval Augmented Generation
Retrieval Augmented Generation is a powerful technique used to enhance the knowledge of Large Language Models (LLMs) by incorporating additional data. While LLMs have extensive knowledge on a wide range of topics, their training data is limited to a certain point in time. This means that there is a possibility that the model has not been trained on a specific document you want to ask questions about.

This is where RAG proves to be invaluable. It allows us to provide relevant information along with the user prompt to the LLM, enabling it to generate accurate answers. By leveraging RAG, we can bridge the gap between the existing knowledge of LLMs and the specific document we want to analyze. 


## Components of a RAG Application

A typical RAG Application consists of two main components. The first part is called "Indexing". During this phase, we load and ingest our data from the source and index it into a Vectorstore. This process usually takes place offline.

The second part is called "Retrieval and Generation". In this phase, we pass the user query to our index, retrieve all the relevant information, and then prompt this information to the LLM (Large Language Model) to generate an appropriate answer.

By dividing the RAG Application into these two components, we can efficiently manage the data and ensure accurate and timely responses to user queries.

We can split the Workflow of our RAG Pipeline into 5 steps.
### Indexing 
1. **Load**: We are first going to load out data into a suitable format for indexing using a _DocumentLoader_ 

2. **Split**: We will divide the loaded data into smaller chunks to facilitate searching in the Vector space where it will be embedded. In this project we are going to use the _RecursiveTextCharacterSplitter_



