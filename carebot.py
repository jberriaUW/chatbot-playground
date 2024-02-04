import os
from pypdf import PdfReader
from tiktoken import get_encoding

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.vectorstores.chroma import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain


class Carebot(object):
    def __init__(self):
        # We're going to use the following vars as a quick way to check state
        # TODO: do this better
        self._data_str = ''
        self._data_chunks = []
        self._data_size = 0
        self._files = []
        self._chunk_size = 0
        self._overlap_size = 0
        self._embedding_model = ''
        self._llm = ''
        self._temp = 0
        self._promt = ''
        self._store = ''

    # Read in the pdf's and parse the text into a single string
    # pdfs can be a file type or string (path)
    def load_pdfs(self, pdfs):
        # TODO don't be lazy, do this check right
        if len(self._files) == len(pdfs):
            return

        self._files.clear()
        for pdf in pdfs:
            self._files.append(pdfs)
            reader = PdfReader(pdf)
            for page in reader.pages:
                self._data_str += page.extract_text()

    # Take our single string of pdf data and create chunks
    def chunk_data(self, chunk_size: int, overlap_size: int):
        if self._chunk_size != chunk_size or self._overlap_size != overlap_size:
            self._chunk_size = chunk_size
            self._overlap_size = overlap_size

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=overlap_size,
                separators=["\n\n", "\n", " ", ""]
            )
            self._data_chunks = splitter.split_text(self._data_str)

    # Combine our data from the input files (i.e pdfs) and a LLM to create a vector store
    def create_vectorstore(self, model: str, store: str):
        if self._embedding_model != model or self._store != store:
            self._embedding_model = model
            self._store = store

            if self._store == 'FAISS':
                self._vector_store = FAISS.from_texts(
                    texts=self._data_chunks,
                    embedding=OpenAIEmbeddings(model=model)
                )
            elif self._store == 'Chroma':
                self._vector_store = Chroma.from_texts(
                    texts=self._data_chunks,
                    embedding=OpenAIEmbeddings(model=model)
                )
                pass

    # Setup the LLM with the prompt and the context (input files).
    def create_chain(self, model: str, temp: int, prompt: str):
        if self._llm != model or self._temp != temp or self._promt != prompt:
            self._llm = model
            self._temp = temp
            self._promt = prompt

            llm = ChatOpenAI(model=model, temperature=temp)
            prompt = ChatPromptTemplate.from_messages([
                ("system", prompt),
                ("system", "{question}\n\n{context}")
            ])
            # TODO: use a refined or reduced chain to see if that uses less tokens
            self._chain = create_stuff_documents_chain(llm=llm, prompt=prompt)

    # Search our vectorstore to find the relative documents from our pdfs based on the
    # the users input. Than feed those documents to the LLM to get an answer
    def _search_vectorstore(self, input: str) -> list[any]:
        return self._vector_store.similarity_search(input)

    # Pass the users input into the vectorstore to get the correct docs from the
    # input files, and pass those as the context and the users question to the LLM
    def ask(self, question: str) -> str:
        docs = self._search_vectorstore(question)
        return self._chain.invoke({"context": docs, "question": question})
