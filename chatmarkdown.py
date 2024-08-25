from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_ollama import OllamaEmbeddings

class ChatMarkdown:
    vector_store = None
    retriever = None
    chain = None

    def __init__(self):
        self.model = ChatOllama(model="mistral")
        headers_to_split_on = [
            ("#", "Header 1"), 
            ("##", "Header 2"), 
            ("###", "Header 3"), 
            ("####", "Header 4")
        ]
        self.text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=True)
        # TODO: prompt template for Ollama
        self.prompt = PromptTemplate.from_template(
            """
            Context information in markdown format is provided below.
            ---------------------
            {context}
            ---------------------
            Given the context information and not prior knowledge, keep the answer concise.
            If you don't know the answer, just say that you don't know. 
            Query: {question}
            Answer: 
            """
        )

    def ingest(self, md_file_path: str, file_name: str):
        loader = TextLoader(file_path=md_file_path, encoding = 'UTF-8')
        docs = loader.load()
        all_chunks=[]

        #==========
        # Chunking
        #==========
        for doc in docs:
            if doc.page_content: 
                chunks = self.text_splitter.split_text(doc.page_content)
                for chunk in chunks: 
                    prefix = file_name + '\n'
                    if 'Header 1' in chunk.metadata:
                        prefix += "# " + chunk.metadata['Header 1'] + '\n'
                    if 'Header 2' in chunk.metadata:
                        prefix += "## " + chunk.metadata['Header 2'] + '\n'
                    chunk.page_content = prefix + chunk.page_content
                all_chunks.extend(chunks)
                   
        # Check if all_chunks is an empty list
        if not all_chunks:
            return False
        
        #============
        # Embedding
        #============
        # using local running mxbai-embed-large embedding model with Ollama, so needs to install Ollama
        # and pull mxbai-embed-large model.  
        embeddings = OllamaEmbeddings(
            model="mxbai-embed-large",
        )
        self.vector_store =  Chroma.from_documents(documents=all_chunks, embedding=embeddings) # FastEmbedEmbeddings() is faster one for local debugging purpose
        self.retriever =  self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 6,
                "score_threshold": 0.3,
            },
        )

        self.chain = ({"context": self.retriever, "question": RunnablePassthrough()}
                    | self.prompt
                    | self.model
                    | StrOutputParser())
        return True

    def ask(self, query: str):
        if not self.chain:
            return "Please, add a markdown document first."
        # return the query results for debugging purpose
        results = self.vector_store.similarity_search_with_score(query, k=6)
        chunks = ""
        for doc, score in results:
            chunks += f"""------[{score:.4f}]-----------------------------------------------------------
            {doc.metadata}
            {doc.page_content}
            """      
        return self.chain.invoke(query) + '\n' + chunks

    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None
