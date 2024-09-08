# from langchain_community.vectorstores import Chroma
import time
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
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain_community.retrievers import BM25Retriever, ElasticSearchBM25Retriever
from langchain.retrievers import EnsembleRetriever
from elasticsearch import Elasticsearch, ConflictError
from langchain_core.documents import Document
from langchain_chroma import Chroma
from MyElasticSearchBM25Retriever import MyElasticSearchBM25Retriever
from MyVectorStoreRetriever import MyVectorStoreRetriever

class ChatMarkdown:
    # vector_store = None
    # retriever = None
    # chain = None
    # keyword_retriever = None
    # elastic_retriever = None
    # ensemble_retriever = None
    elasticsearch_url = "http://localhost:9200"
    index_name = "rag-loc-doc"

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
        #============================
        # Create Elasticsearch index
        #============================
        client = Elasticsearch(self.elasticsearch_url)
        exists = client.indices.exists(index=self.index_name)
        if exists:
            self.elastic_retriever = MyElasticSearchBM25Retriever(
                client=Elasticsearch(self.elasticsearch_url), 
                index_name=self.index_name)
        else: 
            self.elastic_retriever = MyElasticSearchBM25Retriever.create(
                self.elasticsearch_url, 
                self.index_name)
        self.keyword_retriever = self.elastic_retriever
        
        #===========================
        # Embedding: vector search
        #===========================
        # using local running mxbai-embed-large embedding model with Ollama, so needs to install Ollama
        # and pull mxbai-embed-large model.  
        # mxbai-embed-large: https://www.mixedbread.ai/docs/embeddings/mxbai-embed-large-v1
        embeddings = OllamaEmbeddings(
            model="mxbai-embed-large",
        )
        self.vector_store = Chroma(collection_name="my_collection", embedding_function=embeddings)
        # TODO: output retriever result for debugging
        self.retriever = MyVectorStoreRetriever(
            vectorstore=self.vector_store,
            search_type="similarity_score_threshold",
            search_kwargs={
                "score_threshold": 0.45,
                "k": 3
            },
        ) 
        
        #===================================
        # Hybrid serarch: vector + keyword
        #===================================
        ensemble_retriever = EnsembleRetriever(retrievers=[self.elastic_retriever, self.retriever],
                                               weights=[0.5, 0.5])
        self.ensemble_retriever = ensemble_retriever

        #===================================
        # Langchain chain
        #===================================
        self.chain = ({"context": self.ensemble_retriever, "question": RunnablePassthrough()}
                    | self.prompt
                    | self.model
                    | StrOutputParser())

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

        #===========================
        # Insert into vector store
        #===========================
        self.vector_store.add_documents(documents=all_chunks) #  embedding=embeddings FastEmbedEmbeddings() is faster one for local debugging purpose

        #===========================
        # Insert into keyword store
        #===========================
        # Convert to a list of strings
        list_of_contents = [chunk.page_content for chunk in all_chunks]
        self.elastic_retriever.add_texts(list_of_contents)

        #============
        # Reranking 
        #============
        # FlashRank: https://github.com/PrithivirajDamodaran/FlashRank
        # TODO: output reranking result for debugging
        compressor = FlashrankRerank()
        # self.retriever = ContextualCompressionRetriever(
        #     base_compressor=compressor, 
        #     base_retriever=ensemble_retriever
        # )
        return True

    def ask(self, query: str):
        if not self.chain:
            return "Please, add a markdown document first."
        
        chunks = ""
        documents = self.ensemble_retriever.get_relevant_documents(query)
        
        for doc in documents:
            chunks += f"""------Retriever---------------------------------------------------------------
            {doc.metadata}
            {doc.page_content}
            """     
        # return the query results for debugging purpose
        # results = self.vector_store.similarity_search_with_score(query, k=3)
        results = self.retriever.invoke(query)    
        for doc in results:
            chunks += f"""------[Vector:{doc.metadata.get('score'):.4f}]---------------------------------------------------
            {doc.metadata}
            {doc.page_content}
            """     

        keyword_results = self.keyword_retriever.invoke(query)
        for doc in keyword_results:
            chunks += f"""------[Elastic]---------------------------------------------------------------
            {doc}
            """  
        return self.chain.invoke(query) + '\n' + chunks

    def clear(self):
        # self.vector_store = None
        # self.retriever = None
        # self.chain = None
        pass