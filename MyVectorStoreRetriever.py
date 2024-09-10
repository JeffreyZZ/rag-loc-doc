from langchain.schema.vectorstore import VectorStoreRetriever
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.schema.document import Document
from typing import List

class MyVectorStoreRetriever(VectorStoreRetriever):
    # See https://github.com/langchain-ai/langchain/blob/61dd92f8215daef3d9cf1734b0d1f8c70c1571c3/libs/langchain/langchain/vectorstores/base.py#L500
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        docs_and_similarities = (
            self.vectorstore.similarity_search_with_relevance_scores(
                query, **self.search_kwargs
            )
        )

        # =============================================
        # Make the score part of the document metadata
        #==============================================
        for doc, similarity in docs_and_similarities:
            doc.metadata["score"] = similarity
            doc.metadata["source"] = "vector"

        docs = [doc for doc, _ in docs_and_similarities]
        return docs