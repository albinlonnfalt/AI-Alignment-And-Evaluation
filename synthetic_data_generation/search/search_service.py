from abc import abstractmethod
import random
import json
from typing import Any, Iterable
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from openai import AzureOpenAI
from models import QA


class SearchService:
    """
    Service class for handling Azure Search operations and search result manipulation.
    """
    
    def __init__(self, search_client: SearchClient, llm_client: AzureOpenAI):
        self.search_client = search_client
        self.llm_client = llm_client
        self.total_document_count = None

    #
    ## -- Search Operations -- ##
    #

    @abstractmethod
    def get_random_chunk(self):
        """
        Get a random chunk from the search index.
        
        Returns:
            dict: A random chunk document
        """
        pass

    def get_total_document_count(self):
        """
        Get the total document count in the search index.
        
        Returns:
            int: The total count of documents
        """
        pass
    
    def get_all_chunks_of_document(self, chunk_search_record: Any):
        """
        Get all chunks/documents from the search index that contain the same document_id as the input chunk.

        Args:
            chunk_search_record (Any): A search result document containing tags field
            
        Returns:
            list: List of all matching documents with the same document_id
        """
        pass

    @abstractmethod
    def qa_search(self, qa: QA) -> Iterable[Any]:
        """
        Perform a hybrid search to find relevant chunks based on the question and answer in the Q&A object.
        Note: The chunks in QA is excluded from the search results.
        
        Args:
            qa (QA): The generated Q&A object.
        
        Returns:
            list: List of search results containing relevant chunks.
        """
        pass

    @abstractmethod
    def get_search_records_by_ids(self, search_record_ids: list[str]) -> Iterable[Any]:
        """
        Get chunks/documents from the search index by their IDs.
        
        Args:
            search_record_ids (list[str]): List of chunk IDs to retrieve
            
        Returns:
            Iterable[Any]: Iterable of search results containing the specified chunks
        """
        pass

    #
    ## -- Search Result Manipulations -- ##
    #

    @abstractmethod
    @staticmethod
    def sort_chunks_by_part_number(search_records: Iterable[Any]) -> Iterable[Any]:
        """
        Sort a list of chunks by their part number.
        
        Args:
            search_record (Any):
            
        Returns:
            list: Sorted list of chunks by part number
        """
        pass
    
    @abstractmethod
    @staticmethod
    def get_search_record_title(search_record: Any) -> str:
        """
        Get the document title from a search record.
        
        Args:
            search_record (Any): A search result document containing a title field
            
        Returns:
            str: The document title or empty string if not found
        """
        pass

    @abstractmethod
    @staticmethod
    def get_search_record_id(search_record: Any) -> str:
        """
        Get the document ID from a search record.
        
        Args:
            search_record (Any): A search result document containing an id field
            
        Returns:
            str: The document ID or empty string if not found
        """
        pass

    @abstractmethod
    @staticmethod
    def get_search_record_content(search_record: Any) -> str:
        """
        Get the document content from a search record.
        
        Args:
            search_record (Any): A search result document containing a payload field
            
        Returns:
            str: The document content or empty string if not found
        """
        pass