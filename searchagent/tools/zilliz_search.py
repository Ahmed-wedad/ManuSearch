"""
Zilliz Vector Store Search Tool for ManuSearch.

Replaces GoogleSearch with vector database search for ENET'Com documents.
"""

from datetime import datetime
import json
import logging
from typing import Dict, List, Any, Tuple, Union

from pydantic import Field
from .basetool import BaseTool

logger = logging.getLogger(__name__)

# Metadata fields to keep when pruning
KEEP = {
    "document_id",
    "chunk_size",
    "chunk_timestamp",
    "parent_page_url",
    "filename"
}


class ZillizSearch(BaseTool):
    """
    Vector store search tool that replaces GoogleSearch in ManuSearch.

    Uses the exact same retriever interface as your RAG system with intent-based retrieval.
    """

    # Pydantic fields - Now use native name
    name: str = "ZillizSearch"
    description: str = "Search ENET'Com knowledge base using vector similarity with intent-based retrieval"
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Search queries for ENET'Com documents"
            },
            "intent": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Search intent or purpose (optional)"
            }
        },
        "required": ["query"]
    }

    # Additional fields for the tool
    retriever: Any = Field(..., description="ZillizRetriever instance")
    top_k: int = Field(default=5, description="Number of documents to retrieve")

    model_config = {"arbitrary_types_allowed": True}  # Allow arbitrary types like retriever

    def __init__(self, retriever, top_k: int = 5, **kwargs):
        """
        Initialize Zilliz search tool.

        Args:
            retriever: ZillizRetriever instance with get_relevant_docs method
            top_k: Number of documents to retrieve
        """
        # Initialize parent with required fields - use "ZillizSearch" name
        super().__init__(
            name="ZillizSearch",
            description="Search ENET'Com knowledge base using vector similarity with intent-based retrieval",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Search queries for ENET'Com documents"
                    },
                    "intent": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Search intent or purpose (optional)"
                    }
                },
                "required": ["query"]
            },
            retriever=retriever,
            top_k=top_k,
            **kwargs
        )

    def to_schema(self) -> Dict[str, Any]:
        """Return tool schema for ManuSearch agent."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Search queries for the vector database"
                        },
                        "intent": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Search intent or context (e.g., 'compare', 'find', 'summarize', 'detailed', 'latest', 'historical')"
                        }
                    },
                    "required": ["query"]
                }
            }
        }



    def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute vector store search matching ManuSearch's expected format with intent support.

        Args:
            **kwargs: Should contain 'query' and optionally 'intent'
            intent can be: 'compare', 'find', 'summarize', 'detailed', 'latest', 'historical'

        Returns:
            Search results in ManuSearch format with all documents concatenated into single group
        """
        try:
            # Extract parameters
            queries = kwargs.get('query', [])
            intents = kwargs.get('intent', [])

            if isinstance(queries, str):
                queries = [queries]
            if isinstance(intents, str):
                intents = [intents]

            print(queries, intents)

            if not queries:
                return {}

            logger.info(f"ZillizSearch executing with queries: {queries}, intents: {intents}")

            # Process each query individually and collect docs per query-intent pair
            all_docs = []
            
            for i, query in enumerate(queries):
                # logger.info(f"Processing query {i+1}/{len(queries)}: '{query}'")
                
                # # Get corresponding intent or use default
                # current_intent = intents[i] if i < len(intents) else "general"
                
                # Get documents for this individual query
                docs = self.retriever.get_relevant_docs(query, k=self.top_k)
                logger.info(f"Retrieved {len(docs)} documents for query: '{query}' ")
                
                # Add all retrieved documents to the combined list
                
                all_docs.extend(docs)
            
            # Use all collected documents for further processing
            docs = all_docs
            logger.info(f"Total retrieved documents from all queries: {len(docs)}")
            # print("docs", docs)
            # Group by parent_page_url with nested document structure
            url_groups = {}
            # print(docs)
            for doc in docs:
                raw_metadata = doc.get('metadata', {})
                content = doc.get('page_content', '').strip()
                
                # Get group key from metadata
                group_key = raw_metadata.get('parent_page_url', raw_metadata.get('download_url'))
                doc['title'] = raw_metadata.get('parent_page_title', raw_metadata.get('filename', ''))
                doc['content'] = content
                doc['score'] = doc.get('score', 0.0)
                # Prune metadata to keep only relevant fields
                # metadata, _ = prune_metadata_item(raw_metadata, keep_keys=KEEP)
                
                if group_key not in url_groups:
                    # First occurrence of this url
                    url_groups[group_key] = []
          
                url_groups[group_key].append(doc)

            logger.info(f"ZillizSearch returned {len(url_groups)} url groups")

            return { key : {
                    'url': url,
                    'date': datetime.now().strftime("%Y-%m-%d"),
                    'title': docs_list[0]['title'] if docs_list else '',
                    'content': {idx:doc['content'] for idx, doc in enumerate(sorted([d for d in docs_list if 'content' in d], key=lambda x: x['score'], reverse=True))}
                } for key, (url, docs_list) in enumerate(url_groups.items())}

        except Exception as e:
            logger.error(f"ZillizSearch failed: {e}")
            return {}
# 