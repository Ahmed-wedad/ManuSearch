"""
Zilliz Vector Store Search Tool for ManuSearch.

Replaces GoogleSearch with vector database search for ENET'Com documents.
"""

import json
import logging
from typing import Dict, List, Any, Tuple, Union

from pydantic import Field
from .basetool import BaseTool

logger = logging.getLogger(__name__)

# Metadata fields to keep when pruning
KEEP = {
    "document_id",
    "chunk_id",
    "title",
    "url",
    "document_source",
    "date",
    "chunk_size",
    "chunk_timestamp",
    "parent_section_path",
    "source_page_title",
    "download_url",
    "source_type",
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
            Search results in ManuSearch format
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

            # Combine queries if multiple
            if queries:
                search_query = " ".join(queries)
            else:
                return {}

            # Use the primary intent if multiple are provided
            primary_intent = intents[0] if intents else None

            logger.info(f"ZillizSearch executing with queries: {queries}, intent: {primary_intent}")

            # Use the exact same retriever as your RAG system, now with intent support
            docs = self.retriever.get_relevant_docs(search_query, k=self.top_k, intent=primary_intent)

            # Convert to ManuSearch expected format
            search_results = {}

            # Add individual documents
            for i, doc in enumerate(docs):
                content = doc.get('page_content', '')
                metadata = doc.get('metadata', {})

                # Extract title from metadata or content
                title = metadata.get('title', metadata.get('filename', f'ENET\'Com Document {i}'))

                if not title and content:
                    # Try to extract title from first line of markdown
                    lines = content.split('\n')
                    for line in lines:
                        if line.strip().startswith('#'):
                            title = line.strip().lstrip('#').strip()
                            break

                if not title:
                    title = f'ENET\'Com Document {i}'

                # Prune metadata to keep only relevant fields
                clean_meta, _ = prune_metadata_batch(metadata, keep_keys=KEEP)

                search_results[str(i)] = {
                    "title": title,
                    "content": content,  # Full content already available from vector store
                    "date": clean_meta.get('date', '2024'),
                    "score": doc.get('score', 1.0),
                    # "metadata": clean_meta
                }

            logger.info(f"ZillizSearch returned {len(search_results)} results for intent: {primary_intent}")
            return search_results

        except Exception as e:
            logger.error(f"ZillizSearch failed: {e}")
            return {}
def prune_metadata_item(

        meta: Dict,
        remove_keys: Union[set, None] = None,
        keep_keys: Union[set, None] = None,
        in_place: bool = False
    ) -> Tuple[Dict, Dict]:
        """
        Prune a single metadata dict.

        Args:
            meta: original metadata dict.
            remove_keys: set of keys to remove (optional).
            keep_keys: set of keys to KEEP (optional). If provided, removal is (all_keys - keep_keys).
            in_place: if True, mutate meta; otherwise operate on a shallow copy.

        Returns:
            (pruned_meta, removed_fields) where removed_fields is a dict of popped key->value.
        """
        if not in_place:
            meta = dict(meta)  # shallow copy

        if keep_keys is not None:
            # derive remove_keys from keys present in meta
            remove_keys = set(meta.keys()) - set(keep_keys)
        else:
            remove_keys = set(remove_keys or ())

        removed = {}
        for k in list(remove_keys):
            if k in meta:
                removed[k] = meta.pop(k)

        return meta, removed

def prune_metadata_batch(
        metas: Union[Dict, List[Dict]],
        remove_keys: Union[List[str], set, None] = None,
        keep_keys: Union[List[str], set, None] = None,
        in_place: bool = False
    ) -> Tuple[Union[Dict, List[Dict]], List[Dict]]:
        """
        Prune a single dict or list of dicts.

        Args:
            metas: dict or list of dicts.
            remove_keys: iterable of keys to remove (optional).
            keep_keys: iterable of keys to KEEP (optional). If provided, removal is (all_keys - keep_keys).
            in_place: if True, mutate items in-place when metas is a list; otherwise works on copies.

        Returns:
            (pruned_metas, removed_per_item) - pruned_metas is same type as input metas (dict or list of dicts).
            - removed_per_item is a list of dicts showing keys removed for each item (single-item case returns list with one dict).
        """
        rem = set(remove_keys) if remove_keys is not None else None
        keep = set(keep_keys) if keep_keys is not None else None

        if isinstance(metas, dict):
            p, r = prune_metadata_item(metas, remove_keys=rem, keep_keys=keep, in_place=in_place)
            return p, [r]

        pruned_list = []
        removed_list = []
        for item in metas:
            p, r = prune_metadata_item(item, remove_keys=rem, keep_keys=keep, in_place=in_place)
            pruned_list.append(p)
            removed_list.append(r)

        return pruned_list, removed_list