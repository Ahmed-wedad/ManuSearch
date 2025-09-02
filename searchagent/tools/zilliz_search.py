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

    def _is_valid_content(self, content: str) -> bool:
        """
        Check if content is valid and not just formatting artifacts.
        
        Args:
            content: The content string to validate
            
        Returns:
            True if content is valid, False otherwise
        """
        if not content or len(content.strip()) < 10:
            return False
            
        # Clean the content for analysis
        clean_content = content.strip()
        
        # Remove common markdown/formatting characters for analysis
        analysis_content = clean_content.replace('\n', '').replace('*', '').replace('#', '').replace('`', '').replace('-', '').strip()
        
        # Check if content is mostly formatting characters
        if len(analysis_content) < 5:
            return False
            
        # Check ratio of actual content vs formatting
        content_ratio = len(analysis_content) / len(clean_content)
        if content_ratio < 0.3:  # Less than 30% actual content
            return False
            
        # Check for repeated patterns (like many newlines or asterisks)
        if '\n\n\n\n' in content or '****' in content or '```\n\n```' in content:
            return False
            
        # Check for excessive repetition of any character
        for char in ['\n', '*', '#', '`', '-', '_']:
            if content.count(char) > len(content) * 0.5:  # More than 50% of same character
                return False
            
        # Check if content has some actual words (not just symbols)
        import re
        words = re.findall(r'\b\w{3,}\b', analysis_content)
        if len(words) < 3:  # Less than 3 meaningful words
            return False
            
        return True

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

            # Concatenate all queries into a single search (reverting to original approach)
            combined_query = ' '.join(queries)
            combined_intent = intents[0] if intents else None
            
            logger.info(f"Combined query: '{combined_query}' with intent: {combined_intent}")
            
            # Get documents for the combined query
            docs = self.retriever.get_relevant_docs(combined_query, k=self.top_k, intent=combined_intent)
            logger.info(f"Retrieved {len(docs)} documents for combined query")
            
            # Filter valid documents
            valid_docs = []
            for doc in docs:
                doc_content = doc.get('page_content', '').strip()
                if self._is_valid_content(doc_content):
                    valid_docs.append(doc)

            # Check if we have any valid documents
            if not valid_docs:
                logger.warning("No valid documents found after filtering")
                return {}

            logger.info(f"Total valid documents after filtering: {len(valid_docs)}")

            # Create single grouped result with all documents concatenated
            # This mimics how the old reader.py was concatenating chunks
            search_results = {}
            
            # Group all documents into a single result entry
            all_chunks_content = {}
            all_titles = []
            all_scores = []
            all_dates = []
            
            for i, doc in enumerate(valid_docs):
                metadata = doc.get('metadata', {})
                content = doc.get('page_content', '').strip()
                
                # Extract title
                title = metadata.get('title', metadata.get('filename', f'Document {i}'))
                if title and title not in all_titles:
                    all_titles.append(title)
                
                # Collect scores and dates
                all_scores.append(doc.get('score', 1.0))
                date = metadata.get('date', '2024')
                if date not in all_dates:
                    all_dates.append(date)
                
                # Add content as chunk
                chunk_key = f"chunk_{i}"
                all_chunks_content[chunk_key] = content

            # Skip if no valid content found
            if not all_chunks_content:
                logger.warning("No valid content found in any documents")
                return {}

            # Create single search result with all documents combined
            combined_title = " | ".join(all_titles[:3])  # Use first 3 titles
            if len(all_titles) > 3:
                combined_title += f" (+{len(all_titles)-3} more)"
            
            search_results["0"] = {
                "title": combined_title,
                "content": all_chunks_content,  # All chunks in one dict
                "date": all_dates[0] if all_dates else '2024',  # Use first date
                "score": max(all_scores) if all_scores else 1.0,  # Use highest score
            }

            logger.info(f"ZillizSearch returned 1 combined document group with {len(all_chunks_content)} chunks from {len(valid_docs)} documents")
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