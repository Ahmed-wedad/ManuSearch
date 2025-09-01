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

            if not queries:
                return {}

            logger.info(f"ZillizSearch executing with queries: {queries}, intents: {intents}")

            # Handle multiple queries separately instead of concatenating
            all_docs = []
            seen_docs = set()  # To avoid duplicates based on content
            
            # Process each query with its corresponding intent
            for i, query in enumerate(queries):
                # Use corresponding intent or first intent or None
                current_intent = intents[i] if i < len(intents) else (intents[0] if intents else None)
                
                logger.info(f"Processing query {i+1}/{len(queries)}: '{query}' with intent: {current_intent}")
                
                # Get documents for this specific query
                docs = self.retriever.get_relevant_docs(query, k=self.top_k, intent=current_intent)
                logger.info(f"Retrieved {len(docs)} documents for query: '{query}'")
                
                # Add unique documents to avoid duplicates
                valid_docs_count = 0
                for doc in docs:
                    doc_content = doc.get('page_content', '').strip()
                    
                    # Skip invalid content early
                    if not self._is_valid_content(doc_content):
                        logger.debug(f"Skipping invalid content: {doc_content[:100]}...")
                        continue
                        
                    # Use a more robust hash for duplicate detection
                    # Remove whitespace and common formatting for better duplicate detection
                    clean_for_hash = doc_content.replace('\n', ' ').replace('*', '').replace('#', '').strip()
                    content_hash = hash(clean_for_hash[:1000])  # Use first 1000 chars for hashing
                    
                    if content_hash not in seen_docs:
                        seen_docs.add(content_hash)
                        all_docs.append(doc)
                        valid_docs_count += 1
                
                logger.info(f"Added {valid_docs_count} valid documents from query: '{query}'")

            # Check if we have any valid documents
            if not all_docs:
                logger.warning("No valid documents found after filtering")
                return {}

            logger.info(f"Total valid documents collected: {len(all_docs)}")

            # Group documents by document source/title for chunking
            doc_groups = {}
            for doc in all_docs:
                metadata = doc.get('metadata', {})
                # Use document_id, title, or URL as grouping key
                group_key = (
                    metadata.get('document_id') or 
                    metadata.get('title') or 
                    metadata.get('url') or 
                    metadata.get('filename') or
                    'unknown'
                )
                
                if group_key not in doc_groups:
                    doc_groups[group_key] = []
                doc_groups[group_key].append(doc)

            # Convert to ManuSearch expected format with chunked content
            search_results = {}
            result_id = 0

            # Process each document group
            for group_key, docs_in_group in doc_groups.items():
                if not docs_in_group:
                    continue
                    
                # Get representative metadata from first document in group
                first_doc = docs_in_group[0]
                metadata = first_doc.get('metadata', {})
                
                # Extract title from metadata or content
                title = metadata.get('title', metadata.get('filename', f'ENET\'Com Document {result_id}'))
                
                if not title and first_doc.get('page_content'):
                    # Try to extract title from first line of markdown
                    lines = first_doc.get('page_content', '').split('\n')
                    for line in lines:
                        if line.strip().startswith('#'):
                            title = line.strip().lstrip('#').strip()
                            break

                if not title:
                    title = f'ENET\'Com Document {result_id}'

                # Create chunked content dict from all documents in group
                chunk_content = {}
                valid_chunk_count = 0
                for i, doc in enumerate(docs_in_group):
                    content = doc.get('page_content', '').strip()
                    
                    # Filter out low-quality content
                    if self._is_valid_content(content):
                        chunk_key = f"chunk_{valid_chunk_count}"
                        chunk_content[chunk_key] = content
                        valid_chunk_count += 1
                
                # Skip this document group if no valid content found
                if not chunk_content:
                    logger.warning(f"Skipping document group '{group_key}' - no valid content after filtering")
                    continue

                # Prune metadata to keep only relevant fields
                clean_meta, _ = prune_metadata_batch(metadata, keep_keys=KEEP)

                search_results[str(result_id)] = {
                    "title": title,
                    "content": chunk_content,  # Now it's a dict of chunks like ManuSearch expects
                    "date": clean_meta.get('date', '2024'),
                    "score": max(doc.get('score', 1.0) for doc in docs_in_group),  # Use highest score in group
                    # "metadata": clean_meta
                }
                result_id += 1

            logger.info(f"ZillizSearch returned {len(search_results)} document groups from {len(all_docs)} total chunks")
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