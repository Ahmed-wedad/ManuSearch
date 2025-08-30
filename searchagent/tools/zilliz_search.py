"""
Zilliz Vector Store Search Tool for ManuSearch
Replaces GoogleSearch with vector database search for ENET'Com documents.
"""

import json
import logging
from typing import Dict, List, Any, Tuple, Union
from pydantic import Field
from .basetool import BaseTool

logger = logging.getLogger(__name__)
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

    def prune_metadata_item(self,
        meta: Dict,
        remove_keys: Union[set, None] = None,
        keep_keys:  Union[set, None] = None,
        in_place: bool = False
    ) -> Tuple[Dict, Dict]:
        """
        Prune a single metadata dict.

        Args:
        meta: original metadata dict.
        remove_keys: set of keys to remove (optional).
        keep_keys: set of keys to KEEP (optional). If provided, removal is (all_keys - keep_keys).
        in_place: if True, mutate `meta`; otherwise operate on a shallow copy.

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


    def prune_metadata_batch(self,
        metas: Union[Dict, List[Dict]],
        remove_keys: Union[List[str], set, None] = None,
        keep_keys:  Union[List[str], set, None] = None,
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
        (pruned_metas, removed_per_item)
        - pruned_metas is same type as input metas (dict or list of dicts).
        - removed_per_item is a list of dicts showing keys removed for each item (single-item case returns list with one dict).
        """
        rem = set(remove_keys) if remove_keys is not None else None
        keep = set(keep_keys) if keep_keys is not None else None

        if isinstance(metas, dict):
            p, r = self.prune_metadata_item(metas, remove_keys=rem, keep_keys=keep, in_place=in_place)
            return p, [r]

        pruned_list = []
        removed_list = []
        for item in metas:
            p, r = self.prune_metadata_item(item, remove_keys=rem, keep_keys=keep, in_place=in_place)
            pruned_list.append(p)
            removed_list.append(r)
        return pruned_list, removed_list

    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute vector store search with multi-query, intent-aware retrieval and merging.
        Replaces naive concatenation of queries.
        Returns: search_results mapping stable_key -> {title, content, date, score, metadata, intent_used, sources}
        """
        try:
            # --- helpers (local to keep changes minimal) ---
            import hashlib
            import re
            from urllib.parse import urlparse
            from collections import defaultdict

            def _short_hash(text: str, length: int = 10) -> str:
                return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:length]

            def _sanitize_key(s: str, max_len: int = 120) -> str:
                s = re.sub(r"[^A-Za-z0-9_\-.:/]", "_", str(s or ""))
                return s[:max_len]

            def make_chunk_key(metadata: dict, idx: int, content: str = "") -> str:
                chunk_id = metadata.get("chunk_id")
                document_id = metadata.get("document_id")
                url = metadata.get("url")

                if chunk_id and document_id:
                    return _sanitize_key(f"{document_id}::{chunk_id}")
                if chunk_id:
                    return _sanitize_key(str(chunk_id))
                if document_id:
                    return _sanitize_key(f"{document_id}::chunk_{idx}")
                if url:
                    try:
                        parsed = urlparse(url)
                        domain = parsed.netloc.replace(":", "_") or "url"
                        short = _short_hash(url, 8)
                        return _sanitize_key(f"{domain}::{short}")
                    except Exception:
                        pass
                if content:
                    return _sanitize_key("sha::" + _short_hash(content, 12))
                # fallback
                return _sanitize_key(str(_short_hash(str(idx), 8)))

            # --- extract params ---
            queries = kwargs.get('query', [])
            intents = kwargs.get('intent', [])
            if isinstance(queries, str):
                queries = [queries]
            if isinstance(intents, str):
                intents = [intents]
            primary_intent = intents[0] if intents else None

            # if no queries, return empty structure
            if not queries:
                return {}

            logger.info(f"ZillizSearch executing with queries={queries} intent={primary_intent}")

            # Build query variants (simple: include intent-aware prefix + original query)
            # You may extend this to paraphrases or extra facets
            variants = []
            for q in queries:
                if primary_intent:
                    variants.append(f"{primary_intent.capitalize()}: {q}")
                variants.append(q)  # always include the raw form

            # de-duplicate variants
            seen_q = set()
            variants = [v for v in variants if not (v in seen_q or seen_q.add(v))]

            # Overfetch factor: retrieve more per variant, we'll trim/merge later
            per_variant_k = max(self.top_k, 8)

            # Hold all hits keyed by stable provenance key
            merged = {}  # key -> entry
            # entry: { chunk_id, document_id, page_content, metadata, scores:list, sources:list, intents:list }

            for variant in variants:
                try:
                    hits = self.retriever.get_relevant_docs(variant, k=per_variant_k, intent=primary_intent) or []
                except Exception as e:
                    logger.warning(f"retriever failed for variant '{variant}': {e}")
                    hits = []

                for idx, hit in enumerate(hits):
                    md = hit.get("metadata", {}) or {}
                    content = hit.get("page_content") or hit.get("content") or ""
                    # stable provenance key
                    key = None
                    if md.get("chunk_id") and md.get("document_id"):
                        key = f"{md.get('document_id')}::{md.get('chunk_id')}"
                    else:
                        # fallback to deterministic make_chunk_key
                        key = make_chunk_key(md, idx, content)

                    # normalize key
                    key = str(key)

                    score = float(hit.get("combined_score", hit.get("score", hit.get("dense_score", 1.0))))

                    if key not in merged:
                        # prune metadata upfront (keep whitelist)
                        clean_meta, _ = self.prune_metadata_batch(md, keep_keys=KEEP)
                        # keep original retrieval index for traceability
                        clean_meta["_retrieval_index"] = idx
                        merged[key] = {
                            "chunk_id": md.get("chunk_id"),
                            "document_id": md.get("document_id"),
                            "page_content": content,
                            "metadata": clean_meta,
                            "scores": [score],
                            "sources": [variant],
                            "intents": [primary_intent] if primary_intent else [],
                        }
                    else:
                        merged[key]["scores"].append(score)
                        merged[key]["sources"].append(variant)
                        if primary_intent:
                            merged[key]["intents"].append(primary_intent)

            # Combine aggregated entries into final sorted list
            combined_list = []
            for k, v in merged.items():
                # use max score (you can change to avg/sum)
                combined_score = max(v["scores"]) if v["scores"] else 0.0
                combined_list.append({
                    "key": k,
                    "chunk_id": v.get("chunk_id"),
                    "document_id": v.get("document_id"),
                    "page_content": v.get("page_content"),
                    "metadata": v.get("metadata"),
                    "score": combined_score,
                    "sources": list(dict.fromkeys(v.get("sources", []))),  # unique preserve order
                    "intents": list(dict.fromkeys(v.get("intents", [])))
                })

            # Sort by score desc and pick top_k
            final_sorted = sorted(combined_list, key=lambda x: x["score"], reverse=True)[:self.top_k]

            # Replace this block in your execute() where you build search_results
            # ----------------------------------------------------------------
            # final_sorted is the list of aggregated entries (each entry has 'key', 'page_content', 'metadata', 'score', 'sources', 'intents')

            search_results = {}

            for i, entry in enumerate(final_sorted):
                md = dict(entry.get("metadata", {}) or {})
                # keep or set url so older reader code building url2id works
                url = md.get("url") or md.get("download_url") or ""
                md["url"] = url

                # provenance: keep stable provenance key inside metadata (do NOT use as top-level dict key)
                provenance_key = str(entry.get("key", ""))  # e.g. "doc_...::chunk_0007"
                md["provenance_key"] = provenance_key
                # keep sources and intents for traceability
                md["provenance_sources"] = entry.get("sources", [])
                md["provenance_intents"] = entry.get("intents", [])

                # compute title and date as before
                title = md.get("title") or md.get("source_page_title") or f"ENET'Com Document {i}"
                date = md.get("date", "2024")

                # keep score numeric
                score = float(entry.get("score", 1.0))

                # Use integer-like top-level key (string form) so other components can int(key)
                top_level_key = str(i)

                search_results[top_level_key] = {
                    "title": title,
                    "content": entry.get("page_content", ""),
                    "date": date,
                    "score": score,
                    "metadata": md
                }

            logger.info(f"ZillizSearch returned {len(search_results)} results (intent={primary_intent})")
            return search_results


        except Exception as e:
            logger.error(f"ZillizSearch failed: {e}", exc_info=True)
            return {}

