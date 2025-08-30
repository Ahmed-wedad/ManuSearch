from ..utils.utils import *
from ..utils.cache import WebPageCache
from ..models.basellm import GPTAPI, BaseStreamingAgent
from ..tools.visitpage import VisitPage
from concurrent.futures import ThreadPoolExecutor, as_completed 
import os, ast
import concurrent.futures
import re
from collections import defaultdict
from typing import Dict, Tuple, List
class Reader(BaseStreamingAgent):
    def __init__(self, llm:GPTAPI, webpage_cache, summary_prompt, extract_prompt, search_api_key, proxy, **baseconfig):
        self.llm = llm
        self.summary_prompt = summary_prompt
        self.extract_prompt = extract_prompt
        self.input_prompt = """## Date:{date}
        ## Title:{title}
        ## Content:{content}"""
        self.visitpage = VisitPage(api_key=search_api_key, timeout=1, proxy=proxy)
        self.webpage_cache = webpage_cache

        super().__init__(llm, **baseconfig)

    

    MAX_CHARS = 16192
    SEPARATOR = "\n\n==========\n\n"

    def _extract_chunks_from_item(self,key: str, item: Dict) -> List[Dict]:
        """
        Normalize an item into a list of chunk dicts:
        each chunk dict contains: chunk_id, text, score, chunk_timestamp, metadata
        """
        md = item.get("metadata", {}) or {}
        score = item.get("score", item.get("dense_score", 1.0))
        content = item.get("content", "") or ""

        chunks = []
        if isinstance(content, dict):
            # content provided as mapping of chunk_id->text
            for cid, txt in content.items():
                chunks.append({
                    "chunk_id": cid,
                    "text": txt or "",
                    "score": score,
                    "chunk_timestamp": md.get("chunk_timestamp"),
                    "metadata": md,
                    "orig_key": key
                })
        else:
            # single string content
            chunks.append({
                "chunk_id": md.get("chunk_id", key),
                "text": content,
                "score": score,
                "chunk_timestamp": md.get("chunk_timestamp"),
                "metadata": md,
                "orig_key": key
            })
        return chunks

    def _sort_key_for_chunk(self,c: Dict):
        # prefer timestamp, then numeric suffix of chunk_id, else 0
        ts = c.get("chunk_timestamp")
        if ts:
            # try to keep whatever ordering the timestamp string provides
            return (0, ts)
        cid = c.get("chunk_id") or ""
        m = re.search(r"(\d+)$", str(cid))
        if m:
            return (1, int(m.group(1)))
        return (2, 0)

    def group_and_build_messages(self,
        search_results: Dict,
        system_prompt: str,
        input_prompt: str,
        max_chars: int = MAX_CHARS,
        separator: str = SEPARATOR
    ) -> Tuple[Dict, Dict]:
        """
        Takes search_results (mapping keys->items as you showed) and returns:
        - messages: dict of chat message lists ready to be passed to the reader
        - url_to_chunks: mapping of group_key -> concatenated text (for debugging/display)
        """
        # 1) Group chunks by a stable document key (prefer document_id, fallback to url or document_source)
        groups = defaultdict(list)
        for i, (key, item) in enumerate(search_results.items()):
            md = item.get("metadata", {}) or {}
            doc_id = md.get("document_id") or md.get("url") or md.get("document_source") or f"__key__{key}"
            chunks = self._extract_chunks_from_item(key, item)
            groups[doc_id].extend(chunks)

        messages = {}
        url_to_chunks = {}

        # 2) For each document, sort chunks and concatenate them up to max_chars
        for doc_id, chunks in groups.items():
            # sort deterministically
            chunks_sorted = sorted(chunks, key=self._sort_key_for_chunk)

            # build a concise title / date / combined score from members
            titles = [c["metadata"].get("title") for c in chunks_sorted if c["metadata"].get("title")]
            title = titles[0] if titles else ""
            dates = [c["metadata"].get("date") for c in chunks_sorted if c["metadata"].get("date")]
            # prefer the latest date if present, else fallback to the first one we have
            date = max(dates) if dates else (chunks_sorted[0]["metadata"].get("date") if chunks_sorted else "")

            # combined score heuristic (average)
            scores = [float(c.get("score", 1.0)) for c in chunks_sorted if c.get("score") is not None]
            combined_score = sum(scores) / len(scores) if scores else 1.0

            # Concatenate with labels; keep track of provenance ids included
            included_chunk_ids = []
            parts = []
            total_len = 0
            for idx, c in enumerate(chunks_sorted):
                text = (c.get("text") or "").strip()
                if not text:
                    continue
                chunk_label = f"Chunk {c.get('chunk_id','%s_%d' % (doc_id, idx))} (score:{c.get('score'):.3f})"
                part = f"{chunk_label}\n{text}"
                # if adding this part would exceed max_chars, try to truncate the part instead of discarding
                projected_len = total_len + len(part) + len(separator)
                if projected_len > max_chars:
                    # how many chars left?
                    remaining = max_chars - total_len - len(separator) - 3  # reserve for ellipsis
                    if remaining <= 0:
                        break
                    truncated = part[:remaining] + "..."
                    parts.append(truncated)
                    included_chunk_ids.append(c.get("chunk_id"))
                    total_len += len(truncated) + len(separator)
                    break
                parts.append(part)
                included_chunk_ids.append(c.get("chunk_id"))
                total_len += len(part) + len(separator)

            if not parts:
                # nothing to send for this document
                continue

            concat_text = separator.join(parts)
            # prepare the final content using the existing input_prompt (date/title/content)
            content_for_reader = input_prompt.format(date=(date or "2024"), title=(title or ""), content=concat_text)

            # build chatbox exactly like your current code expects
            chatbox = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content_for_reader}
            ]

            # pick a stable message key: prefer doc_id, but make it unique string (manusearch expected str keys earlier)
            message_key = str(doc_id)

            messages[message_key] = chatbox
            # keep a mapping for debugging / possible re-summarization
            url_to_chunks[message_key] = {
                "concatenated_text": concat_text,
                "included_chunk_ids": included_chunk_ids,
                "combined_score": combined_score,
                "title": title,
                "date": date
            }

        return messages, url_to_chunks

    def get_llm_summ(self, search_results:dict, question, user_query, search_intent, current_query):
        # url2id = {value['url']: key for key,value in search_results.items()}
        # select_urls = []
        # for key in url2id.keys():
        #     if key:
        #         select_urls.append(key)

        # # First read the stored url from cache
        # cached_results = {} 
        # if self.webpage_cache:
        #     for key in select_urls:
        #         success, content = self.webpage_cache.get_content(url=key)
        #         if success:
        #             cached_results[url2id[key]] = content 
        #             select_urls.remove(key)

        # cleaned_tool_return = {}

        # # If there is an unstored url, access it
        # if select_urls:
        #     with timeit("reader fetch all pages"):
        #         tool_return = self.visitpage.execute(select_urls=select_urls, search_results=search_results, url_to_chunk_score = None, webpage_cache=self.webpage_cache)# 筛选出提取出来正文的进行进一步summary

        #     if not tool_return:
        #         print("Visitpage couldn't execute")
        #         return search_results, "Visitpage couldn't execute"

        #     cleaned_tool_return = self.extract_text(tool_return=tool_return)
        #     # cache accessed urls        
        #     for items in cleaned_tool_return.values():
        #         self.webpage_cache.store_content(url=items['url'], data=items)

        # # merge cache hit and miss urls
        # if cleaned_tool_return:
        #     cleaned_tool_return.update(cached_results)
        # else:
        #     cleaned_tool_return = cached_results
        
        messages, url_to_chunks={}, {}
        system_prompt = self.summary_prompt.format(current_plan=question, user_query = user_query, search_intent=search_intent, current_query=current_query)
        # for key, item in search_results.items():
        #     url_to_chunks[key] = item['content']
        #     if 'content' not in item or not item['content']:
        #         continue
        #     # chunked_str = '=========='.join([f"Chunk {key}:{value}" for key, value in item['content'].items()])
        #     # chunked_str = chunked_str[:16192]
        #     if 'title' not in item:
        #         item['title'] = ""
        #     content = self.input_prompt.format(date=item['date'], title=item['title'], content=item['content'])
        #     chatbox=[
        #         {"role": 'system', 'content': system_prompt},
        #         {'role': 'user', 'content': content}
        #     ]
        #     messages[key]=chatbox

        messages, url_to_chunks = self.group_and_build_messages(
    search_results=search_results,
    system_prompt=system_prompt,
    input_prompt=self.input_prompt,   # your existing prompt string with {date},{title},{content}
    max_chars=16192
)
        with timeit("reader llm summ"):
            url2summ = {}
            with ThreadPoolExecutor(max_workers=5) as executor:
                future_to_url = {
                    executor.submit(self.llm.chat, chatbox): url
                    for url, chatbox in messages.items()
                }
                try:
                    for future in concurrent.futures.as_completed(future_to_url):
                        url = future_to_url[future]
                        try:
                            # Set the timeout period
                            ret = future.result(timeout=10)
                            llm_summ = ret.content
                            url2summ[url] = llm_summ

                        except concurrent.futures.TimeoutError:
                            print(f"Task for {url} timed out")
                            url2summ[url] = "Timeout Error"
                        
                        except Exception as e:
                            print(f"Task for {url} generated an exception: {e}")
                            # Handle other possible exceptions
                            url2summ[url] = f"Error: {str(e)}"
                except concurrent.futures.TimeoutError:
                    raise ValueError("concurrent.futures TimeoutError!")

        llm_summs = url2summ # {url: summ}
        for key in llm_summs:
            reader_json = parse_resp_to_json(llm_summs[key])
            try:
                llm_summs[key] = reader_json.get('related_information', '')
            except:
                pass
        for key, page in search_results.items():
            if key in llm_summs:
                page['content'] = llm_summs[key]
            else:
                page['content'] = ""
        return search_results, None # {url: {chunk_dict, scores}}

    def extract_text(self, tool_return):
        messages = {}
        for item in tool_return.values():
            url = item['url']
            messages[url] = []
            if 'content' not in item or not item['content']:
                continue
            if isinstance(item['content'], str):
                chunked_str = item['content']
            chunked_str = ''.join(list(item['content'].values()))
            if len(chunked_str) > 128000:
                chunked_str = chunked_str[:64000]
            if len(chunked_str) > 16192: 
                content = chunked_str[:16192] 
                i=0
                while 16192*(i+1) <= len(chunked_str):
                    content = chunked_str[16192*i:16192*(i+1)] 
                    i += 1
                    chatbox=[
                        {"role": 'system', 'content': self.extract_prompt},
                        {'role': 'user', 'content': content}
                    ]
                    messages[url].append(chatbox)
            else:
                content = chunked_str
                chatbox=[
                    {"role": 'system', 'content': self.extract_prompt},
                    {'role': 'user', 'content': content}
                ]
                messages[url].append(chatbox)

        webtexts = {}
        inputs = []
        with ThreadPoolExecutor(max_workers=20) as executor:
            for url, chatboxes in messages.items():
                for chatbox in chatboxes:
                    inputs.append((url, chatbox))
            future_to_url = {
                executor.submit(self.llm.chat, chatbox): url
                for url, chatbox in inputs
            }
            try:
                for future in concurrent.futures.as_completed(future_to_url):
                    url = future_to_url[future]
                    try:
                    
                        ret = future.result(timeout=10)
                        text = ret.content
                        if url in webtexts:
                            webtexts[url] += text
                        else:
                            webtexts[url] = text

                    except concurrent.futures.TimeoutError:
                        print(f"Task for {url} timed out")
                        webtexts[url] = "Timeout Error"

                    except Exception as e:
                        print(f"Task for {url} generated an exception: {e}")
                        webtexts[url] = f"Error: {str(e)}"
                        
            except concurrent.futures.TimeoutError:
                raise ValueError("concurrent.futures TimeoutError!")

        for item in tool_return.values():
            url = item['url']
            if 'content' not in item or not item['content']:
                continue
            if url in webtexts and webtexts[url]:
                item['content'] = self.chunk_content(webtexts[url], chunk_size=512)

        return tool_return
    
    def chunk_content(self, text, chunk_size=512):
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        chunk_dict = {i: chunk for i, chunk in enumerate(chunks)}
        return chunk_dict