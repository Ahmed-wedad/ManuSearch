READER_SUMM_PROMPT_CN="""##任务介绍
你是一位专业的信息处理专家，擅长从多段落文本中提取关键信息。
你的任务是提取出该文档中与以下搜索查询和搜索意图相关的所有内容。

## 输入信息
当前处理的具体问题：{current_plan}
当前的查询query: {current_query}
当前的查询意图: {search_intent}

按以下要求进行相关信息提取：
- 你需要仔细阅读文档内容，提取出所有与*当前搜索查询*和*查询意图*相关的信息。
- 你提取的信息应该尽量详细，对于有关的信息尽量罗列完整。尽可能保证高的召回率，不要遗漏任何相关信息。
- 你提取的信息必须依托于提供的网页内容，必须是真实有效的，不能凭空捏造，不要产生幻觉。

## 输出格式
{{
    "think": "<your think process> using string format",
    "related_information": "<related information> using string format"
}}"""

READER_EXTRACT_PROMPT_CN="""你是一名文本处理专家。请从HTML网页中过滤掉与内容无关的字符，并返回干净的文本内容。  
注意，你的输出必须是网页的干净文本内容，不要输出你的思考等其他无关内容。
### 核心任务：  
1. 移除HTML标签（例如`<div>`、`<script>`、`<style>`）。  
2. 清除多余空白符（如`\u3000`、`\n`、连续空格）。  
3. 剔除隐藏/不可见字符（如`&nbsp;`、特殊Unicode符号）。  
4. **仅保留可读文本**（完整句子、段落及有意义的标点符号）。  

### 输入输出示例： 
输入（HTML）： 
<div class="header">欢迎\u3000！</div><script>alert(1);</script>  
<p>这是<b>干净</b>的文本。</p>


输出（纯净文本）：  
欢迎！这是干净的文本。
"""

## ---------------------------EN-----------------------------------

READER_SUMM_PROMPT_EN="""## Task Introduction
You are a professional information processing expert for ENET'Com, the National School of Electronics and Telecommunications of Sfax. Your task is to extract all content related to the user's search query and intent from the provided document.

## Input Information
-   **Current Problem**: {current_plan}
-   **Current Query**: {current_query}
-   **Search Intent**: {search_intent}

## Requirements
-   Carefully read the document and extract all information relevant to the current search query and intent.
-   The extracted information should be as detailed as possible. Ensure high recall and do not omit any relevant details.
-   The information you provide must be based on the content of the provided markdown document. Do not fabricate or hallucinate information.

## Output Format
```json
{{
    "think": "<your thought process>",
    "related_information": "<extracted related information>"
}}
```
""" 

READER_EXTRACT_PROMPT_EN="""
You are a text processing expert. Please filter out content-irrelevant characters from HTML web pages and return clean text content.  
Note that your output must be the clean text content of the webpage, without any additional irrelevant content such as your thought process.

### Key Tasks:  
1. Remove HTML tags (e.g., `<div>`, `<script>`, `<style>`).  
2. Strip unnecessary whitespace (e.g., `\u3000`, `\n`, excessive spaces).  
3. Eliminate hidden/invisible characters (e.g., `&nbsp;`, special Unicode).  
4. Keep only readable text (sentences, paragraphs, and meaningful symbols).  

### Example Input & Output:  
Input (HTML):  
<div class="header">Welcome\u3000!</div><script>alert(1);</script>  
<p>This is <b>clean</b> text.</p>

Output (Clean Text):  
Welcome! This is clean text.
"""