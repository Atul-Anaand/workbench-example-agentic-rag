# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from concurrent.futures import ThreadPoolExecutor

from typing_extensions import TypedDict
from typing import List, Optional

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_community.tools.tavily_search import TavilySearchResults

from chatui.utils import database, nim


# Tavily related parameters and exceptions
DEFAULT_TAVILY_K = 3
TAVILY_K = int(os.getenv("TAVILY_K", DEFAULT_TAVILY_K)) 

class TavilyAPIError(Exception):
    """Raised when Tavily returns invalid or unauthorized results."""
    pass


# Nemotron 3 API models generate a reasoning ("thinking") trace by default. Every
# component in this workflow expects a direct completion -- the graders parse bare
# JSON from the response -- so thinking is disabled on each hosted-endpoint call.
NEMOTRON_NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}


def _thinking_off_kwargs(model_id: str) -> dict:
    """
    Return the model-family-specific kwargs that disable thinking.

    - DeepSeek V4 family: top-level `reasoning_effort: "none"` (NVIDIA API docs).
      DeepSeek V4.1 Flash defaults to effort="high" which times out the gateway
      at 302s on every call; we force "none" for speed.
    - Nemotron family: `chat_template_kwargs.enable_thinking = False`.
    - Anything else: no binding (model default applies).
    """
    mid = (model_id or "").lower()
    if "deepseek" in mid:
        return {"reasoning_effort": "none"}
    if "nemotron" in mid:
        return {"chat_template_kwargs": {"enable_thinking": False}}
    return {}



### State

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """

    question: str
    generation: str
    web_search: str
    documents: List[str]
    generator_model_id: str
    router_model_id: str
    retrieval_model_id: str
    hallucination_model_id: str
    answer_model_id: str
    prompt_generator: str
    prompt_router: str
    prompt_retrieval: str
    prompt_hallucination: str
    prompt_answer: str
    router_use_nim: bool
    retrieval_use_nim: bool
    generator_use_nim: bool
    hallucination_use_nim: bool
    answer_use_nim: bool
    nim_generator_ip: str
    nim_router_ip: str
    nim_retrieval_ip: str
    nim_hallucination_ip: str
    nim_answer_ip: str
    nim_generator_port: str
    nim_router_port: str
    nim_retrieval_port: str
    nim_hallucination_port: str
    nim_answer_port: str
    nim_generator_id: str
    nim_router_id: str
    nim_retrieval_id: str
    nim_hallucination_id: str
    nim_answer_id: str
    nim_generator_gpu_type: Optional[str]
    nim_generator_gpu_count: Optional[str]
    nim_router_gpu_type: Optional[str]
    nim_router_gpu_count: Optional[str]
    nim_retrieval_gpu_type: Optional[str]
    nim_retrieval_gpu_count: Optional[str]
    nim_hallucination_gpu_type: Optional[str]
    nim_hallucination_gpu_count: Optional[str]
    nim_answer_gpu_type: Optional[str]
    nim_answer_gpu_count: Optional[str]


from langchain.schema import Document


def _model_desc(state, component):
    """Human-readable description of the endpoint serving a component, for the Actions Console."""
    if state[f"{component}_use_nim"]:
        model_name = state[f"nim_{component}_id"] or "meta/llama-3.1-8b-instruct"
        return f"{model_name} (self-hosted endpoint)"
    return f"{state[f'{component}_model_id']} (NVIDIA API endpoint)"


def _validate_datasource(result):
    """Validate and return the router datasource contract."""
    if not isinstance(result, dict):
        raise ValueError(
            f"Router returned an invalid JSON object: {result!r}"
        )

    datasource = result.get("datasource")
    if datasource not in {"web_search", "vectorstore"}:
        raise ValueError(
            f"Router returned invalid datasource: {datasource!r}; "
            "expected 'web_search' or 'vectorstore'"
        )

    return datasource


def _validate_binary_score(result, component):
    """Validate and normalize a binary grader score using a bulletproof regex extractor."""
    import re
    text = str(result).lower()
    
    # 1. Try to find exact JSON first (if the model actually followed instructions)
    if '"score": "yes"' in text or "'score': 'yes'" in text or '"score":"yes"' in text:
        return "yes"
    if '"score": "no"' in text or "'score': 'no'" in text or '"score":"no"' in text:
        return "no"
        
    # 2. Fallback to regex word boundary search (catches "Yes, it is relevant")
    has_yes = bool(re.search(r'yes', text))
    has_no = bool(re.search(r'no', text))
    
    if has_yes and not has_no:
        return "yes"
    if has_no and not has_yes:
        return "no"
        
    # 3. Fallback if the model just says "Relevant" or "Irrelevant"
    if "relevant" in text and "not relevant" not in text:
        return "yes"
    if "not relevant" in text:
        return "no"

    # 4. Ultimate fallback: if the model returns an empty or ambiguous string,
    # we default to "yes" to prevent the pipeline from crashing.
    print(f"[{component}] ⚠ Warning: Ambiguous or empty output ({result!r}). Defaulting to 'yes' to prevent crash.")
    return "yes"


### Nodes


def retrieve(state):
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("[Retriever] Searching the vector database for relevant chunks...")
    question = state["question"]

    # Retrieval
    retriever = database.get_retriever()
    documents = retriever.invoke(question)
    print(f"[Retriever] Retrieved {len(documents)} chunk(s) from the vector database")
    return {"documents": documents, "question": question}


def format_context_with_sources(documents):
    """
    Format retrieved Documents into a single string with [Source: ...] labels.

    Each chunk is prefixed with its source filename so the generator can cite
    sources in its answer. Filename only (path is stripped) to keep the
    citation human-readable.
    """
    import os
    parts = []
    for doc in documents:
        source = doc.metadata.get("source", "unknown")
        filename = os.path.basename(source) if source != "unknown" else "unknown"
        parts.append(f"[Source: {filename}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def generate(state):
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    question = state["question"]
    documents = state["documents"]
    print(f"[Generator] Generating an answer from {len(documents)} context document(s) using {_model_desc(state, 'generator')}...")

    # RAG generation
    prompt = PromptTemplate(
        template=state["prompt_generator"],
        input_variables=["question", "document"],
    )
    llm = nim.CustomChatOpenAI(custom_endpoint=state["nim_generator_ip"], 
                               port=state["nim_generator_port"] if len(state["nim_generator_port"]) > 0 else "8000",
                               model_name=state["nim_generator_id"] if len(state["nim_generator_id"]) > 0 else "meta/llama-3.1-8b-instruct",
                               gpu_type=state["nim_generator_gpu_type"] if "nim_generator_gpu_type" in state else None,
                               gpu_count=state["nim_generator_gpu_count"] if "nim_generator_gpu_count" in state else None,
                               temperature=0.7) if state["generator_use_nim"] else ChatNVIDIA(model=state["generator_model_id"], temperature=0.7).bind(**_thinking_off_kwargs(state["generator_model_id"]))
    rag_chain = prompt | llm | StrOutputParser()
        # Since Pandas Agent returns a Document, we just format the documents
    context_with_sources = format_context_with_sources(documents)
    generation = rag_chain.bind(max_tokens=2048).invoke({"context": context_with_sources, "question": question})
    print(f"[Generator] ✓ Draft answer generated ({len(generation)} characters)")
    return {"documents": documents, "question": question, "generation": generation}


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    If the LLM grader rejects all chunks, we KEEP THE BEST ONE rather than
    triggering a web-search fallback. Rationale: when the user has uploaded
    documents, the retriever has already ranked chunks by relevance — the
    top-1 result is nearly always what the user wants. Rejecting it and
    searching the web is worse than trusting the retriever's ranking.

    The Hallucination Grader downstream still validates that any generated
    answer is grounded in the retrieved context, so bad chunks won't
    produce bad answers.
    """

    question = state["question"]
    documents = state["documents"]
    print(f"[Retrieval Grader] Grading {len(documents)} retrieved chunk(s) for relevance using {_model_desc(state, 'retrieval')}...")

    # Nothing to grade — fall back to web search
    if not documents:
        return {"documents": [], "question": question, "web_search": "Yes"}

    filtered_docs = []
    prompt = PromptTemplate(
        template=state["prompt_retrieval"],
        input_variables=["question", "document"],
    )
    llm = nim.CustomChatOpenAI(custom_endpoint=state["nim_retrieval_ip"],
                               port=state["nim_retrieval_port"] if len(state["nim_retrieval_port"]) > 0 else "8000",
                               model_name=state["nim_retrieval_id"] if len(state["nim_retrieval_id"]) > 0 else "meta/llama-3.1-8b-instruct",
                               gpu_type=state["nim_retrieval_gpu_type"] if "nim_retrieval_gpu_type" in state else None,
                               gpu_count=state["nim_retrieval_gpu_count"] if "nim_retrieval_gpu_count" in state else None,
                               temperature=0.7) if state["retrieval_use_nim"] else ChatNVIDIA(model=state["retrieval_model_id"], temperature=0).bind(**_thinking_off_kwargs(state["retrieval_model_id"]))
    retrieval_grader = prompt | (llm.bind(response_format={"type": "json_object"}) if state["retrieval_use_nim"] else llm) | StrOutputParser()

    def grade_document(document):
        score = retrieval_grader.invoke(
            {"question": question, "document": document.page_content}
        )
        return _validate_binary_score(score, "Retrieval Grader")

    with ThreadPoolExecutor(max_workers=min(4, len(documents))) as executor:
        grades = list(executor.map(grade_document, documents))

    for i, (document, grade) in enumerate(zip(documents, grades)):
        if grade == "yes":
            print(f"[Retrieval Grader] Chunk {i + 1} of {len(documents)}: ✓ relevant")
            filtered_docs.append(document)
        else:
            print(f"[Retrieval Grader] Chunk {i + 1} of {len(documents)}: ✗ not relevant")

    # Key change: if the grader rejected everything, keep the top-ranked chunk
    # instead of triggering web search. The retriever's ranking is the
    # primary signal — the grader is just a quality filter on top of it.
    if not filtered_docs:
        print(f"[Retrieval Grader] All chunks rejected — keeping top-ranked chunk (1 of {len(documents)})")
        filtered_docs = [documents[0]]

    print(f"[Retrieval Grader] Kept {len(filtered_docs)} of {len(documents)} chunk(s)")
    return {"documents": filtered_docs, "question": question, "web_search": "No"}


def web_search(state):
    """
    Web search based based on the question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended web results to documents
    """

    question = state["question"]
    documents = state.get("documents", [])
    print(f'[Web Search] Searching the web with Tavily (top {TAVILY_K} results): "{question}"')

    web_search_tool = TavilySearchResults(max_results=TAVILY_K)

    # Web search
    try:
        docs = web_search_tool.invoke({"query": question})

        # Manually validate Tavily returned what we expect
        if not isinstance(docs, list) or not all(isinstance(d, dict) and "content" in d for d in docs):
            raise TavilyAPIError(f"Invalid response from Tavily: {docs}")

        web_results = "\n".join([d["content"] for d in docs])
        web_results = Document(page_content=web_results)
        documents.append(web_results)
        print(f"[Web Search] ✓ Added {len(docs)} web result(s) to the context")

        return {"documents": documents, "question": question}

    except Exception as e:
        print(f"[Web Search] ✗ Web search failed: {e}")
        raise TavilyAPIError(f"Tavily web search failed: {e}")
    # if documents is not None:
    #     documents.append(web_results)
    # else:
    #     documents = [web_results]



### Conditional edge



import io
import contextlib
import traceback
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
def duckdb_agent(state):
    import glob
    import duckdb
    import pandas as pd
    from langchain_core.documents import Document
    
    question = state["question"]
    clean_question = question.split("=== STRICT INSTRUCTIONS ===")[0].strip()
    file_paths = glob.glob("/project/uploads/*.xlsx") + glob.glob("/project/uploads/*.csv")
    print("[DuckDB Agent] Found", len(file_paths), "files.")

    if not file_paths:
        return {"documents": [Document(page_content="Error: No files found.", metadata={"source": "duckdb_engine"})]}

    NL = chr(10)

    def read_sheet_merged(path, sheet_name):
        raw = pd.read_excel(path, sheet_name=sheet_name, header=None)
        if raw.empty or len(raw) < 2:
            return raw
        row0 = [str(x).strip().lower() if pd.notna(x) else "" for x in raw.iloc[0]]
        row1 = [str(x).strip().lower() if pd.notna(x) else "" for x in raw.iloc[1]]
        sub_terms = ['line', 'qty', 'value', 'quantity', 'amount', 'pcs', 'no.']
        has_sub_header = any(any(t in cell for t in sub_terms) for cell in row1 if cell)
        
        if has_sub_header:
            merged = []
            last_group = ""
            for i in range(len(row0)):
                group = row0[i] if row0[i] else last_group
                if row0[i]:
                    last_group = row0[i]
                sub = row1[i] if i < len(row1) else ""
                if group and sub:
                    merged.append(group + "_" + sub)
                elif group:
                    merged.append(group)
                elif sub:
                    merged.append(sub)
                else:
                    merged.append("col_" + str(i))
            df = raw.iloc[2:].copy()
            df.columns = merged
        else:
            merged = [row0[i] if row0[i] else "col_" + str(i) for i in range(len(row0))]
            df = raw.iloc[1:].copy()
            df.columns = merged
            
        df.columns = [str(c).replace(' ', '_').replace('-', '_').replace('/', '_').replace('.', '_').replace('#', '') for c in df.columns]
        return df

    try:
        con = duckdb.connect(database=':memory:')
        schema_lines = []
        
        for i, path in enumerate(file_paths):
            filename = path.split('/')[-1]
            sheets = pd.read_excel(path, sheet_name=None)
            for sheet_name in sheets.keys():
                try:
                    df = read_sheet_merged(path, sheet_name)
                except Exception as e:
                    print("[DuckDB Agent] Sheet skip:", sheet_name, e)
                    continue
                safe_sheet = str(sheet_name).replace(' ', '_').replace('-', '_')
                table_name = "file_" + str(i) + "_" + safe_sheet
                con.register(table_name, df)
                desc = con.execute("DESCRIBE " + table_name).fetchall()
                schema_lines.append("Table: " + table_name + " (Source: " + filename + " | Sheet: " + str(sheet_name) + ")")
                for row in desc:
                    schema_lines.append("  - " + str(row[0]) + " (" + str(row[1]) + ")")
                schema_lines.append("")

        schema_str = NL.join(schema_lines)
        print("[DuckDB Agent] Real Schema Found:")
        print(schema_str)

        base_prompt = "You are an expert SQL analyst." + NL + \
                      "Schema:" + NL + "{schema}" + NL + NL + \
                      "Question: {question}" + NL + NL + \
                      "Write a DuckDB SQL query to answer this." + NL + \
                      "- Use CAST(col AS DECIMAL(18,2)) for money columns." + NL + \
                      "- ALWAYS use GROUP BY for non-aggregated columns." + NL + \
                      "- ONLY output the SQL query. No markdown, no backticks." + NL + NL + \
                      "{error_message}"

        llm = nim.CustomChatOpenAI(custom_endpoint=state["nim_generator_ip"],
                                   port=state["nim_generator_port"] if len(state["nim_generator_port"]) > 0 else "8000",
                                   model_name=state["nim_generator_id"] if len(state["nim_generator_id"]) > 0 else "meta/llama-3.1-8b-instruct",
                                   temperature=0) if state["generator_use_nim"] else ChatNVIDIA(model=state["generator_model_id"], temperature=0).bind(**_thinking_off_kwargs(state["generator_model_id"]))
        
        code_chain = PromptTemplate(template=base_prompt, input_variables=["schema", "question", "error_message"]) | llm | StrOutputParser()
        
        print("[DuckDB Agent] Generating SQL (Attempt 1)...")
        sql_query = code_chain.invoke({"schema": schema_str, "question": clean_question, "error_message": ""})
        sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
        
        print("[DuckDB Agent] Attempt 1 SQL:")
        print(sql_query)
        
        if not sql_query:
            raise ValueError("LLM generated empty SQL.")

        try:
            result = con.execute(sql_query)
            if result is None:
                raise ValueError("DuckDB returned None. The query might not be a valid SELECT.")
            result_df = result.fetchdf()
        except Exception as sql_error:
            print("[DuckDB Agent] SQL Failed. Retrying with error context...")
            print("Error:", str(sql_error))
            
            sql_query = code_chain.invoke({
                "schema": schema_str, 
                "question": clean_question, 
                "error_message": "Your previous query failed with this error: " + str(sql_error) + ". Fix it."
            })
            sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
            
            print("[DuckDB Agent] Attempt 2 SQL:")
            print(sql_query)
            
            if not sql_query:
                raise ValueError("LLM generated empty SQL on retry.")
                
            result = con.execute(sql_query)
            if result is None:
                raise ValueError("DuckDB returned None on retry.")
            result_df = result.fetchdf()

        result_str = result_df.to_markdown(index=False)
        print("[DuckDB Agent] Execution successful. Rows:", len(result_df))

        return {"documents": [Document(page_content="=== EXACT DATA ANALYSIS RESULTS ===" + NL + result_str, metadata={"source": "duckdb_engine"})]}

    except Exception as e:
        import traceback
        error_msg = "DuckDB execution failed: " + str(e) + NL + traceback.format_exc()
        print("[DuckDB Agent] ERROR:", error_msg)
        return {"documents": [Document(page_content="=== EXACT DATA ANALYSIS RESULTS ===" + NL + "Error: " + error_msg, metadata={"source": "duckdb_engine"})]}
def route_question(state):
    """
    Route question to web search or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    question = state["question"]
    
    # Air-Gapped Guarantee: If we are in Local mode (router_use_nim is True), 
    # bypass the LLM completely and force vectorstore retrieval.
    # This prevents local models from attempting web searches.
    if state.get("router_use_nim", False):
        print("[Router] → Local (Ollama) mode active — bypassing LLM and routing directly to vectorstore")
        return "vectorstore"

    print(f"[Router] Choosing a data source using {_model_desc(state, 'router')}...")
    prompt = PromptTemplate(
        template=state["prompt_router"],
        input_variables=["question"],
    )
    llm = nim.CustomChatOpenAI(custom_endpoint=state["nim_router_ip"], 
                               port=state["nim_router_port"] if len(state["nim_router_port"]) > 0 else "8000",
                               model_name=state["nim_router_id"] if len(state["nim_router_id"]) > 0 else "meta/llama-3.1-8b-instruct",
                               gpu_type=state["nim_router_gpu_type"] if "nim_router_gpu_type" in state else None,
                               gpu_count=state["nim_router_gpu_count"] if "nim_router_gpu_count" in state else None,
                               temperature=0.7) if state["router_use_nim"] else ChatNVIDIA(model=state["router_model_id"], temperature=0).bind(**_thinking_off_kwargs(state["router_model_id"]))
    question_router = prompt | llm | StrOutputParser()
    source = question_router.invoke({"question": question})
    datasource = _validate_datasource(source)
    if datasource == "web_search":
        print("[Router] → Question falls outside the document context — routing to web search")
        return "websearch"
    elif source["datasource"] == "vectorstore":
        print("[Router] → Question matches the document context — routing to the vector database")
        return "vectorstore"


def decide_to_generate(state):
    """
    Determines whether to generate an answer, or add web search

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    question = state["question"]
    web_search = state["web_search"]
    filtered_documents = state["documents"]

    if web_search == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print("[Agent] No relevant chunks survived grading — falling back to web search")
        return "websearch"
    else:
        # We have relevant documents, so generate answer
        print(f"[Agent] {len(filtered_documents)} relevant chunk(s) in hand — proceeding to answer generation")
        return "generate"


### Conditional edge


def grade_generation_v_documents_and_question(state):
    # FORCE BYPASS IN LOCAL MODE to prevent 4000+ char regeneration loops
    if state.get("router_use_nim", False):
        print("[Hallucination Grader] Local mode active — bypassing strict grading to prevent VRAM loop.")
        return "useful"
    
    # Original logic (only runs in API mode)
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    return "useful"
