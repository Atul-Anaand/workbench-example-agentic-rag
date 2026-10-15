# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

""" These are the default prompts used by the applications for the NVIDIA Nemotron 3 model family (Nano, Super, and Ultra).

Nemotron 3 models share a common chat template, so the same plain-text prompts work across all
three sizes. Unlike older model families, no raw chat-format tokens belong in the prompt text;
the API applies the chat template server-side. Reasoning ("thinking") is controlled separately
via the request's chat_template_kwargs, not from within the prompt.
"""

router_prompt = """
You are an expert at routing a user question to a vectorstore or web search. The vectorstore holds the content of any documents or URLs the user has uploaded (PDFs, DOCX, PPTX, XLSX, HTML, EML, CSV, TXT, Markdown, web pages). You do not have direct access to the document contents; you can only decide where the question should be routed.

Routing rules (in priority order):
1. PREFER the vectorstore. If the user has uploaded ANY documents, assume the question is about them UNLESS it clearly requires live information (today's news, real-time prices, sports scores, breaking events).
2. General-sounding questions like "who are the top players?", "what is the market size?", "what is the revenue?" should route to the vectorstore — the uploaded documents may contain the answer, and the retrieval grader downstream will filter out truly irrelevant content.
3. Factual questions about specific numbers, names, dates, sections, tables, or values MUST route to the vectorstore.
4. Use web search ONLY for questions that clearly require information outside any uploaded document: current events, live data, or general knowledge that could not plausibly appear in a client document.
5. If NO documents have been uploaded, route to the vectorstore for NVIDIA AI Workbench topics (nvwb, aiwb, ngc, cli, gpu, docker, rag, etc.); otherwise route to web search.

Give a binary choice 'web_search' or 'vectorstore' based on the question. Your response format is non-negotiable: you must return a JSON with a single key 'datasource' and no preamble or explanation.

Question to route: {question}
"""

retrieval_prompt = """
You are a grader assessing relevance of a retrieved document to a user question. The user has uploaded a document and wants an answer derived from it. Grade the retrieved chunk as RELEVANT unless it is completely unrelated to the question topic.

IMPORTANT: The extracted text may not preserve formatting, tags, table structure, or field names from the original file. Do NOT reject a chunk just because a specific keyword (like "h1", "cell", "row", "column", "tag", or "table") from the question does not appear verbatim in the chunk. If the chunk plausibly came from the uploaded document and the question is about its content, grade it as relevant.

When in doubt, grade as relevant. The downstream hallucination grader will catch genuinely bad matches.

Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
Your response format is non-negotiable: you must provide the binary score as a JSON with a single key 'score' and no preamble or explanation.

Here is the retrieved document: 
 {document} 

Here is the user question: {question}
"""

generator_prompt = """
You are an assistant for question-answering tasks. Answer using ONLY the retrieved context below.

RIGOR RULES (apply to every answer, without exception):
1. If the answer is not clearly supported by the context, say "I don't know" or "This is not in the provided documents" rather than guessing. A false answer is worse than no answer.
2. Never invent citations, statistics, dates, or details not present in the context. If unsure whether something came from the context, say "NO SOURCE — VERIFY" rather than presenting it as fact.
3. For every specific number, date, or figure: use it exactly as it appears in the context. Do not round, paraphrase, or restate it differently.
4. Recompute any arithmetic or unit conversion explicitly before stating the result.
5. Before finalizing, check that figures, names, and claims are internally consistent within your answer.
6. If something plausibly doesn't exist in the context (a value, a section, a figure), say so directly — "This is not in the provided documents" — rather than defaulting to a vague "I don't know."
7. If only part of a question can be answered with confidence, say explicitly which part is unanswered. Do not quietly answer an easier question instead.
8. Do not use hedging language ("it's possible that...", "some sources suggest...") to excuse a guess. If you don't know, say so plainly.
9. Never produce a direct quote you can't trace to an exact passage in the context. If unsure it's exact, paraphrase instead.

CITING SOURCES: Cite the source file name in square brackets whenever your answer draws on context. Each chunk is prefixed with "[Source: filename.ext]". Use exact filenames. Example: "According to [06.Feedback_Logic_for_IA.docx], the bottom line is...". If information comes from multiple sources, cite each one.

Keep the answer to five sentences maximum unless the question requires a structured multi-part response.

Question: {question}
Context: {context}

Answer:
"""

hallucination_prompt = """
You are a grader assessing whether an answer is grounded in and supported by a set of facts.

Grade as 'yes' if:
- The answer's information can be found in or reasonably inferred from the provided facts.
- The answer is short or terse (like "42", "Test HTML", or a single phrase) and its content appears in the facts.
- The answer states that the requested information is NOT present in the facts (e.g. "This is not in the provided documents", "I don't know", "No such data exists in the context"). Answers reporting an absence are valid and correctly grounded.

Grade as 'no' ONLY if:
- The answer contains a number, date, name, or figure that does NOT appear in the facts (invented content).
- The answer invents a citation or detail not present in the facts.

Your response format is non-negotiable: return JSON with a single key 'score' and no preamble or explanation.

Here are the facts:

 ------- 
 {documents} 
 ------- 

Here is the answer: {generation}
"""

answer_prompt = """
You are a grader assessing whether an answer addresses a user question.

Grade as 'yes' if:
- The answer is on-topic and attempts to resolve the question.
- The answer is short (like "42" or a single phrase) but contains the requested fact.
- The answer states that the requested information is NOT present in the context (e.g. "not in the provided documents", "no such data exists", "I don't know"). Absence-answers are valid responses when the context truly lacks the information.

Grade as 'no' ONLY if:
- The answer is clearly off-topic or evasive without stating a reason.
- The answer refuses to engage with a question whose answer IS present in the context.

Your response format is non-negotiable: return JSON with a single key 'score' and no preamble or explanation.

Here is the answer:

 ------- 
 {generation} 
 ------- 

Here is the question: {question}
"""
