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
You are an expert at routing a user question to a vectorstore or web search. The vectorstore holds the content of any documents or URLs the user has uploaded: PDF, DOCX, PPTX, XLSX, HTML, EML, CSV, TXT, Markdown, and web pages. You do not have direct access to the document contents; you can only decide where the question should be routed.

Routing rules:
- Prefer the vectorstore. If the user has uploaded any documents, assume the question is about them unless it is clearly about something else (breaking news, current events, live data, or unrelated general knowledge).
- Route to the vectorstore for questions about prices, values, tables, names, dates, figures, or any factual content that could plausibly appear in an uploaded document.
- Use web search ONLY when the question is clearly about current events, real-time information, or a topic unrelated to any uploaded document.
- If no documents have been uploaded, use web search unless the question is about NVIDIA AI Workbench, nvwb, aiwb, troubleshooting, ngc, cli, svc, wb-svc, logs, gpu, docker, podman, nim, rag, gradio, or jupyterlab.

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
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use five sentences maximum and keep the answer concise but helpful.

Question: {question}
Context: {context}

Answer:
"""

hallucination_prompt = """
You are a grader assessing whether an answer is grounded in and supported by a set of facts. Give a binary 'yes' or 'no' score to indicate whether the answer is grounded in and supported by a set of facts. Your response format is non-negotiable: you must provide the binary score as a JSON with a single key 'score' and no preamble or explanation.

Here are the facts:
\n ------- \n {documents} \n ------- \n
Here is the answer: {generation}
"""

answer_prompt = """
You are a grader assessing whether an answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is useful to resolve a question. Your response format is non-negotiable: you must provide the binary score as a JSON with a single key 'score' and no preamble or explanation.

Here is the answer:
\n ------- \n {generation} \n ------- \n
Here is the question: {question}
"""
