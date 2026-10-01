import importlib
import json
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch


ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "code"

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


class TestLocalOllamaEmbeddings(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # database.py checks/downloads NLTK resources at import time.
        # Prevent the unit suite from ever attempting a network download.
        import nltk

        cls._nltk_find = nltk.data.find
        cls._nltk_download = nltk.download
        nltk.data.find = lambda *_args, **_kwargs: True
        nltk.download = lambda *_args, **_kwargs: True

        cls.database = importlib.import_module("chatui.utils.database")

        nltk.data.find = cls._nltk_find
        nltk.download = cls._nltk_download

    def test_embed_query_returns_single_embedding(self):
        fake_response = SimpleNamespace(
            read=lambda: json.dumps({"embeddings": [[0.1, 0.2, 0.3]]}).encode("utf-8")
        )

        class ResponseContext:
            def __enter__(self):
                return fake_response

            def __exit__(self, exc_type, exc_value, traceback):
                return False

        with patch.object(
            self.database.urllib.request,
            "urlopen",
            return_value=ResponseContext(),
        ) as mocked_urlopen:
            adapter = self.database.OllamaEmbeddings(
                model="test-embedding",
                url="http://test-ollama/api/embed",
            )

            result = adapter.embed_query("hello world")

        self.assertEqual(result, [0.1, 0.2, 0.3])
        self.assertEqual(mocked_urlopen.call_count, 1)

        request = mocked_urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))

        self.assertEqual(payload["model"], "test-embedding")
        self.assertEqual(payload["input"], ["hello world"])

    def test_embed_documents_batches_at_32(self):
        responses = [
            {"embeddings": [[float(i)] for i in range(32)]},
            {"embeddings": [[32.0]]},
        ]

        class ResponseContext:
            def __init__(self, payload):
                self.payload = payload

            def __enter__(self):
                return SimpleNamespace(
                    read=lambda: json.dumps(self.payload).encode("utf-8")
                )

            def __exit__(self, exc_type, exc_value, traceback):
                return False

        def fake_urlopen(request, timeout=120):
            self.assertEqual(timeout, 120)
            payload = responses.pop(0)
            return ResponseContext(payload)

        texts = [f"text-{i}" for i in range(33)]

        with patch.object(
            self.database.urllib.request,
            "urlopen",
            side_effect=fake_urlopen,
        ) as mocked_urlopen:
            adapter = self.database.OllamaEmbeddings(
                model="test-embedding",
                url="http://test-ollama/api/embed",
            )
            result = adapter.embed_documents(texts)

        self.assertEqual(len(result), 33)
        self.assertEqual(mocked_urlopen.call_count, 2)
        self.assertEqual(responses, [])

    def test_missing_embeddings_raises_runtime_error(self):
        class ResponseContext:
            def __enter__(self):
                return SimpleNamespace(
                    read=lambda: b'{"model":"test-embedding"}'
                )

            def __exit__(self, exc_type, exc_value, traceback):
                return False

        with patch.object(
            self.database.urllib.request,
            "urlopen",
            return_value=ResponseContext(),
        ):
            adapter = self.database.OllamaEmbeddings(
                model="test-embedding",
                url="http://test-ollama/api/embed",
            )

            with self.assertRaisesRegex(
                RuntimeError, "response missing embeddings"
            ):
                adapter.embed_query("failure case")

    def test_http_failure_is_wrapped_with_batch_context(self):
        with patch.object(
            self.database.urllib.request,
            "urlopen",
            side_effect=OSError("connection refused"),
        ):
            adapter = self.database.OllamaEmbeddings(
                model="test-embedding",
                url="http://test-ollama/api/embed",
            )

            with self.assertRaisesRegex(
                RuntimeError, "batch 1-1"
            ):
                adapter.embed_query("failure case")


class TestGraphLogic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graph = importlib.import_module("chatui.utils.graph")

    def test_model_desc_self_hosted(self):
        state = {
            "generator_use_nim": True,
            "nim_generator_id": "deepseek-r1:32b",
            "generator_model_id": "unused",
        }

        result = self.graph._model_desc(state, "generator")

        self.assertEqual(
            result,
            "deepseek-r1:32b (self-hosted endpoint)",
        )

    def test_model_desc_nvidia_api(self):
        state = {
            "generator_use_nim": False,
            "nim_generator_id": "unused",
            "generator_model_id": "nvidia-model",
        }

        result = self.graph._model_desc(state, "generator")

        self.assertEqual(result, "nvidia-model (NVIDIA API endpoint)")

    def test_decide_to_generate_uses_websearch_when_flagged(self):
        state = {
            "question": "question",
            "web_search": "Yes",
            "documents": [],
        }

        self.assertEqual(
            self.graph.decide_to_generate(state),
            "websearch",
        )

    def test_decide_to_generate_generates_with_relevant_documents(self):
        state = {
            "question": "question",
            "web_search": "No",
            "documents": [SimpleNamespace(page_content="relevant")],
        }

        self.assertEqual(
            self.graph.decide_to_generate(state),
            "generate",
        )

    def test_route_question_vectorstore(self):
        class DummyChain:
            def __or__(self, _other):
                return self

            def bind(self, **_kwargs):
                return self

            def invoke(self, _payload):
                return {"datasource": "vectorstore"}

        with patch.object(self.graph, "PromptTemplate", return_value=DummyChain()), \
             patch.object(self.graph, "ChatNVIDIA", return_value=DummyChain()):
            state = {
                "question": "What is in the documents?",
                "prompt_router": "route {question}",
                "router_use_nim": False,
                "router_model_id": "test-model",
                "nim_router_ip": "unused",
                "nim_router_port": "unused",
                "nim_router_id": "unused",
                "nim_router_gpu_type": None,
                "nim_router_gpu_count": None,
            }

            self.assertEqual(
                self.graph.route_question(state),
                "vectorstore",
            )

    def test_route_question_websearch(self):
        class DummyChain:
            def __or__(self, _other):
                return self

            def bind(self, **_kwargs):
                return self

            def invoke(self, _payload):
                return {"datasource": "web_search"}

        with patch.object(self.graph, "PromptTemplate", return_value=DummyChain()), \
             patch.object(self.graph, "ChatNVIDIA", return_value=DummyChain()):
            state = {
                "question": "Latest unrelated question",
                "prompt_router": "route {question}",
                "router_use_nim": False,
                "router_model_id": "test-model",
                "nim_router_ip": "unused",
                "nim_router_port": "unused",
                "nim_router_id": "unused",
                "nim_router_gpu_type": None,
                "nim_router_gpu_count": None,
            }

            self.assertEqual(
                self.graph.route_question(state),
                "websearch",
            )


class TestCustomChatOpenAI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nim = importlib.import_module("chatui.utils.nim")

    def test_chat_result_conversion(self):
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="test response"
                    )
                )
            ]
        )

        client = self.nim.CustomChatOpenAI(
            custom_endpoint="host.docker.internal",
            port="11434",
            model_name="deepseek-r1:32b",
        )

        result = client._create_chat_result(response)

        self.assertEqual(len(result.generations), 1)
        self.assertEqual(
            result.generations[0].message.content,
            "test response",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
