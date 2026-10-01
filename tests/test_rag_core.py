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




class TestCustomChatOpenAIRequest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nim = importlib.import_module("chatui.utils.nim")
        import openai
        cls.openai = openai

    def test_call_custom_endpoint_builds_expected_request(self):
        from langchain_core.messages import HumanMessage

        captured = {}

        def fake_create(**config):
            captured.update(config)
            return SimpleNamespace()

        original_api_key = getattr(self.openai, "api_key", None)
        original_base_url = getattr(self.openai, "base_url", None)

        try:
            with patch.object(
                self.openai.chat.completions,
                "create",
                side_effect=fake_create,
            ):
                self.openai_adapter = self.nim.CustomChatOpenAI(
                    custom_endpoint="host.docker.internal",
                    port="11434",
                    model_name="deepseek-r1:32b",
                    temperature=0.25,
                )

                self.openai_adapter._call_custom_endpoint(
                    [HumanMessage(content="hello world")]
                )

            self.assertEqual(
                self.openai.base_url,
                "http://host.docker.internal:11434/v1/",
            )
            self.assertEqual(captured["model"], "deepseek-r1:32b")
            self.assertEqual(captured["temperature"], 0.25)
            self.assertEqual(
                captured["messages"],
                [{"role": "user", "content": "hello world"}],
            )
            self.assertNotIn("gpu_config", captured)
        finally:
            self.openai.api_key = original_api_key
            self.openai.base_url = original_base_url

    def test_call_custom_endpoint_includes_gpu_configuration(self):
        from langchain_core.messages import HumanMessage

        captured = {}

        def fake_create(**config):
            captured.update(config)
            return SimpleNamespace()

        original_api_key = getattr(self.openai, "api_key", None)
        original_base_url = getattr(self.openai, "base_url", None)

        try:
            with patch.object(
                self.openai.chat.completions,
                "create",
                side_effect=fake_create,
            ):
                adapter = self.nim.CustomChatOpenAI(
                    custom_endpoint="gpu-host",
                    port="9000",
                    model_name="test-model",
                    gpu_type="RTX 5070 Ti",
                    gpu_count="1",
                )

                with patch.object(
                    self.nim.gpu_compatibility,
                    "get_compatible_models",
                    return_value={
                        "warning_message": "",
                        "compatible_models": ["test-model"],
                    },
                ):
                    adapter._call_custom_endpoint(
                        [HumanMessage(content="gpu test")]
                    )

            self.assertEqual(
                captured["gpu_config"],
                {"type": "RTX 5070 Ti", "count": "1"},
            )
        finally:
            self.openai.api_key = original_api_key
            self.openai.base_url = original_base_url

    def test_custom_endpoint_error_propagates_without_gpu_configuration(self):
        from langchain_core.messages import HumanMessage

        expected = RuntimeError("endpoint unavailable")

        original_api_key = getattr(self.openai, "api_key", None)
        original_base_url = getattr(self.openai, "base_url", None)

        try:
            with patch.object(
                self.openai.chat.completions,
                "create",
                side_effect=expected,
            ):
                adapter = self.nim.CustomChatOpenAI(
                    custom_endpoint="host",
                    port="11434",
                    model_name="test-model",
                )

                with self.assertRaises(RTError if False else RuntimeError) as ctx:
                    adapter._call_custom_endpoint(
                        [HumanMessage(content="failure")]
                    )

                self.assertIs(ctx.exception, expected)
        finally:
            self.openai.api_key = original_api_key
            self.openai.base_url = original_base_url

    def test_custom_endpoint_error_with_gpu_configuration_is_contextualized(self):
        from langchain_core.messages import HumanMessage

        original_api_key = getattr(self.openai, "api_key", None)
        original_base_url = getattr(self.openai, "base_url", None)

        try:
            with patch.object(
                self.openai.chat.completions,
                "create",
                side_effect=RuntimeError("endpoint unavailable"),
            ):
                adapter = self.nim.CustomChatOpenAI(
                    custom_endpoint="host",
                    port="11434",
                    model_name="test-model",
                    gpu_type="RTX 5070 Ti",
                    gpu_count="1",
                )

                with patch.object(
                    self.nim.gpu_compatibility,
                    "get_compatible_models",
                    return_value={
                        "warning_message": "",
                        "compatible_models": ["test-model"],
                    },
                ):
                    with self.assertRaisesRegex(
                        ValueError,
                        r"Error with GPU configuration \(RTX 5070 Ti, 1 GPUs\)",
                    ):
                        adapter._call_custom_endpoint(
                            [HumanMessage(content="failure")]
                        )
        finally:
            self.openai.api_key = original_api_key
            self.openai.base_url = original_base_url

class TestLocalLLMConfiguration(unittest.TestCase):
    def test_local_llm_defaults(self):
        import os
        import importlib

        keys = [
            "LOCAL_LLM_ENABLED",
            "LOCAL_LLM_HOST",
            "LOCAL_LLM_PORT",
            "LOCAL_LLM_MODEL",
        ]
        saved = {key: os.environ.get(key) for key in keys}

        try:
            for key in keys:
                os.environ.pop(key, None)

            sys.modules.pop("chatui.pages.converse", None)
            converse = importlib.import_module("chatui.pages.converse")

            self.assertTrue(converse.LOCAL_LLM_ENABLED)
            self.assertEqual(
                converse.LOCAL_LLM_HOST,
                "host.docker.internal",
            )
            self.assertEqual(converse.LOCAL_LLM_PORT, "11434")
            self.assertEqual(
                converse.LOCAL_LLM_MODEL,
                "deepseek-r1:32b",
            )
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            sys.modules.pop("chatui.pages.converse", None)

    def test_local_llm_environment_overrides(self):
        import os
        import importlib

        overrides = {
            "LOCAL_LLM_ENABLED": "false",
            "LOCAL_LLM_HOST": "127.0.0.1",
            "LOCAL_LLM_PORT": "12345",
            "LOCAL_LLM_MODEL": "test-model",
        }
        keys = list(overrides)
        saved = {key: os.environ.get(key) for key in keys}

        try:
            os.environ.update(overrides)

            sys.modules.pop("chatui.pages.converse", None)
            converse = importlib.import_module("chatui.pages.converse")

            self.assertFalse(converse.LOCAL_LLM_ENABLED)
            self.assertEqual(converse.LOCAL_LLM_HOST, "127.0.0.1")
            self.assertEqual(converse.LOCAL_LLM_PORT, "12345")
            self.assertEqual(converse.LOCAL_LLM_MODEL, "test-model")
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            sys.modules.pop("chatui.pages.converse", None)

    def test_local_llm_enabled_accepts_supported_true_values(self):
        import os
        import importlib

        original = os.environ.get("LOCAL_LLM_ENABLED")

        try:
            for value in ("1", "true", "TRUE", "yes", "on"):
                os.environ["LOCAL_LLM_ENABLED"] = value
                sys.modules.pop("chatui.pages.converse", None)
                converse = importlib.import_module("chatui.pages.converse")
                self.assertTrue(
                    converse.LOCAL_LLM_ENABLED,
                    msg=f"Expected '{value}' to enable local LLM",
                )
        finally:
            if original is None:
                os.environ.pop("LOCAL_LLM_ENABLED", None)
            else:
                os.environ["LOCAL_LLM_ENABLED"] = original
            sys.modules.pop("chatui.pages.converse", None)


class TestLLMOutputContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graph = importlib.import_module("chatui.utils.graph")

    def test_validate_datasource_accepts_vectorstore(self):
        self.assertEqual(
            self.graph._validate_datasource({"datasource": "vectorstore"}),
            "vectorstore",
        )

    def test_validate_datasource_accepts_web_search(self):
        self.assertEqual(
            self.graph._validate_datasource({"datasource": "web_search"}),
            "web_search",
        )

    def test_validate_datasource_rejects_missing_field(self):
        with self.assertRaisesRegex(ValueError, "invalid datasource"):
            self.graph._validate_datasource({})

    def test_validate_datasource_rejects_unknown_value(self):
        with self.assertRaisesRegex(ValueError, "invalid datasource"):
            self.graph._validate_datasource({"datasource": "database"})

    def test_validate_binary_score_normalizes_valid_values(self):
        self.assertEqual(
            self.graph._validate_binary_score(
                {"score": "YES"}, "Test Grader"
            ),
            "yes",
        )
        self.assertEqual(
            self.graph._validate_binary_score(
                {"score": "no"}, "Test Grader"
            ),
            "no",
        )

    def test_validate_binary_score_rejects_missing_field(self):
        with self.assertRaisesRegex(ValueError, "invalid score"):
            self.graph._validate_binary_score({}, "Test Grader")

    def test_validate_binary_score_rejects_invalid_value(self):
        with self.assertRaisesRegex(ValueError, "invalid score"):
            self.graph._validate_binary_score(
                {"score": "maybe"}, "Test Grader"
            )

    def test_validate_binary_score_rejects_non_object(self):
        with self.assertRaisesRegex(ValueError, "invalid JSON object"):
            self.graph._validate_binary_score(
                ["yes"], "Test Grader"
            )

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



class TestGenerationGrading(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graph = importlib.import_module("chatui.utils.graph")

    def _state(self):
        return {
            "question": "test question",
            "documents": [SimpleNamespace(page_content="test context")],
            "generation": "test answer",
            "prompt_hallucination": "check {generation} {documents}",
            "prompt_answer": "check {generation} {question}",
            "hallucination_use_nim": False,
            "hallucination_model_id": "test-model",
            "nim_hallucination_ip": "unused",
            "nim_hallucination_port": "unused",
            "nim_hallucination_id": "unused",
            "nim_hallucination_gpu_type": None,
            "nim_hallucination_gpu_count": None,
            "answer_use_nim": False,
            "answer_model_id": "test-model",
            "nim_answer_ip": "unused",
            "nim_answer_port": "unused",
            "nim_answer_id": "unused",
            "nim_answer_gpu_type": None,
            "nim_answer_gpu_count": None,
        }

    def _run_with_scores(self, scores):
        class DummyChain:
            def __init__(self):
                self.index = 0

            def __or__(self, _other):
                return self

            def bind(self, **_kwargs):
                return self

            def invoke(self, _payload):
                result = scores[self.index]
                self.index += 1
                return result

        chain = DummyChain()

        with patch.object(
            self.graph, "PromptTemplate", return_value=chain
        ), patch.object(
            self.graph, "ChatNVIDIA", return_value=chain
        ):
            return self.graph.grade_generation_v_documents_and_question(
                self._state()
            )

    def test_grounded_and_useful_returns_useful(self):
        self.assertEqual(
            self._run_with_scores(
                [{"score": "yes"}, {"score": "yes"}]
            ),
            "useful",
        )

    def test_grounded_but_not_useful_returns_not_useful(self):
        self.assertEqual(
            self._run_with_scores(
                [{"score": "yes"}, {"score": "no"}]
            ),
            "not useful",
        )

    def test_ungrounded_returns_not_supported(self):
        self.assertEqual(
            self._run_with_scores(
                [{"score": "no"}]
            ),
            "not supported",
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
