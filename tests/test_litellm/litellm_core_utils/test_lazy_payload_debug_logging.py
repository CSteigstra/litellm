"""Payload formatting must be conditional on the DEBUG level, not request size."""

import importlib
import logging

import pytest

import litellm.utils
from litellm.completion_extras.litellm_responses_transformation import transformation

token_counter_module = importlib.import_module("litellm.litellm_core_utils.token_counter")


class ReprTrackedString(str):
    def __new__(cls, value):
        obj = super().__new__(cls, value)
        obj.repr_calls = 0
        return obj

    def __repr__(self):
        self.repr_calls += 1
        return super().__repr__()


class FormattedRecords(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


@pytest.mark.parametrize("debug_enabled", [False, True])
@pytest.mark.parametrize("operation", ["parameters", "token_count", "responses"])
def test_payload_formatting_respects_log_level(monkeypatch, operation, debug_enabled):
    logger = logging.Logger("payload-logging-test")
    logger.setLevel(logging.DEBUG if debug_enabled else logging.WARNING)
    records = FormattedRecords()
    logger.addHandler(records)
    for module in (litellm.utils, token_counter_module, transformation):
        monkeypatch.setattr(module, "verbose_logger", logger)

    payload = ReprTrackedString("synthetic payload")
    messages = [{"role": "user", "content": payload}]
    if operation == "parameters":
        result = litellm.utils.get_optional_params(
            model="gpt-4o",
            custom_llm_provider="openai",
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
        )
        assert result["stream"] is True
        assert result["stream_options"] == {"include_usage": True}
        expected_prefix = "\nLiteLLM: Params passed to completion() "
    elif operation == "token_count":
        count = token_counter_module.token_counter(model="gpt-4o", messages=messages)
        assert count == token_counter_module.token_counter(
            model="gpt-4o", messages=[{"role": "user", "content": str(payload)}]
        )
        assert count > 0
        expected_prefix = "messages in token_counter: "
    else:
        content = [{
            "type": "file",
            "file": {"file_data": payload, "filename": "synthetic.pdf"},
        }]
        handler = transformation.LiteLLMResponsesTransformationHandler()
        result = handler._convert_content_to_responses_format(content, "user")
        assert result == [{
            "type": "input_file", "file_data": payload, "filename": "synthetic.pdf",
        }]
        assert result[0]["file_data"] is payload
        expected_prefix = "Chat provider: Final converted content: "

    # Check before formatting expected strings, which intentionally calls repr.
    assert (payload.repr_calls > 0) is debug_enabled
    if not debug_enabled:
        assert records.messages == []
        return
    assert any(message.startswith(expected_prefix) for message in records.messages)
    if operation == "token_count":
        assert (
            f"messages in token_counter: {messages}, text in token_counter: None"
            in records.messages
        )
    elif operation == "responses":
        assert (
            f"Chat provider: Processing content item 0: {type(content[0])} = {content[0]}"
            in records.messages
        )
        assert f"Chat provider: Final converted content: {result}" in records.messages
