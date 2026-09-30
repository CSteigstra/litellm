import copy
import unittest

from litellm.completion_extras.litellm_responses_transformation.transformation import (
    LiteLLMResponsesTransformationHandler,
)


class TestPromptCacheBreakpointPassthrough(unittest.TestCase):
    def test_user_prefix_marker_survives_conversion_without_mutation(self):
        messages = [{"role": "user", "content": [
            {"type": "text", "text": "Stable context",
             "prompt_cache_breakpoint": {"mode": "explicit"}},
            {"type": "text", "text": "Changing edit"},
        ]}]
        original = copy.deepcopy(messages)
        converted, _ = LiteLLMResponsesTransformationHandler().convert_chat_completion_messages_to_responses_api(messages)
        self.assertEqual(converted[0]["content"], [
            {"type": "input_text", "text": "Stable context",
             "prompt_cache_breakpoint": {"mode": "explicit"}},
            {"type": "input_text", "text": "Changing edit"},
        ])
        self.assertEqual(messages, original)

    def test_unmarked_and_null_marked_blocks_remain_unmarked(self):
        for block in [{"type": "text", "text": "Context"},
                      {"type": "text", "text": "Context", "prompt_cache_breakpoint": None}]:
            with self.subTest(block=block):
                converted, _ = LiteLLMResponsesTransformationHandler().convert_chat_completion_messages_to_responses_api(
                    [{"role": "user", "content": [block]}]
                )
                self.assertEqual(converted[0]["content"], [{"type": "input_text", "text": "Context"}])


if __name__ == "__main__":
    unittest.main()
