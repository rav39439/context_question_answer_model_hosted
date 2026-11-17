import unittest

from main import monitor_answer_stream


class HallucinationMonitorTests(unittest.TestCase):

    def test_terminates_before_emitting_unsupported_sentence(self):
        context = "Plants absorb sunlight and water to make glucose."
        generated_prefixes = [
            "Plants absorb sunlight and water to make glucose. ",
            (
                "Plants absorb sunlight and water to make glucose. "
                "The Eiffel Tower stands in Paris. "
            ),
            (
                "Plants absorb sunlight and water to make glucose. "
                "The Eiffel Tower stands in Paris. More generated text."
            ),
        ]
        consumed_prefixes = []

        def generated_text():
            for prefix in generated_prefixes:
                consumed_prefixes.append(prefix)
                yield prefix

        events = list(monitor_answer_stream(context, generated_text()))

        self.assertEqual(
            events,
            [
                {
                    "token":
                    "Plants absorb sunlight and water to make glucose. "
                },
                {"signal": "<terminated>"},
            ],
        )
        self.assertEqual(len(consumed_prefixes), 2)

    def test_emits_supported_final_text_without_terminal_punctuation(self):
        context = "Plants absorb sunlight and water to make glucose."
        generated_text = [
            "Plants absorb sunlight and water to make glucose"
        ]

        events = list(monitor_answer_stream(context, generated_text))

        self.assertEqual(
            events,
            [
                {
                    "token":
                    "Plants absorb sunlight and water to make glucose"
                }
            ],
        )

    def test_terminates_when_a_word_is_generated_three_times(self):
        generated_prefixes = [
            "echo",
            "echo echo",
            "echo echo echo",
            "echo echo echo more text",
        ]
        consumed_prefixes = []

        def generated_text():
            for prefix in generated_prefixes:
                consumed_prefixes.append(prefix)
                yield prefix

        events = list(monitor_answer_stream("echo", generated_text()))

        self.assertEqual(events, [{"signal": "<terminated>"}])
        self.assertEqual(len(consumed_prefixes), 3)

    def test_repeated_words_are_counted_separately_per_sentence(self):
        events = list(
            monitor_answer_stream(
                "echo",
                ["echo echo. ", "echo echo. echo echo. "],
            )
        )

        self.assertEqual(
            events,
            [
                {"token": "echo echo. "},
                {"token": "echo echo. "},
            ],
        )


if __name__ == "__main__":
    unittest.main()
