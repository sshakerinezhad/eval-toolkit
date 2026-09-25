import pytest

import grade


@pytest.mark.parametrize("text,expected", [
    ('{"rationale": "r", "is_criteria_true": true}', (True, "r")),
    ('```json\n{"rationale": "r", "is_criteria_true": false}\n```', (False, "r")),
    ("the criterion is met", (None, "the criterion is met")),
    ('{"rationale": "r", "is_criteria_true": "yes"}', (None, '{"rationale": "r", "is_criteria_true": "yes"}')),
    (None, (None, "(no reply)")),
    ('{"rationale": "line1\nline2", "is_criteria_true": true}', (True, "line1\nline2")),
])
def test_parse_verdict_plain_fenced_bad(text, expected):
    assert grade.parse_verdict(text) == expected
