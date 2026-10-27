from chess_video.providers import _llm_payload, _script_schema, _validation_feedback
from chess_video.settings import Settings


def test_schema_requires_exact_position_count():
    schema = _script_schema(position_count=34)
    assert schema["properties"]["beats"]["minItems"] == 34
    assert schema["properties"]["beats"]["maxItems"] == 34


def test_gemini_thinking_is_bounded_and_leaves_room_for_structured_output():
    settings = Settings(_env_file=None, llm_provider="vertex", llm_model="gemini-3.1-pro-preview")
    payload = _llm_payload([], settings, position_count=34)
    assert payload["generationConfig"]["thinkingConfig"]["thinkingLevel"] == "high"
    assert "minItems" not in payload["generationConfig"]["responseJsonSchema"]["properties"]["beats"]


def test_core_validation_feedback_explains_ply_coverage_instead_of_generic_repair():
    feedback = _validation_feedback(ValueError("Script needs exactly one ordered beat per ply, including initial zero"))
    assert "ordered beat per ply" in feedback
