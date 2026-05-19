# tests/test_llm.py
from unittest.mock import MagicMock
from src.llm import call_anthropic, call_openai

def test_call_anthropic_passes_system_and_user(mocker):
    mock_client = MagicMock()
    mock_client.messages.create.return_value = MagicMock(
        content=[MagicMock(text="hello")],
        usage=MagicMock(input_tokens=10, output_tokens=5),
    )
    result = call_anthropic(
        client=mock_client,
        model="claude-haiku-4-5-20251001",
        system="be brief",
        user="say hi",
        temperature=0.2,
        max_tokens=100,
    )
    assert result["text"] == "hello"
    assert result["input_tokens"] == 10
    assert result["output_tokens"] == 5
    mock_client.messages.create.assert_called_once()
    kwargs = mock_client.messages.create.call_args.kwargs
    assert kwargs["system"] == "be brief"
    assert kwargs["messages"] == [{"role": "user", "content": "say hi"}]
    assert kwargs["temperature"] == 0.2
