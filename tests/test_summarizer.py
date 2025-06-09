import pytest
from unittest.mock import patch, MagicMock
from src.summarizer import summarize_text
from langchain.llms import LlamaCpp

@patch("src.summarizer.LLMChain")
def test_summarize_text(mock_llm_chain):
    """
    Test the summarize_text function with a mocked LLMChain.
    """
    # Arrange
    mock_llm = MagicMock(spec=LlamaCpp)
    text_to_summarize = "This is a long text that needs to be summarized."
    prompt_template = "Summarize: {text}"
    expected_summary = "This is a summary."
    
    # Configure the mock chain to return the expected summary
    mock_chain_instance = MagicMock()
    mock_chain_instance.run.return_value = expected_summary
    mock_llm_chain.return_value = mock_chain_instance

    # Act
    summary = summarize_text(llm=mock_llm, text=text_to_summarize, prompt_template=prompt_template)

    # Assert
    assert summary == expected_summary
    mock_llm_chain.assert_called_once()
    # Check that the prompt was created correctly
    assert mock_llm_chain.call_args.kwargs['prompt'].template == prompt_template
    # Check that the run method was called with the correct text
    mock_chain_instance.run.assert_called_once_with(text_to_summarize)




