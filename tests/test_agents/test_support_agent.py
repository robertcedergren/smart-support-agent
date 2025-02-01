import pytest
from src.smart_support_agent.agents.support_agent import SupportAgent, SupportTicket


@pytest.fixture
def support_agent():
    return SupportAgent()


@pytest.mark.asyncio
async def test_process_ticket(support_agent):
    # Test ticket content
    ticket_content = """
    Hi, I'm having trouble logging into my account. 
    I've tried resetting my password but I'm not receiving the reset email.
    This is urgent as I need to access my account for work.
    """
    
    # Process ticket
    result = await support_agent.process_ticket(ticket_content)
    
    # Verify result is a SupportTicket
    assert isinstance(result, SupportTicket)
    
    # Verify all fields are populated
    assert result.id is not None
    assert result.content == ticket_content
    assert result.category is not None
    assert result.priority is not None
    assert result.suggested_response is not None
    
    # Verify category is one of the expected values
    assert result.category.strip() in [
        "Technical Issue",
        "Billing Question",
        "Feature Request",
        "Account Management",
        "General Inquiry"
    ]
    
    # Verify priority is one of the expected values
    assert result.priority.strip() in ["High", "Medium", "Low"]
