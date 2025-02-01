from typing import Dict, List, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from loguru import logger
from pydantic import BaseModel

from ..llm.model import get_llm


def truncate_text(text: str, max_chars: int = 1000) -> str:
    """Truncate text to a maximum number of characters while keeping whole words."""
    if len(text) <= max_chars:
        return text
    
    truncated = text[:max_chars]
    # Find the last space to avoid cutting words
    last_space = truncated.rfind(' ')
    if last_space > 0:
        truncated = truncated[:last_space]
    return truncated + "..."


class SupportTicket(BaseModel):
    """Support ticket data model."""
    id: str
    content: str
    category: Optional[str] = None
    priority: Optional[str] = None
    suggested_response: Optional[str] = None


class SupportAgent:
    """AI agent for handling customer support tickets."""
    
    # Define valid categories and priorities
    VALID_CATEGORIES = [
        "Technical Issue",
        "Billing Question",
        "Feature Request",
        "Account Management",
        "General Inquiry"
    ]
    
    VALID_PRIORITIES = [
        "High",   # Urgent issues affecting business operations
        "Medium", # Important issues that need attention soon
        "Low"     # General questions or minor issues
    ]
    
    def __init__(self):
        logger.info("Initializing Support Agent")
        self.llm = get_llm()
        self._setup_chains()
        logger.info("Support Agent initialized successfully")

    def _setup_chains(self):
        """Initialize the LLM chains for different tasks."""
        logger.debug("Setting up processing chains")
        
        # Categorization chain
        categorize_prompt = PromptTemplate(
            input_variables=["ticket_content"],
            template="""You are a customer support agent. Your task is to categorize the following support ticket into exactly one of these categories:
{categories}

Ticket content: {ticket_content}

Respond with ONLY the category name, nothing else. Choose the most appropriate category from the list above.""".format(
                categories="\n".join(f"- {cat}" for cat in self.VALID_CATEGORIES),
                ticket_content="{ticket_content}"
            )
        )
        self.categorize_chain = categorize_prompt | self.llm | StrOutputParser()
        logger.debug("Categorization chain initialized")

        # Priority chain
        priority_prompt = PromptTemplate(
            input_variables=["ticket_content"],
            template="""You are a customer support agent. Your task is to assign ONE priority level to the following support ticket:
{priorities}

Ticket content: {ticket_content}

Respond with ONLY the priority level (High, Medium, or Low), nothing else. Consider business impact and urgency.""".format(
                priorities="\n".join([
                    "- High: Urgent issues affecting business operations",
                    "- Medium: Important issues that need attention soon",
                    "- Low: General questions or minor issues"
                ]),
                ticket_content="{ticket_content}"
            )
        )
        self.priority_chain = priority_prompt | self.llm | StrOutputParser()
        logger.debug("Priority chain initialized")

        # Response chain
        response_prompt = PromptTemplate(
            input_variables=["ticket_content", "category", "priority"],
            template="""You are a professional customer support agent. Generate a helpful response following this EXACT structure:

CONTEXT:
- Category: {category}
- Priority: {priority}
- Issue: {ticket_content}

RESPONSE REQUIREMENTS:
1. Start with a greeting and acknowledgment of the specific issue
2. For Technical Issues: Provide 2-3 immediate troubleshooting steps
3. For Billing Questions: Reference any transaction IDs and payment details
4. For High Priority: Emphasize immediate handling and escalation if needed
5. Keep total response under 200 words
6. Use clear, professional language
7. End with a specific next step or request for information

STRUCTURE YOUR RESPONSE LIKE THIS:
1. Greeting and acknowledgment
2. Immediate solution/steps (if applicable)
3. What we'll do to help
4. What we need from them (if anything)
5. Closing with next step

Write your response following this structure, starting with 'Dear Customer' or 'Hello':"""
        )
        self.response_chain = response_prompt | self.llm | StrOutputParser()
        logger.debug("Response chain initialized")

    async def process_ticket(self, ticket_content: str) -> SupportTicket:
        """Process a support ticket and return categorization and suggested response."""
        logger.info(f"Processing new support ticket: {ticket_content[:100]}...")
        
        # Truncate ticket content if too long
        truncated_content = truncate_text(ticket_content)
        if truncated_content != ticket_content:
            logger.warning(f"Ticket content truncated from {len(ticket_content)} to {len(truncated_content)} characters")
        
        # Create ticket ID (in production, this would come from a database)
        ticket_id = "TICKET-" + str(hash(truncated_content))[:8]
        logger.debug(f"Generated ticket ID: {ticket_id}")

        # Categorize ticket
        logger.debug("Categorizing ticket")
        category = await self.categorize_chain.ainvoke({"ticket_content": truncated_content})
        category = category.strip()
        if category not in self.VALID_CATEGORIES:
            logger.warning(f"Invalid category '{category}', defaulting to 'General Inquiry'")
            category = "General Inquiry"
        logger.info(f"Ticket categorized as: {category}")

        # Determine priority
        logger.debug("Determining ticket priority")
        priority = await self.priority_chain.ainvoke({"ticket_content": truncated_content})
        priority = priority.strip()
        if priority not in self.VALID_PRIORITIES:
            logger.warning(f"Invalid priority '{priority}', defaulting to 'Medium'")
            priority = "Medium"
        logger.info(f"Ticket priority set to: {priority}")

        # Generate response
        logger.debug("Generating response")
        response = await self.response_chain.ainvoke({
            "ticket_content": truncated_content,
            "category": category,
            "priority": priority,
        })
        response = response.strip()
        logger.debug("Response generated successfully")

        ticket = SupportTicket(
            id=ticket_id,
            content=ticket_content,  # Keep original content in ticket
            category=category,
            priority=priority,
            suggested_response=response
        )
        logger.info(f"Ticket {ticket_id} processed successfully")
        return ticket
