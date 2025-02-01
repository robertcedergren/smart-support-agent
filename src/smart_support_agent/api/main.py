from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ..agents.support_agent import SupportAgent, SupportTicket

app = FastAPI(title="Smart Support Agent API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize support agent
support_agent = SupportAgent()


class TicketRequest(BaseModel):
    """Incoming ticket request model."""
    content: str


@app.post("/api/process-ticket", response_model=SupportTicket)
async def process_ticket(ticket: TicketRequest):
    """Process a support ticket and return analysis."""
    try:
        result = await support_agent.process_ticket(ticket.content)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
