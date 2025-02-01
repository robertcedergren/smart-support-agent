# Smart Support Agent

An AI-powered customer support system that uses open-source LLMs to analyze, route, and respond to customer inquiries intelligently.

## Why Agentic AI?

The Smart Support Agent demonstrates an agentic AI approach to customer support by operating autonomously and making intelligent decisions through multiple processing stages:

1. **Autonomous Decision Making**: Rather than simply generating responses, the agent:
   - Analyzes ticket content to understand the core issue
   - Makes independent decisions about categorization
   - Determines priority levels based on business impact
   - Generates contextually appropriate responses

2. **Multi-Step Reasoning**: The agent breaks down the support process into discrete steps:
   - First understands the ticket content
   - Then categorizes based on predefined business rules
   - Assesses urgency and priority
   - Finally generates a response considering all previous analysis

3. **Contextual Awareness**: Each step informs the next:
   - Category influences response tone and content
   - Priority level affects response urgency
   - Previous decisions create a coherent support experience

## Business Use Case

### Problem Statement
Customer support teams face several challenges:
- High volume of incoming tickets
- Need for 24/7 response capability
- Inconsistent ticket categorization
- Varying response quality
- Delayed response times
- Manual ticket routing

### Solution
The Smart Support Agent addresses these challenges by:
- Providing instant initial responses
- Ensuring consistent ticket categorization
- Automatically prioritizing urgent issues
- Reducing manual ticket routing
- Maintaining response quality standards
- Operating continuously without fatigue

### Example Use Case

Consider this customer inquiry:

```
Subject: Unable to access premium features after payment

Message: Hi, I just upgraded to the premium plan and completed the payment about an hour ago. 
However, I still can't access any of the premium features. I have an important presentation 
tomorrow and really need these features. My transaction ID is TRX-123456. Please help!
```

The agent processes this ticket as follows:

**Agent Analysis:**
```json
{
  "category": "Technical Issue",
  "priority": "High",
  "reasoning": {
    "category": "Involves access issues with paid features",
    "priority": "Customer has immediate business need and payment is involved"
  }
}
```

**Generated Response:**
```
Dear Customer,

I understand you're unable to access premium features after your recent payment, and this is urgent given your presentation tomorrow. I'm treating this with high priority.

I can see your transaction (TRX-123456) and will help resolve this immediately. While I investigate, please try these quick steps:
1. Log out and log back into your account
2. Clear your browser cache
3. Verify the payment status in your account settings

If these steps don't resolve the issue, our technical team will be notified immediately to expedite the resolution.

For fastest assistance, please provide:
- Your account email
- Browser and version you're using
- Any error messages you're seeing

We'll ensure you have access to the premium features for your presentation.

Best regards,
Support Team
```

### Business Impact

- **Efficiency**: Reduction in initial response time
- **Consistency**: Higher accuracy in ticket categorization
- **Scalability**: Handle more tickets without additional staffing
- **Customer Satisfaction**: Immediate acknowledgment and structured responses
- **Cost Reduction**: Decrease in ticket handling time
- **24/7 Coverage**: Continuous support availability

## Overview

The Smart Support Agent is designed to streamline and enhance customer support operations by leveraging state-of-the-art language models to automatically process support tickets. The system performs three key functions:

1. **Ticket Categorization**: Automatically classifies tickets into predefined categories:
   - Technical Issue
   - Billing Question
   - Feature Request
   - Account Management
   - General Inquiry

2. **Priority Assignment**: Determines ticket urgency using intelligent analysis:
   - High: Critical issues affecting business operations
   - Medium: Important issues needing timely attention
   - Low: General questions and minor issues

3. **Response Generation**: Creates contextually appropriate initial responses based on the ticket's category and content.

## System Architecture

### Components

1. **LLM Integration** (`src/smart_support_agent/llm/`):
   - Utilizes Facebook's OPT-125M model for efficient development and testing
   - Supports multiple compute devices (CPU, CUDA, MPS for M1/M2 Macs)
   - Implements proper text generation with nucleus sampling

2. **Support Agent** (`src/smart_support_agent/agents/`):
   - Implements the core ticket processing logic
   - Uses LangChain for efficient prompt management and chain composition
   - Provides async processing capabilities

3. **API Layer** (`src/smart_support_agent/api/`):
   - FastAPI-based REST endpoints
   - Handles ticket submission and processing
   - Provides health check endpoint

4. **User Interface** (`src/smart_support_agent/ui/`):
   - Streamlit-based web interface
   - Real-time ticket processing
   - Displays categorization, priority, and suggested responses

### Processing Flow

1. User submits a support ticket through the UI or API
2. The system generates a unique ticket ID
3. The Support Agent processes the ticket:
   - Analyzes content for categorization
   - Determines priority level
   - Generates an appropriate response
4. Results are returned to the user
5. All actions are logged for monitoring and debugging

## Features

- 🤖 Open-source LLM integration
- 🔄 Intelligent ticket routing
- 💬 Automated initial response generation
- 🌐 Modern web interface
- 📊 Support ticket analytics
- 📝 Comprehensive logging
- ⚡ Async processing support
- 🎯 Device-optimized inference

## Requirements

- Python 3.11+
- UV package manager
- Git

## Installation

1. Clone the repository and navigate to the project directory:
```bash
git clone https://github.com/yourusername/smart-support-agent.git
cd smart-support-agent  # Make sure you are in this directory for all following commands
```

2. Create and activate a virtual environment:
```bash
# Using venv
python3.11 -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
```

3. Install UV and project dependencies:
```bash
# Install UV (recommended for local development)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Option 1: Install only main dependencies
uv pip install -e .

# Option 2: Install main + development dependencies
uv pip install -e ".[dev]"
```

4. Install pre-commit hooks (only if you installed dev dependencies):
```bash
pre-commit install
```

## Usage

1. Start the API server:
```bash
uvicorn src.smart_support_agent.api.main:app --reload
```

2. Launch the UI:
```bash
# Make sure you're in the smart-support-agent directory
streamlit run ./src/smart_support_agent/ui/app.py
```

3. Visit `http://localhost:8501` in your browser to access the UI.

## Development

- Run tests: `pytest`
- Format code: `black .`
- Check linting: `ruff check .`
- Sort imports: `isort .`

## TODO

### Model Experiments
- [ ] Test Llama 3.1 when available
- [ ] Compare performance with current model
- [ ] Optimize token length settings

### Prompt Engineering
- [ ] Test system prompts for better role adherence
- [ ] Add few-shot examples to improve accuracy
- [ ] Experiment with temperature and sampling settings

## Customization

### Model Selection
You can customize the LLM by setting the `LLM_MODEL` environment variable or modifying the `DEFAULT_MODEL` in `src/smart_support_agent/llm/model.py`. For production use, consider using more powerful models like:
- GPT-J-6B
- BLOOM
- Llama 2

### Categories and Priorities
Modify the prompts in `src/smart_support_agent/agents/support_agent.py` to adjust:
- Ticket categories
- Priority levels
- Response templates
