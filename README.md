# 🌊 CrewAI Flows - Complete Guide & Examples

<div align="center">

![CrewAI](https://img.shields.io/badge/CrewAI-Flows-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10+-green?style=for-the-badge&logo=python)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**A comprehensive collection of CrewAI Flow examples demonstrating sequential workflows, agent orchestration, and advanced state management patterns.**

[Features](#-features) • [Getting Started](#-getting-started) • [Examples](#-examples) • [Documentation](#-documentation) • [Best Practices](#-best-practices)

</div>

---

## 📖 Table of Contents

- [Overview](#-overview)
- [What are CrewAI Flows?](#-what-are-crewai-flows)
- [Features](#-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Examples](#-examples)
  - [1. Basic Flow Structure](#1-first_flowpy---basic-flow-structure)
  - [2. Agent Integration](#2-first_flow_agentpy---agent-integration-in-flows)
  - [3. Advanced State Management](#3-agent_flowpy---structured-state-management-with-agents)
  - [4. Unstructured State](#4-unstructured_state_flowpy---unstructured-state-management)
  - [5. Structured State](#5-structured_flow_statepy---structured-state-with-pydantic)
- [Core Concepts](#-core-concepts)
- [API Reference](#-api-reference)
- [Best Practices](#-best-practices)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [Resources](#-resources)

---

## 🎯 Overview

This repository serves as a **comprehensive learning resource** for CrewAI Flows, providing hands-on examples that progress from basic flow concepts to advanced agent orchestration with state management. Each example is self-contained, well-documented, and demonstrates specific aspects of the CrewAI Flow framework.

**What You'll Learn:**
- Building sequential and parallel workflow structures
- Integrating AI agents into flows
- Managing state across workflow nodes
- Implementing type-safe data flows
- Visualizing and debugging complex workflows
- Best practices for production workflows

---

## 🌊 What are CrewAI Flows?

**CrewAI Flows** is a powerful workflow orchestration framework that enables you to:

- **Chain Operations**: Connect multiple nodes in sequential or parallel patterns
- **Orchestrate Agents**: Coordinate multiple AI agents to work together on complex tasks
- **Manage State**: Maintain and pass data between workflow nodes
- **Visualize Workflows**: Generate visual representations of your flows for debugging
- **Scale Intelligence**: Build sophisticated multi-step AI workflows

### Why Use Flows?

| Traditional Approach | With CrewAI Flows |
|---------------------|-------------------|
| ❌ Manual coordination | ✅ Automatic orchestration |
| ❌ Error-prone data passing | ✅ Type-safe state management |
| ❌ Difficult to debug | ✅ Visual flow diagrams |
| ❌ Hard to scale | ✅ Modular, reusable components |
| ❌ Complex error handling | ✅ Built-in error management |

---

## ✨ Features

### 🔄 **Workflow Orchestration**
- Sequential and parallel node execution
- Event-driven architecture with `@listen` decorators
- Automatic output passing between nodes

### 🤖 **Agent Integration**
- Seamless CrewAI Agent integration
- Multi-agent collaboration patterns
- Structured agent outputs with Pydantic

### 📊 **State Management**
- **Unstructured State**: Dictionary-based for flexibility
- **Structured State**: Pydantic models for type safety
- Persistent state across the entire flow

### 🎨 **Visualization**
- HTML-based flow diagrams
- Real-time flow execution tracking
- Debug-friendly output formats

### 🔒 **Type Safety**
- Full Pydantic integration
- Type hints for better IDE support
- Runtime validation

---

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager
- OpenAI API key (or other LLM provider)

### Step 1: Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install crewai
pip install python-dotenv
pip install pydantic
```

### Step 2: Set Up Environment Variables

Create a `.env` file in the project root:

```bash
# .env file
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL_NAME=gpt-4  # Optional: specify model

# For other LLM providers
# ANTHROPIC_API_KEY=your_key
# GROQ_API_KEY=your_key
```

### Step 3: Verify Installation

```bash
python -c "import crewai; print(crewai.__version__)"
```

---

## ⚡ Quick Start

### Run Your First Flow

```bash
# Clone or download the repository
cd CrewAI-Flows

# Run the basic flow example
python first_flow.py

# Output:
# output of start node: Flow started Successfully
# output of the node 2: Data Process Complete
# flow completed
```

### View Flow Visualization

After running any example, open the generated HTML file:

```bash
# Opens in your default browser
open first_flow.html  # macOS
start first_flow.html  # Windows
xdg-open first_flow.html  # Linux
```

---

## 📁 Project Structure

```
CrewAI-Flows/
│
├── first_flow.py                  # Basic flow structure and concepts
├── first_flow_agent.py            # Agent integration basics
├── agent_flow.py                  # Advanced agent state management
├── unstructured_state_flow.py     # Dictionary-based state
├── structured_flow_state.py       # Pydantic-based state
│
├── .env                           # Environment variables (create this)
├── README.md                      # This file
│
└── outputs/                       # Generated flow visualizations
    ├── first_flow.html
    ├── agent_flow.html
    └── ...
```

---

## 📚 Examples

### 1. `first_flow.py` - Basic Flow Structure

**Purpose**: Learn the fundamental building blocks of CrewAI Flows

**What it demonstrates:**
- Creating a flow class that inherits from `Flow`
- Using `@start` decorator to mark entry points
- Chaining nodes with `@listen` decorator
- Passing output between nodes
- Generating flow visualizations

**Code Overview:**

```python
class MyFirstFlow(Flow):
    @start
    def start_flow(self) -> str:
        return "Flow started Successfully"
    
    @listen(start_flow)
    def processing_node(self, output) -> str:
        print(f"output of start node: {output}")
        return "Data Process Complete"
    
    @listen(processing_node)
    def flow_complete(self, output_2) -> str:
        print(f"output of the node 2: {output_2}")
        return "flow completed"
```

**Flow Diagram:**

```
┌─────────────┐
│ start_flow  │
└──────┬──────┘
       │
       ▼
┌──────────────────┐
│ processing_node  │
└──────┬───────────┘
       │
       ▼
┌──────────────┐
│ flow_complete│
└──────────────┘
```

**Run it:**

```bash
python first_flow.py
```

**Expected Output:**

```
output of start node: Flow started Successfully
output of the node 2: Data Process Complete
flow completed
```

**Key Learnings:**
- Flow execution is automatic once `kickoff()` is called
- Each node receives the previous node's output
- Nodes execute in the order defined by `@listen` connections
- Visualizations help debug complex flows

---

### 2. `first_flow_agent.py` - Agent Integration in Flows

**Purpose**: Integrate CrewAI Agents into your flows for AI-powered tasks

**What it demonstrates:**
- Creating agents with roles, goals, and backstories
- Using Pydantic models for structured agent outputs
- Passing data between agent-based nodes
- Environment variable management

**Code Overview:**

```python
class CityName(BaseModel):
    city_name: str = Field(description='Name of the City')
    country_name: str = Field(description='Name of the Country')

class MyFirstAgent(Flow):
    @start()
    def generate_city_name(self) -> dict[str, str]:
        city_agent = Agent(
            role="Expert in City name and Country name in the World",
            goal='Generate a random city name and country name',
            backstory='You have been working this for 3 years.'
        )
        
        prompt = 'Generate a City name and the Country name'
        agent_output = city_agent.kickoff(
            messages=prompt,
            response_format=CityName
        )
        
        return agent_output.pydantic.model_dump()
    
    @listen(generate_city_name)
    def generate_fact(self, output_node) -> str:
        city_name = output_node["city_name"]
        country_name = output_node["country_name"]
        
        fact_agent = Agent(
            role='Expert in fun facts about cities',
            goal=f'Generate a fun fact about {city_name}, {country_name}',
            backstory='You have been generating fun facts for 2 years'
        )
        
        prompt = f"Generate a fun fact about {city_name}"
        fact = fact_agent.kickoff(prompt)
        
        return fact.raw
```

**Flow Diagram:**

```
┌──────────────────────┐
│ generate_city_name   │
│   (Agent 1)          │
└──────────┬───────────┘
           │
           │ {city: "Paris", country: "France"}
           │
           ▼
┌──────────────────────┐
│   generate_fact      │
│   (Agent 2)          │
└──────────────────────┘
```

**Run it:**

```bash
python first_flow_agent.py
```

**Expected Output:**

```
City name output is: {'city_name': 'Tokyo', 'country_name': 'Japan'}
Flow Output:
Did you know that Tokyo, Japan, has more Michelin-starred restaurants 
than any other city in the world? ...
```

**Key Learnings:**
- Agents can be instantiated within flow nodes
- Use `response_format` for structured outputs
- Pydantic models ensure data consistency
- Agent outputs can be passed between nodes

**Configuration Tips:**

```python
# Customize agent behavior
agent = Agent(
    role="Your role",
    goal="Your goal",
    backstory="Your backstory",
    verbose=True,              # Enable detailed logging
    allow_delegation=False,    # Prevent delegation to other agents
    max_iter=5,               # Maximum iterations
    memory=True               # Enable memory
)
```

---

### 3. `agent_flow.py` - Structured State Management with Agents

**Purpose**: Build type-safe flows with Pydantic state management

**What it demonstrates:**
- Defining flow state with Pydantic models
- Using `Flow[StateModel]` for type safety
- Updating state across nodes
- Type hints with `Literal` for validation

**Code Overview:**

```python
class AgentState(BaseModel):
    topic: str = Field(description="Topic given as input", default="")
    description: str = Field(description="Description of topic", default="")
    topic_category: Literal["Technical", "Non-Technical"] | None = Field(
        description="Category of topic",
        default=None
    )

class AgentOutput(BaseModel):
    topic_description: str = Field(description="One paragraph description")
    category: Literal["Technical", "Non-Technical"] = Field(
        description="Category of the topic"
    )

class AgenticStateFlow(Flow[AgentState]):
    @start()
    def get_topic(self) -> str:
        topic = self.state.topic
        return topic
    
    @listen(get_topic)
    def gen_description(self, topic) -> dict[str, str]:
        agent = Agent(
            role=f"Expert in Describing {topic}",
            goal="Give description and categorize as technical or non-technical",
            backstory=f"You are an expert in {topic}"
        )
        
        prompt = f"Describe {topic} and categorize it"
        response = agent.kickoff(prompt, response_format=AgentOutput)
        
        # Update state
        self.state.description = response.pydantic.topic_description
        self.state.topic_category = response.pydantic.category
        
        return self.state.model_dump()
```

**State Flow:**

```
Input State:
{
  "topic": "AI Agents in Coding",
  "description": "",
  "topic_category": null
}
         │
         ▼
    get_topic()
         │
         ▼
  gen_description()
         │
         ▼
Final State:
{
  "topic": "AI Agents in Coding",
  "description": "AI agents in coding are...",
  "topic_category": "Technical"
}
```

**Run it:**

```bash
python agent_flow.py
```

**Expected Output:**

```
Final State: {
    'topic': 'AI Agents in Coding',
    'description': 'AI agents in coding represent intelligent systems...',
    'topic_category': 'Technical'
}
```

**Key Learnings:**
- State is accessible via `self.state` throughout the flow
- Type validation happens automatically with Pydantic
- `Literal` types restrict values to specific options
- State persists across all nodes

**Advanced Pattern:**

```python
# Complex state with nested models
class UserProfile(BaseModel):
    name: str
    email: str

class ComplexState(BaseModel):
    user: UserProfile
    data: list[dict[str, Any]]
    metadata: dict[str, str]

class ComplexFlow(Flow[ComplexState]):
    # Access nested state
    @start()
    def process(self):
        user_name = self.state.user.name
        # ...
```

---

### 4. `unstructured_state_flow.py` - Unstructured State Management

**Purpose**: Use flexible dictionary-based state for dynamic workflows

**What it demonstrates:**
- Dictionary-based state access
- Dynamic state updates
- No predefined schema
- Flexible data structures

**Code Overview:**

```python
class UnstructuredStateFlow(Flow):
    @start()
    def get_input(self) -> dict[str, Any]:
        # Access state as dictionary
        self.state['name'] = 'Atiq'
        self.state['Age'] = 25
        self.state['Friend_list'] = ['Atiq', 'Hassan', 'Fahim']
        
        return self.state
    
    @listen(get_input)
    def update_state(self, last_output) -> str:
        print('Last_output', last_output)
        return "State Updated"
```

**When to Use Unstructured State:**

✅ **Good for:**
- Prototyping and experimentation
- Dynamic data structures
- Simple workflows
- Unknown data shapes

❌ **Not ideal for:**
- Production systems requiring validation
- Complex data structures
- Type safety requirements
- Large team projects

**Run it:**

```bash
python unstructured_state_flow.py
```

**Expected Output:**

```
Last_output {'name': 'Atiq', 'Age': 25, 'Friend_list': ['Atiq', 'Hassan', 'Fahim']}
Output is: State Updated
```

**Key Learnings:**
- State is a simple Python dictionary
- No validation or type checking
- Maximum flexibility
- Useful for rapid prototyping

---

### 5. `structured_flow_state.py` - Structured State with Pydantic

**Purpose**: Implement type-safe, validated state management

**What it demonstrates:**
- Pydantic models for state definition
- Type validation and default values
- Field descriptions for documentation
- Complex data structures (lists, dicts)

**Code Overview:**

```python
class PersonState(BaseModel):
    name: str = Field(description='Name of the person', default='')
    age: int = Field(description='Age of the person', default=0)
    friend_list: list[str] = Field(
        description='Friend list of the person',
        default_factory=list
    )
    contact_info: dict[str, str | int] = Field(
        description='Contact information',
        default={}
    )

class StructuredFlowState(Flow[PersonState]):
    @start()
    def get_inputs(self):
        print(f'state is {self.state}')
        return self.state
    
    @listen(get_inputs)
    def update_state(self):
        # Type-safe updates
        self.state.friend_list.append('khan')
        self.state.contact_info['email'] = 'abc@123'
        
        return 'State updated'
```

**Benefits of Structured State:**

| Feature | Benefit |
|---------|---------|
| **Type Safety** | Catch errors at runtime |
| **Validation** | Automatic data validation |
| **Documentation** | Self-documenting with Field descriptions |
| **IDE Support** | Autocomplete and type hints |
| **Default Values** | Sensible defaults for all fields |

**Run it:**

```bash
python structured_flow_state.py
```

**Expected Output:**

```
state is name='Aamir' age=33 friend_list=['yaqoob', 'Hadi'] contact_info={'email': 'khan@122334'}
output is State updated
Flow State: name='Aamir' age=33 friend_list=['yaqoob', 'Hadi', 'khan'] contact_info={'email': 'abc@123'}
```

**Key Learnings:**
- Define state schema upfront with Pydantic
- Get validation and type checking for free
- Initialize state via `kickoff()` parameters
- Access state properties with dot notation

**Advanced Pydantic Features:**

```python
from pydantic import BaseModel, Field, validator

class AdvancedState(BaseModel):
    email: str = Field(..., regex=r'^[\w\.-]+@[\w\.-]+\.\w+$')
    age: int = Field(..., gt=0, lt=150)
    
    @validator('email')
    def validate_email(cls, v):
        if not v.endswith('@company.com'):
            raise ValueError('Must be company email')
        return v
```

---

## 🧠 Core Concepts

### 1. Flow Architecture

```python
from crewai.flow.flow import Flow, start, listen

class MyFlow(Flow):
    # Flow nodes are methods decorated with @start or @listen
    pass
```

**Key Components:**
- **Flow Class**: Container for workflow logic
- **Nodes**: Individual methods that perform tasks
- **Edges**: Connections between nodes via `@listen`
- **State**: Shared data accessible across nodes

### 2. Decorators

#### `@start`
- Marks the entry point(s) of your flow
- A flow can have multiple start nodes
- Start nodes accept no input from previous nodes

```python
@start
def initialize(self):
    return "Started"
```

#### `@listen(node_function)`
- Connects a node to listen to another node's output
- Creates directed edges in the flow graph
- Can listen to multiple nodes

```python
@listen(initialize)
def process(self, init_output):
    # process receives output from initialize
    pass

# Listen to multiple nodes
@listen(node1, node2)
def combine(self, output1, output2):
    pass
```

### 3. State Management

#### Unstructured State (Dictionary)

```python
class MyFlow(Flow):
    @start()
    def node(self):
        self.state['key'] = 'value'
        self.state['nested'] = {'data': [1, 2, 3]}
```

**Pros:**
- Flexible
- No setup required
- Quick prototyping

**Cons:**
- No type safety
- No validation
- Harder to maintain

#### Structured State (Pydantic)

```python
class MyState(BaseModel):
    name: str
    age: int

class MyFlow(Flow[MyState]):
    @start()
    def node(self):
        self.state.name = "Alice"  # Type-safe!
```

**Pros:**
- Type safety
- Automatic validation
- Better IDE support
- Self-documenting

**Cons:**
- Requires upfront schema definition
- Less flexible

### 4. Agent Integration

```python
from crewai import Agent

@listen(previous_node)
def agent_node(self, input_data):
    agent = Agent(
        role="Specialist",
        goal="Achieve specific objective",
        backstory="Context and expertise"
    )
    
    result = agent.kickoff(
        messages="Your prompt",
        response_format=OutputModel  # Optional structured output
    )
    
    return result.raw  # or result.pydantic for structured
```

### 5. Flow Execution

```python
# Create flow instance
flow = MyFlow()

# Execute without initial state
result = flow.kickoff()

# Execute with initial state (for structured state)
result = flow.kickoff({
    "field1": "value1",
    "field2": "value2"
})

# Visualize flow
flow.plot('output.html')
```

---

## 📖 API Reference

### Flow Class

```python
class Flow:
    def kickoff(self, inputs: dict = None) -> Any:
        """
        Execute the flow
        
        Args:
            inputs: Initial state values (for structured state)
            
        Returns:
            Output from the final node
        """
        
    def plot(self, filename: str) -> None:
        """
        Generate HTML visualization of the flow
        
        Args:
            filename: Output HTML file path
        """
        
    @property
    def state(self) -> dict | BaseModel:
        """
        Access flow state
        
        Returns:
            Dictionary (unstructured) or Pydantic model (structured)
        """
```

### Decorators

```python
@start
def entry_point(self):
    """Marks flow entry point"""
    
@listen(func1, func2, ...)
def dependent_node(self, output1, output2, ...):
    """Listens to one or more nodes"""
```

### Agent Class

```python
class Agent:
    def __init__(
        self,
        role: str,
        goal: str,
        backstory: str,
        verbose: bool = False,
        allow_delegation: bool = False,
        max_iter: int = 5,
        memory: bool = False
    ):
        """
        Create an agent
        
        Args:
            role: Agent's role description
            goal: What the agent should achieve
            backstory: Context about the agent
            verbose: Enable detailed logging
            allow_delegation: Allow delegating to other agents
            max_iter: Maximum iterations for task
            memory: Enable agent memory
        """
        
    def kickoff(
        self,
        messages: str | list,
        response_format: Type[BaseModel] = None
    ) -> AgentOutput:
        """
        Execute agent task
        
        Args:
            messages: Prompt or conversation history
            response_format: Pydantic model for structured output
            
        Returns:
            AgentOutput with .raw or .pydantic attributes
        """
```

---

## 💡 Best Practices

### 1. Flow Design

✅ **DO:**
```python
# Clear, descriptive node names
@start
def fetch_user_data(self):
    pass

@listen(fetch_user_data)
def validate_and_enrich_data(self, user_data):
    pass
```

❌ **DON'T:**
```python
# Vague names
@start
def node1(self):
    pass

@listen(node1)
def process(self, data):
    pass
```

### 2. State Management

✅ **DO:**
```python
# Use structured state for production
class ProductionState(BaseModel):
    user_id: str
    processed_data: list[dict]
    status: Literal["pending", "complete", "failed"]

class ProductionFlow(Flow[ProductionState]):
    pass
```

❌ **DON'T:**
```python
# Unstructured state in production
class ProductionFlow(Flow):
    def node(self):
        self.state['usr'] = 123  # Typo-prone
```

### 3. Agent Configuration

✅ **DO:**
```python
# Specific, actionable prompts
agent = Agent(
    role="Python Code Reviewer",
    goal="Review Python code for PEP 8 compliance and security issues",
    backstory="Senior developer with 10 years of Python experience"
)

prompt = """
Review this Python code:
{code}

Focus on:
1. PEP 8 compliance
2. Security vulnerabilities
3. Performance issues
"""
```

❌ **DON'T:**
```python
# Vague prompts
agent = Agent(
    role="Coder",
    goal="Code stuff",
    backstory="Does coding"
)

prompt = "Look at this code"
```

### 4. Error Handling

✅ **DO:**
```python
@listen(previous_node)
def robust_node(self, input_data):
    try:
        result = self.process_data(input_data)
        return result
    except Exception as e:
        print(f"Error in robust_node: {e}")
        return {"status": "error", "message": str(e)}
```

### 5. Structured Outputs

✅ **DO:**
```python
# Always use Pydantic for agent outputs
class AnalysisOutput(BaseModel):
    summary: str = Field(description="Brief summary")
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float = Field(ge=0, le=1)

result = agent.kickoff(prompt, response_format=AnalysisOutput)
data = result.pydantic.model_dump()
```

### 6. Visualization and Debugging

✅ **DO:**
```python
if __name__ == "__main__":
    flow = MyFlow()
    
    # Add logging
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Execute
    result = flow.kickoff()
    
    # Visualize
    flow.plot('debug_flow.html')
    
    # Print state
    print(f"Final state: {flow.state}")
```

### 7. Environment Management

✅ **DO:**
```python
from dotenv import load_dotenv
import os

load_dotenv()

# Validate environment
required_vars = ['OPENAI_API_KEY']
missing = [var for var in required_vars if not os.getenv(var)]
if missing:
    raise EnvironmentError(f"Missing variables: {missing}")
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. Import Errors

**Problem:**
```
ImportError: cannot import name 'Flow' from 'crewai.flow.flow'
```

**Solution:**
```bash
# Update CrewAI to latest version
pip install --upgrade crewai
```

#### 2. API Key Issues

**Problem:**
```
Error: OpenAI API key not found
```

**Solution:**
```bash
# Check .env file exists and is loaded
cat .env  # Should show OPENAI_API_KEY=...

# Verify in Python
from dotenv import load_dotenv
import os
load_dotenv()
print(os.getenv('OPENAI_API_KEY'))  # Should not be None
```

#### 3. Pydantic Validation Errors

**Problem:**
```
ValidationError: field required
```

**Solution:**
```python
# Always provide default values or use Optional
class MyState(BaseModel):
    required_field: str  # ❌ Will fail if not provided
    optional_field: str = ""  # ✅ Has default
    nullable_field: str | None = None  # ✅ Can be None
```

#### 4. Flow Not Executing

**Problem:**
Flow seems to hang or not execute nodes

**Solution:**
```python
# 1. Check decorator usage
@start  # ✅ Correct
def node(self):
    pass

@start()  # ✅ Also correct
def node2(self):
    pass

# 2. Verify @listen points to correct function
@listen(node)  # Function reference, not string
def next_node(self, output):
    pass

# 3. Ensure kickoff() is called
flow = MyFlow()
result = flow.kickoff()  # Don't forget this!
```

#### 5. Agent Errors

**Problem:**
```
Agent failed to generate response
```

**Solution:**
```python
# 1. Enable verbose mode
agent = Agent(
    role="...",
    goal="...",
    backstory="...",
    verbose=True  # See detailed logs
)

# 2. Check API quota and rate limits
# 3. Simplify the prompt
# 4. Try a different model
```

---

## 🎓 Advanced Patterns

### 1. Parallel Execution

```python
class ParallelFlow(Flow):
    @start
    def start(self):
        return "data"
    
    # These run in parallel
    @listen(start)
    def branch_a(self, data):
        return "result_a"
    
    @listen(start)
    def branch_b(self, data):
        return "result_b"
    
    # Waits for both
    @listen(branch_a, branch_b)
    def merge(self, result_a, result_b):
        return f"{result_a} + {result_b}"
```

### 2. Conditional Flows

```python
class ConditionalFlow(Flow[MyState]):
    @start
    def start(self):
        return self.state.input_type
    
    @listen(start)
    def route(self, input_type):
        if input_type == "text":
            return self.process_text()
        elif input_type == "image":
            return self.process_image()
        else:
            return self.process_default()
```

### 3. Loop Patterns

```python
class LoopFlow(Flow):
    @start
    def initialize(self):
        self.state['count'] = 0
        return "start"
    
    @listen(initialize)
    def loop_node(self, _):
        self.state['count'] += 1
        
        if self.state['count'] < 5:
            # Continue loop
            return self.loop_node(None)
        else:
            return "done"
```

### 4. Multi-Agent Collaboration

```python
class MultiAgentFlow(Flow):
    @start
    def coordinator(self):
        # Coordinator agent assigns tasks
        coordinator = Agent(
            role="Project Coordinator",
            goal="Analyze task and delegate to specialists"
        )
        task_plan = coordinator.kickoff("Plan the project")
        return task_plan.raw
    
    @listen(coordinator)
    def specialist_team(self, task_plan):
        # Multiple specialists work in parallel
        researcher = Agent(role="Researcher", ...)
        writer = Agent(role="Writer", ...)
        reviewer = Agent(role="Reviewer", ...)
        
        research = researcher.kickoff(f"Research: {task_plan}")
        draft = writer.kickoff(f"Write based on: {research.raw}")
        final = reviewer.kickoff(f"Review: {draft.raw}")
        
        return final.raw
```

---

## 🧪 Testing Your Flows

### Unit Testing

```python
import pytest
from my_flow import MyFlow

def test_flow_initialization():
    flow = MyFlow()
    assert flow.state is not None

def test_flow_execution():
    flow = MyFlow()
    result = flow.kickoff({"input": "test"})
    assert result is not None
    assert flow.state.processed == True

def test_node_output():
    flow = MyFlow()
    output = flow.start_node()
    assert isinstance(output, str)
```

### Integration Testing

```python
def test_complete_flow():
    flow = MyFlow()
    
    # Mock API calls if needed
    with patch('crewai.Agent.kickoff') as mock_agent:
        mock_agent.return_value = MockOutput("test result")
        
        result = flow.kickoff({
            "user_id": "123",
            "data": "test"
        })
        
        assert flow.state.status == "complete"
        assert result.contains("test result")
```

---

## 📊 Performance Tips

### 1. Optimize Agent Calls

```python
# ❌ Don't create agents repeatedly
@listen(node)
def inefficient(self, data):
    for item in data:
        agent = Agent(...)  # Creates new agent each time
        agent.kickoff(item)

# ✅ Create once, reuse
class MyFlow(Flow):
    def __init__(self):
        super().__init__()
        self.agent = Agent(...)  # Create once
    
    @listen(node)
    def efficient(self, data):
        for item in data:
            self.agent.kickoff(item)  # Reuse agent
```

### 2. Batch Processing

```python
# Process items in batches
@listen(fetch_data)
def process_in_batches(self, items):
    batch_size = 10
    results = []
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i+batch_size]
        result = self.process_batch(batch)
        results.extend(result)
    
    return results
```

### 3. Caching

```python
from functools import lru_cache

class CachedFlow(Flow):
    @lru_cache(maxsize=128)
    def expensive_computation(self, input_data):
        # This result will be cached
        return complex_calculation(input_data)
```

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### Adding New Examples

1. Create a new Python file following the naming convention
2. Include clear comments and docstrings
3. Add a section in this README
4. Test thoroughly

### Improving Documentation

- Fix typos or unclear explanations
- Add more examples or use cases
- Improve code comments
- Translate documentation

### Reporting Issues

Open an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment details (Python version, OS, etc.)

---

## 📚 Resources

### Official Documentation
- [CrewAI Documentation](https://docs.crewai.com/)
- [CrewAI Flows Guide](https://docs.crewai.com/concepts/flows)
- [Pydantic Documentation](https://docs.pydantic.dev/)

### Community
- [CrewAI GitHub](https://github.com/joaomdmoura/crewAI)
- [CrewAI Discord](https://discord.gg/crewai)
- [Stack Overflow - CrewAI](https://stackoverflow.com/questions/tagged/crewai)

### Tutorials
- [Building Your First Flow](https://docs.crewai.com/tutorials/first-flow)
- [Agent Orchestration Patterns](https://docs.crewai.com/tutorials/orchestration)
- [State Management Best Practices](https://docs.crewai.com/tutorials/state-management)

### Related Projects
- [LangChain](https://langchain.com/) - LLM application framework
- [AutoGen](https://microsoft.github.io/autogen/) - Multi-agent conversations
- [LlamaIndex](https://www.llamaindex.ai/) - Data framework for LLMs

---


## 🙏 Acknowledgments

- CrewAI team for the amazing framework
- Community contributors for examples and feedback
- OpenAI for GPT models
- Anthropic for Claude models

---

## 📞 Support

Need help? 

- 📧 Email: [atiqurrehmandatascientist@gmail.com](mailto:your-email@example.com)
- 🐛 Issues: Open a GitHub issue
- 📖 Docs: Check the official documentation

---

## 🗺️ Roadmap

### Coming Soon
- [ ] Async flow execution examples
- [ ] Database integration patterns
- [ ] API endpoint integration
- [ ] Error handling strategies
- [ ] Production deployment guide
- [ ] Docker containerization
- [ ] CI/CD pipeline examples

### Future Enhancements
- [ ] Multi-language support
- [ ] Visual flow builder
- [ ] Performance benchmarks
- [ ] More agent collaboration patterns
- [ ] Real-world use case examples

---

## ⭐ Star History

If you find this repository helpful, please consider giving it a star! ⭐

---

<div align="center">

**Made with ❤️ by the CrewAI Community**

[Report Bug](https://github.com/yourusername/crewai-flows/issues) · [Request Feature](https://github.com/yourusername/crewai-flows/issues) · [Contribute](https://github.com/yourusername/crewai-flows/pulls)

</div>