from .bash import bash_exec
from .subagents.rough_filter_agent import rough_filter_agent
from .subagents.fine_filter_agent import fine_filter_agent
from .subagents.base_model_filter_agent import base_model_filter_agent
from .subagents.discipline_discovery_agent import discipline_discovery_agent
from .python import init_python_env

# Initialize python environment
init_python_env()

TOOLS = [
    discipline_discovery_agent,
    rough_filter_agent,
    fine_filter_agent,
    base_model_filter_agent,
    bash_exec
]
