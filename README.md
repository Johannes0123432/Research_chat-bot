# Research Simulation Chatbot

Standalone Streamlit application for exploratory scientific research support.

**Pipeline**
1. Generate novel, testable hypotheses in a chosen field
2. Design computational simulation plans
3. Write, run, and iteratively refine Python simulations
4. Interpret results
5. Draft structured research-style articles grounded in the simulation outputs

> **Important**: This tool produces **exploratory, simulation-based** work. It does not generate empirical scientific discoveries. Always treat outputs as hypotheses + computational experiments that require human expertise, validation, and proper scientific process.

## Features

- Multi-provider LLM support:
  - Grok (xAI)
  - Google Gemini
  - OpenRouter
  - Local Ollama
- Real local Python execution environment with scientific libraries
- Semi-autonomous workflow (you approve key steps)
- Simulation history
- One-click research-style article drafting
- Download drafts as Markdown

## Quick Start

```bash
# 1. Clone / enter the project
cd research_chatbot

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

## Using Local Ollama

1. Install Ollama from https://ollama.com
2. Pull a model, e.g.:
   ```bash
   ollama pull llama3.1
   # or
   ollama pull gemma2
   ollama pull qwen2.5
   ```
3. In the sidebar select provider = `ollama` and set the model name.
4. Leave the API key empty.

## Recommended Models

| Provider    | Good starting models                  |
|-------------|---------------------------------------|
| Grok        | grok-2-latest / grok-beta             |
| Gemini      | gemini-1.5-pro / gemini-2.0-flash     |
| OpenRouter  | anthropic/claude-3.5-sonnet, etc.     |
| Ollama      | llama3.1, gemma2, qwen2.5, mistral    |

## Project Structure

```
research_chatbot/
├── app.py                  # Main Streamlit application
├── requirements.txt
├── README.md
├── .streamlit/
│   └── config.toml
└── utils/
    ├── __init__.py
    ├── llm_clients.py      # Multi-provider LLM interface
    └── code_executor.py    # Local Python execution
```

## Safety Notes

- The code executor runs Python in the same process with common scientific libraries pre-loaded.
- It is suitable for trusted local use and research exploration.
- Do **not** expose this app publicly without adding proper sandboxing (Docker, gVisor, E2B, etc.).
- Never execute untrusted code.

## Limitations

- Simulation results depend entirely on the quality of the model, assumptions, and parameters.
- Novelty is combinatorial / exploratory, not guaranteed frontier scientific discovery.
- Always perform proper literature review and expert validation before treating any output as research.

---

Built as a standalone research assistant.
