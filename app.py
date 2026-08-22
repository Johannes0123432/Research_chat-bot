"""
Research Simulation Chatbot
Standalone Streamlit app for:
1. Generating novel hypotheses in a scientific field
2. Designing and running iterative simulations
3. Producing grounded research-style article drafts

Optimal configuration:
- Streamlit UI
- Multi-provider LLMs (Grok, Gemini, OpenRouter, Ollama)
- Local Python execution environment
- Semi-autonomous workflow
"""

import streamlit as st
import json
from typing import List, Dict, Any, Optional
from datetime import datetime

from utils.llm_clients import chat_with_provider, DEFAULT_MODELS
from utils.code_executor import execute_code, format_execution_result


# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Research Simulation Chatbot",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# Session state initialization
# --------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "hypothesis" not in st.session_state:
    st.session_state.hypothesis = None
if "simulation_plan" not in st.session_state:
    st.session_state.simulation_plan = None
if "simulation_history" not in st.session_state:
    st.session_state.simulation_history = []
if "article_draft" not in st.session_state:
    st.session_state.article_draft = None
if "stage" not in st.session_state:
    st.session_state.stage = "chat"  # chat | hypothesis | plan | simulate | article


# --------------------------------------------------
# Sidebar – Configuration
# --------------------------------------------------
with st.sidebar:
    st.title("🔬 Research Chatbot")
    st.markdown("Standalone hypothesis → simulation → article system")

    st.subheader("LLM Provider")
    provider = st.selectbox(
        "Provider",
        options=["grok", "gemini", "openrouter", "ollama"],
        index=0,
        help="Choose which backend to use for reasoning and code generation",
    )

    default_model = DEFAULT_MODELS.get(provider, "llama3.1")
    model = st.text_input("Model name", value=default_model)

    api_key = st.text_input(
        "API Key (leave empty for Ollama)",
        type="password",
        help="Required for Grok, Gemini, and OpenRouter. Not needed for local Ollama.",
    )

    ollama_base = st.text_input(
        "Ollama Base URL",
        value="http://localhost:11434",
        help="Only used when provider = ollama",
    )

    st.divider()
    st.subheader("Workflow Controls")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.35, 0.05)

    st.markdown("---")
    st.caption("This tool generates **exploratory simulation-based research**. Results are not empirical discoveries and should be treated as hypotheses + computational experiments only.")


# --------------------------------------------------
# Helper: call the LLM
# --------------------------------------------------
def ask_llm(system_prompt: str, user_prompt: str, history: Optional[List[Dict]] = None) -> str:
    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_prompt})

    try:
        reply = chat_with_provider(
            provider=provider,
            model=model,
            messages=messages,
            api_key=api_key if api_key else None,
            temperature=temperature,
            max_tokens=4096,
            base_url=ollama_base if provider == "ollama" else None,
        )
        return reply
    except Exception as e:
        return f"[Error calling LLM: {type(e).__name__}: {e}]"


# --------------------------------------------------
# System prompts
# --------------------------------------------------
HYPOTHESIS_SYSTEM = """You are a careful scientific research assistant focused on generating testable computational hypotheses.

Strict rules:
- Propose hypotheses that are specific, falsifiable, and suitable for simulation.
- You may be creative, but you must clearly label the work as exploratory.
- Never claim the hypothesis is novel in an absolute sense. Use cautious language such as "potentially underexplored", "worth investigating computationally", or "a possible direction".
- Never invent papers, authors, or prior results.
- Clearly list assumptions.
- Do not overstate significance.
Respond in clear Markdown."""

SIMULATION_PLAN_SYSTEM = """You are an expert in designing careful computational experiments.
Given a hypothesis, produce a concrete simulation plan implementable in Python (NumPy, SciPy, Pandas, Matplotlib, NetworkX, SymPy, statsmodels, sklearn).

Rules:
- Focus on what can actually be computed.
- Include parameters, metrics, and what the simulation can and cannot tell us.
- Prefer designs that produce clear quantitative outputs.
- Do not claim the simulation will prove the hypothesis.
Output a structured plan in Markdown."""

CODE_SYSTEM = """You are a careful scientific Python programmer.
Turn the hypothesis and plan into complete, runnable simulation code.

Hard rules:
- Use only: numpy, scipy, pandas, matplotlib, seaborn, sympy, networkx, statsmodels, sklearn.
- The code must assign main quantitative findings to a variable named `result` (dict or DataFrame preferred).
- Print clear summary statistics that come directly from the computation.
- Do not hard-code fake results.
- Do not use input(), file writes, network calls, or dangerous operations.
- Return ONLY the Python code (use one ```python block)."""

INTERPRET_SYSTEM = """You are a strictly conservative scientific interpreter of computational results.

Hard rules you must follow:
- Only describe what the executed code actually produced.
- Separate clearly:
  1. Computed facts (numbers, statistics, patterns present in the output)
  2. Possible interpretations (tentative only)
- Use very cautious language: "the simulation shows", "under these assumptions", "consistent with", "does not demonstrate".
- Never claim the results prove a real-world phenomenon.
- Never invent literature or external findings.
- Explicitly state limitations.
- If the output is weak, noisy, or inconclusive, say so directly.
Structure your response with clear headings."""

ARTICLE_SYSTEM = """You are a conservative scientific writer producing an exploratory computational report.

Hard constraints:
- Base every quantitative claim strictly on the provided simulation outputs.
- Do NOT invent papers, citations, or prior empirical findings.
- Do NOT claim real-world discovery or strong novelty.
- Use cautious language throughout.
- Include a prominent Limitations section.
- Clearly label the work as simulation-based exploratory analysis.
- Prefer phrases such as "the simulations indicate", "under the modeled assumptions", "these computational experiments suggest".
- If evidence is weak, say so.
Structure the report with standard sections but keep claims tightly grounded in the executed results."""


# --------------------------------------------------
# Main UI
# --------------------------------------------------
st.title("🔬 Research Simulation Chatbot")
st.markdown(
    "Generate hypotheses → design & run simulations → produce tightly grounded exploratory reports."
)
st.warning(
    "**Important limitations**: This tool produces **simulation-based exploratory analysis only**. "
    "It does not generate real scientific discoveries. Quantitative numbers come from actual code execution, "
    "but interpretations and any broader claims remain provisional and must be treated with caution. "
    "Never publish or rely on outputs without expert human review."
)

# Stage indicator
cols = st.columns(5)
stages = ["Chat", "Hypothesis", "Plan", "Simulate", "Article"]
stage_map = {"chat": 0, "hypothesis": 1, "plan": 2, "simulate": 3, "article": 4}
for i, name in enumerate(stages):
    with cols[i]:
        if i == stage_map.get(st.session_state.stage, 0):
            st.markdown(f"**→ {name}**")
        else:
            st.markdown(name)

st.divider()

# --------------------------------------------------
# Chat / Hypothesis Generation Section
# --------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat & Hypothesis", "🧪 Simulation Lab", "📄 Article Draft", "📊 History"])

with tab1:
    st.subheader("1. Request a Novel Hypothesis")

    field = st.text_input(
        "Scientific / technical field",
        placeholder="e.g. computational social science, epidemiology, ecology, agent-based economics, complex systems...",
        value="computational social science",
    )

    user_request = st.text_area(
        "Describe what kind of novel hypothesis you want",
        height=100,
        placeholder="Example: Propose a novel hypothesis about how recommendation algorithms affect political polarization under different network topologies...",
    )

    st.markdown("#### Optional: Document / Model Description Mode")
    st.caption("Paste text from a paper, model description, or trusted source. The system will help turn it into a simulation using the built-in libraries only.")
    document_text = st.text_area(
        "Paste paper excerpt, model equations, or description (optional)",
        height=140,
        placeholder="Paste relevant sections from a paper or a model description you trust. The system will try to turn this into runnable simulation code using NumPy/SciPy/etc.",
        key="document_text",
    )

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.button("Generate Novel Hypothesis", type="primary", use_container_width=True):
            if not user_request.strip() and not document_text.strip():
                st.warning("Please describe the kind of hypothesis you want, or paste a document/model description.")
            else:
                with st.spinner("Generating hypothesis..."):
                    doc_section = ""
                    if document_text.strip():
                        doc_section = f"\n\nAdditional source material provided by the user (use only as context, do not invent extra literature):\n{document_text[:6000]}\n"

                    prompt = f"""Field: {field}

User request: {user_request}
{doc_section}

Generate 1 clear, testable hypothesis suitable for computational simulation.
If source material was provided, ground the hypothesis in that material where possible.
Use cautious language. Do not invent papers or external findings.

Structure your response with:
- Hypothesis statement (clear and bold)
- Why this is worth exploring computationally
- Key variables
- Testable predictions
- Core assumptions
- Suggested directions for simulation
"""
                    reply = ask_llm(HYPOTHESIS_SYSTEM, prompt)
                    st.session_state.hypothesis = reply
                    st.session_state.stage = "hypothesis"
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                    st.rerun()

    with col_b:
        if st.button("Clear Hypothesis", use_container_width=True):
            st.session_state.hypothesis = None
            st.session_state.simulation_plan = None
            st.session_state.stage = "chat"
            st.rerun()

    with col_c:
        if st.button("📄 Turn Document into Simulation", use_container_width=True):
            if not document_text.strip():
                st.warning("Please paste a model description or paper excerpt first.")
            else:
                with st.spinner("Turning provided document/model text into simulation code..."):
                    doc_prompt = f"""The user provided the following model description or paper excerpt:

{document_text[:8000]}

Field context: {field}
User interest: {user_request}

Task:
1. Extract the key mechanisms, equations, or processes described.
2. Design a practical computational simulation that implements a simplified but faithful version of the described model using only NumPy, SciPy, Pandas, Matplotlib, NetworkX, SymPy, etc.
3. Write complete runnable Python code.
4. Assign main quantitative findings to a variable named `result`.
5. Do not invent external literature. Stay close to the provided text.
6. Clearly list simplifying assumptions in comments.

Return the Python code in a ```python block.
"""
                    doc_code = ask_llm(CODE_SYSTEM, doc_prompt)
                    if "```python" in doc_code:
                        doc_code = doc_code.split("```python")[1].split("```")[0].strip()
                    elif "```" in doc_code:
                        doc_code = doc_code.split("```")[1].split("```")[0].strip()

                    st.session_state.last_code = doc_code
                    st.session_state.hypothesis = st.session_state.hypothesis or f"(Derived from user-provided document in field: {field})"
                    st.session_state.stage = "simulate"
                    st.success("Simulation code generated from the provided document. Go to the Simulation Lab tab.")
                    st.rerun()

    if st.session_state.hypothesis:
        st.markdown("### Current Hypothesis")
        st.markdown(st.session_state.hypothesis)

        col_accept1, col_accept2 = st.columns(2)

        with col_accept1:
            if st.button("Accept & Auto-Build Full Simulation (Plan + Code)", type="primary", use_container_width=True):
                with st.spinner("Designing simulation plan and writing complete code... This may take a moment."):
                    # Step 1: Create detailed plan
                    plan_prompt = f"""Here is the hypothesis:

{st.session_state.hypothesis}

Create a detailed, practical, and creative simulation plan that can be implemented in Python.
Focus on making the simulation able to discover unexpected or non-obvious patterns.
"""
                    plan = ask_llm(SIMULATION_PLAN_SYSTEM, plan_prompt)
                    st.session_state.simulation_plan = plan

                    # Step 2: Immediately write full runnable code from the plan
                    code_prompt = f"""Hypothesis:
{st.session_state.hypothesis}

Simulation plan:
{plan}

Now write a complete, high-quality, fully runnable Python simulation based on the plan above.
The code should be ready to execute and capable of producing interesting quantitative results.
Assign main findings to a variable named `result`.
"""
                    generated_code = ask_llm(CODE_SYSTEM, code_prompt)

                    # Clean code block markers if present
                    if "```python" in generated_code:
                        generated_code = generated_code.split("```python")[1].split("```")[0].strip()
                    elif "```" in generated_code:
                        generated_code = generated_code.split("```")[1].split("```")[0].strip()

                    st.session_state.last_code = generated_code
                    st.session_state.stage = "simulate"
                    st.success("Simulation plan and full code have been automatically generated. Go to the Simulation Lab tab.")
                    st.rerun()

        with col_accept2:
            if st.button("Only create Plan (manual code later)", use_container_width=True):
                with st.spinner("Designing simulation plan..."):
                    plan_prompt = f"""Here is the hypothesis:

{st.session_state.hypothesis}

Create a detailed, practical simulation plan that can be implemented in Python.
"""
                    plan = ask_llm(SIMULATION_PLAN_SYSTEM, plan_prompt)
                    st.session_state.simulation_plan = plan
                    st.session_state.stage = "plan"
                    st.rerun()

        st.markdown("---")
        st.markdown("### 🔬 Adaptive Gap Exploration Mode")
        st.caption("This mode systematically varies parameters, looks for under-explored regions in the simulation space, and runs an adaptive feedback loop. It stays fully grounded in computational results.")

        if st.button("Start Adaptive Gap Exploration (Plan + Code + Parameter Sweep Design)", type="primary", use_container_width=True):
            with st.spinner("Designing adaptive gap-exploration simulation (this focuses on parameter scrutiny and feedback loops)..."):
                adaptive_prompt = f"""Hypothesis:
{st.session_state.hypothesis}

Design a simulation specifically for ADAPTIVE GAP EXPLORATION.

Requirements:
1. Identify the key parameters that could hide interesting or under-explored behavior.
2. Create a plan that systematically varies those parameters (sweeps, grids, adaptive sampling, or iterative refinement).
3. The simulation should try to detect regions where results change sharply, where outcomes are unstable, or where behavior is unexpected.
4. Include clear quantitative metrics that will be stored in a variable called `result`.
5. Make the design suitable for multiple iterative rounds (the code should be easy to modify based on previous outputs).
6. Stay strictly computational — do not claim real-world discovery.

First output a clear adaptive simulation plan, then output complete runnable Python code that implements a solid first version of this parameter-exploring simulation.
"""
                adaptive_response = ask_llm(SIMULATION_PLAN_SYSTEM + "\n\nAlso follow the CODE_SYSTEM rules when writing code.", adaptive_prompt)

                # Try to split plan and code
                plan_part = adaptive_response
                code_part = ""

                if "```python" in adaptive_response:
                    parts = adaptive_response.split("```python")
                    plan_part = parts[0].strip()
                    code_part = parts[1].split("```")[0].strip()
                elif "```" in adaptive_response:
                    parts = adaptive_response.split("```")
                    plan_part = parts[0].strip()
                    code_part = parts[1].strip() if len(parts) > 1 else ""

                st.session_state.simulation_plan = plan_part if plan_part else adaptive_response
                if code_part:
                    st.session_state.last_code = code_part
                else:
                    # Fallback: ask specifically for code
                    code_only = ask_llm(CODE_SYSTEM, f"Hypothesis:\n{st.session_state.hypothesis}\n\nPlan:\n{st.session_state.simulation_plan}\n\nWrite the full adaptive parameter-exploration simulation code. Assign findings to `result`.")
                    if "```python" in code_only:
                        code_only = code_only.split("```python")[1].split("```")[0].strip()
                    st.session_state.last_code = code_only

                st.session_state.stage = "simulate"
                st.success("Adaptive Gap Exploration plan and code generated. Go to the Simulation Lab and use Auto-Iterate for feedback loops.")
                st.rerun()


# --------------------------------------------------
# Simulation Lab
# --------------------------------------------------
with tab2:
    st.subheader("2. Simulation Laboratory")

    if not st.session_state.hypothesis:
        st.info("Generate and accept a hypothesis first. Use the big blue button to auto-build the full simulation.")
    else:
        if st.session_state.simulation_plan:
            with st.expander("Current Simulation Plan", expanded=False):
                st.markdown(st.session_state.simulation_plan)

        st.markdown("#### Simulation Code")
        st.caption("Code is automatically generated when you click 'Accept & Auto-Build Full Simulation'. You can still edit it.")

        default_code = st.session_state.get("last_code", """# No code generated yet.
# Go back to the first tab, generate a hypothesis, then click
# "Accept & Auto-Build Full Simulation (Plan + Code)"
""")

        code = st.text_area("Python simulation code", value=default_code, height=300)

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            run_clicked = st.button("▶ Run Simulation", type="primary", use_container_width=True)
        with col2:
            if st.button("Improve Code", use_container_width=True):
                with st.spinner("Asking LLM to improve the code..."):
                    improve_prompt = f"""Hypothesis:
{st.session_state.hypothesis}

Current plan:
{st.session_state.simulation_plan or "None"}

Current code:
```python
{code}
```

Improve or complete this simulation code. Return only the improved Python code.
"""
                    improved = ask_llm(CODE_SYSTEM, improve_prompt)
                    if "```python" in improved:
                        improved = improved.split("```python")[1].split("```")[0].strip()
                    elif "```" in improved:
                        improved = improved.split("```")[1].split("```")[0].strip()
                    st.session_state.last_code = improved
                    st.rerun()

        with col3:
            if st.button("Fresh Code from Plan", use_container_width=True):
                with st.spinner("Generating code from plan..."):
                    gen_prompt = f"""Hypothesis:
{st.session_state.hypothesis}

Simulation plan:
{st.session_state.simulation_plan}

Write a complete, runnable Python simulation script based on the plan.
Assign main findings to a variable named `result`.
"""
                    generated = ask_llm(CODE_SYSTEM, gen_prompt)
                    if "```python" in generated:
                        generated = generated.split("```python")[1].split("```")[0].strip()
                    st.session_state.last_code = generated
                    st.rerun()

        with col4:
            auto_iterate_clicked = st.button("🔄 Auto-Iterate 3 Rounds", use_container_width=True, help="Run → Interpret → Improve code → Repeat (3 times)")

        with col5:
            sensitivity_clicked = st.button("📊 Parameter Sensitivity", use_container_width=True, help="Generate and run parameter sensitivity analysis")

        if run_clicked:
            with st.spinner("Executing simulation..."):
                exec_result = execute_code(code)
                formatted = format_execution_result(exec_result)

                # Store in history
                st.session_state.simulation_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "code": code,
                    "result": formatted,
                    "success": exec_result["success"],
                })
                st.session_state.last_code = code
                st.session_state.stage = "simulate"

                st.markdown("### Execution Result")
                if exec_result["success"]:
                    st.success("Simulation finished successfully")
                else:
                    st.error("Simulation failed")
                st.code(formatted, language="text")

                # Ask LLM to interpret (strict conservative mode)
                with st.spinner("Interpreting results (conservative mode)..."):
                    interpret_prompt = f"""Hypothesis:
{st.session_state.hypothesis}

Simulation code:
```python
{code}
```

Exact execution output:
{formatted}

Provide a strictly conservative interpretation following the rules.
Use these headings:
### Computed Facts
(Only what the numbers/output actually show)

### Tentative Observations
(Very cautious possible readings)

### Limitations
(What this run cannot tell us)

### Possible Next Computational Steps
"""
                    interpretation = ask_llm(INTERPRET_SYSTEM, interpret_prompt)
                    st.markdown("### Interpretation (Conservative)")
                    st.markdown(interpretation)

        # --------------------------------------------------
        # Parameter Sensitivity Analysis
        # --------------------------------------------------
        if sensitivity_clicked:
            if not st.session_state.hypothesis:
                st.warning("Generate a hypothesis first.")
            else:
                with st.spinner("Designing Parameter Sensitivity Analysis..."):
                    sens_prompt = f"""Hypothesis:
{st.session_state.hypothesis}

Current simulation plan (if any):
{st.session_state.simulation_plan or "None"}

Current code (if any):
```python
{code}
```

Create a clear Parameter Sensitivity Analysis simulation in Python.

Requirements:
1. Identify the most important parameters from the hypothesis/plan.
2. Systematically vary each key parameter while holding others at baseline values (one-at-a-time sensitivity) OR use a simple sampling approach.
3. Record how the main output metric(s) change with each parameter.
4. Store structured results in a variable named `result` (preferably a dict or pandas DataFrame summarizing sensitivity).
5. Print clear sensitivity summary statistics (e.g. range of outcomes, which parameter caused the largest change).
6. Use only numpy, scipy, pandas, matplotlib, seaborn, etc.
7. Return complete runnable Python code.

Return only the Python code in a ```python block.
"""
                    sens_code = ask_llm(CODE_SYSTEM, sens_prompt)
                    if "```python" in sens_code:
                        sens_code = sens_code.split("```python")[1].split("```")[0].strip()
                    elif "```" in sens_code:
                        sens_code = sens_code.split("```")[1].split("```")[0].strip()

                    st.session_state.last_code = sens_code
                    st.success("Parameter Sensitivity Analysis code generated. Running it now...")

                    # Immediately run it
                    exec_result = execute_code(sens_code)
                    formatted = format_execution_result(exec_result)

                    st.session_state.simulation_history.append({
                        "timestamp": datetime.now().isoformat(),
                        "code": sens_code,
                        "result": formatted,
                        "success": exec_result["success"],
                        "type": "sensitivity_analysis",
                    })

                    st.markdown("### Parameter Sensitivity Analysis Result")
                    if exec_result["success"]:
                        st.success("Sensitivity analysis finished successfully")
                    else:
                        st.error("Sensitivity analysis failed")
                    st.code(formatted, language="text")

                    # Conservative interpretation
                    with st.spinner("Interpreting sensitivity results..."):
                        sens_interpret = ask_llm(INTERPRET_SYSTEM, f"""Hypothesis:
{st.session_state.hypothesis}

Parameter Sensitivity Analysis code:
```python
{sens_code}
```

Exact output:
{formatted}

Provide a conservative interpretation focused on:
### Computed Sensitivity Facts
### Which parameters appear most influential (from the numbers only)
### Limitations of this sensitivity analysis
### Suggested next parameter ranges to explore
""")
                        st.markdown(sens_interpret)

        # --------------------------------------------------
        # Auto-Iterate 3 Rounds
        # --------------------------------------------------
        if auto_iterate_clicked:
            if not code or code.strip().startswith("# No code generated"):
                st.warning("Please generate simulation code first (use 'Accept & Auto-Build Full Simulation').")
            else:
                st.markdown("### 🔄 Auto-Iterate 3 Rounds")
                progress = st.progress(0)
                status_text = st.empty()
                results_container = st.container()

                current_code = code

                for round_num in range(1, 4):
                    status_text.markdown(f"**Round {round_num}/3** — Running simulation...")
                    progress.progress((round_num - 1) / 3)

                    # 1. Execute
                    exec_result = execute_code(current_code)
                    formatted = format_execution_result(exec_result)

                    # Store in history
                    st.session_state.simulation_history.append({
                        "timestamp": datetime.now().isoformat(),
                        "code": current_code,
                        "result": formatted,
                        "success": exec_result["success"],
                        "round": round_num,
                    })

                    with results_container:
                        st.markdown(f"#### Round {round_num} Result")
                        if exec_result["success"]:
                            st.success(f"Round {round_num} finished successfully")
                        else:
                            st.error(f"Round {round_num} failed")
                        st.code(formatted[:2000] + ("..." if len(formatted) > 2000 else ""), language="text")

                    # 2. Interpret + decide improvements
                    status_text.markdown(f"**Round {round_num}/3** — Interpreting & improving code...")

                    improve_prompt = f"""Hypothesis:
{st.session_state.hypothesis}

Current simulation plan:
{st.session_state.simulation_plan or "None"}

Current code (Round {round_num}):
```python
{current_code}
```

Exact execution output:
{formatted}

This is round {round_num} of 3 in an ADAPTIVE GAP EXPLORATION loop.

Your goals for this round:
1. State only what the computed output actually shows.
2. Scrutinize the parameters: Which parameter ranges were explored? Which regions look under-explored, unstable, or interesting?
3. Propose specific new parameter values, wider/narrower sweeps, or adaptive sampling to better explore potential gaps.
4. Return a complete improved Python simulation code that implements those parameter adaptations.
5. Keep assigning main quantitative findings to a variable named `result`.
6. Stay strictly within computational exploration. Do not invent real-world conclusions.

Return the improved code in a ```python block.
"""
                    improved_response = ask_llm(CODE_SYSTEM, improve_prompt)

                    # Extract code
                    if "```python" in improved_response:
                        new_code = improved_response.split("```python")[1].split("```")[0].strip()
                    elif "```" in improved_response:
                        new_code = improved_response.split("```")[1].split("```")[0].strip()
                    else:
                        new_code = improved_response

                    current_code = new_code
                    st.session_state.last_code = current_code

                    with results_container:
                        with st.expander(f"Round {round_num} — Code improvements applied"):
                            st.code(current_code, language="python")

                progress.progress(1.0)
                status_text.markdown("**✅ Auto-Iterate complete (3 rounds).** Final code is loaded in the editor above.")
                st.success("Auto-Iterate 3 Rounds finished. Check the history tab for full details.")
                st.rerun()


# --------------------------------------------------
# Article Draft
# --------------------------------------------------
with tab3:
    st.subheader("3. Research-Style Article Draft")

    if not st.session_state.hypothesis:
        st.info("Generate a hypothesis and run at least one simulation first.")
    else:
        if st.button("Generate Article Draft from Current Work", type="primary"):
            with st.spinner("Drafting conservative, tightly grounded report..."):
                history_summary = ""
                for i, h in enumerate(st.session_state.simulation_history[-6:], 1):
                    history_summary += f"\n\n--- Run {i} ---\nSuccess: {h['success']}\n{h['result'][:1800]}"

                article_prompt = f"""Create a structured exploratory computational report based ONLY on the material below.

Hypothesis:
{st.session_state.hypothesis}

Simulation plan:
{st.session_state.simulation_plan or "Not formally defined"}

Executed simulation outputs (ground truth for all quantitative claims):
{history_summary or "No simulations have been run yet."}

Mandatory requirements:
- Every quantitative statement must be traceable to the executed outputs above.
- Do NOT invent any papers, citations, or external empirical findings.
- Do NOT claim real-world scientific discovery or strong novelty.
- Use cautious language ("the simulations show", "under these assumptions", "these computational experiments indicate").
- Include a clear and prominent Limitations section.
- If the results are weak or inconclusive, state that directly.

Recommended structure:
1. Title (neutral, not sensational)
2. Abstract
3. Introduction (short, cautious)
4. Hypothesis
5. Methods / Simulation Design
6. Results (closely tied to the executed outputs)
7. Discussion (tentative only)
8. Limitations (required and detailed)
9. Conclusion & Possible Future Computational Work
"""
                draft = ask_llm(ARTICLE_SYSTEM, article_prompt)
                st.session_state.article_draft = draft
                st.session_state.stage = "article"
                st.rerun()

        if st.session_state.article_draft:
            st.markdown(st.session_state.article_draft)
            st.download_button(
                "Download Draft as Markdown",
                data=st.session_state.article_draft,
                file_name=f"research_draft_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                mime="text/markdown",
            )


# --------------------------------------------------
# History
# --------------------------------------------------
with tab4:
    st.subheader("Simulation History")
    if not st.session_state.simulation_history:
        st.info("No simulations run yet.")
    else:
        for i, entry in enumerate(reversed(st.session_state.simulation_history), 1):
            with st.expander(f"Run {len(st.session_state.simulation_history) - i + 1} — {entry['timestamp']} — {'✅' if entry['success'] else '❌'}"):
                st.code(entry["code"], language="python")
                st.text(entry["result"])


# --------------------------------------------------
# Footer
# --------------------------------------------------
st.divider()
st.caption(
    "Research Simulation Chatbot • Standalone exploratory tool • "
    "Results are computational experiments only • Always verify assumptions and limitations"
)
