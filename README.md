# SPM_agent_simple

A minimal agentic system for running autonomous PFM (Piezoresponse Force Microscopy)
experiments on a scanning probe microscope. Built on **LangGraph**: an LLM decides
*what* to measure next (scan / hysteresis loop / stop), while all measurement
processing, parameter extraction, and map arithmetic stay deterministic — so every
LLM contribution is separable and benchmarkable.

## How it works

````mermaid
flowchart LR
    DEC{{"decide (LLM)"}} -- scan --> AQ["acquire"] --> AN["analysis"]
    DEC -- loop --> PK["pick point"] --> AQ
    DEC -- stop --> E([END])
    AN --> DEC
````

Each cycle: **decide** (Claude, structured output, deterministic guards) →
**acquire** (SPM MCP server; currently mocked with `.ibw` files) → **analysis**:

- **image** → channel recommendation (GPT, structured) → agentic segmentation
  (Claude + image-op toolbox) → importance map (Claude coding agent in an
  isolated Python sandbox; deliverables validated, final map computed deterministically)
- **loop** → SS-PFM on/off segmentation → hysteresis parameter extraction
  (deterministic) → loop review (Claude, vision QC)

All results accumulate as append-only `ExperimentalRecord`s; the decision node
sees a deterministic multimodal digest of everything measured so far.

## Install

````bash
git clone https://github.com/Slautin/SPM_agent_simple.git
cd SPM_agent_simple
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
python -m spm_agent.sandbox        # one-time: build the isolated code-exec sandbox
````

Create `.env` with `ANTHROPIC_API_KEY` and `OPENAI_API_KEY`.
Set the instrument MCP URL in `src/spm_agent/config.py` (`SPM_MCP_SERVER_CONFIG`).

## Usage

The current entry point is `notebooks/14_analysis_graph_and_decision.ipynb`:
set `EXPERIMENT_TASKS` / `EXPERIMENT_CONTEXT`, call `new_run()`, and invoke the
graph. Artifacts land in `src/runs/<timestamp>/` (segmentation masks, importance
components + scoring code, loop data + annotated figures, decision digests).

## Repository layout

````
src/spm_agent/
├── config.py           # models, caps, dirs, MCP configs
├── sandbox.py          # isolated venv for LLM-generated code
├── graphs/             # analysis graph (image / loop branches)
├── nodes/              # LangGraph nodes (one file each)
├── prompts/ schemas/   # prompt builders, Pydantic contracts
├── states/ tools/ utils/ mcp/
notebooks/              # development history; 14 = full experiment loop
````

## Status

Working end-to-end with mocked acquisition. Not yet wired: real instrument
acquisition, deterministic point picking with spatial penalty, top-level graph
in `src/`. See `SYSTEM_REPORT_2026-07-16.md` for full technical details.
````
````
