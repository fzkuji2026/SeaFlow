# SeaFlow

Code for **Self-Evolving Agentic Workflows via Semantic Decomposition of Complex Tasks**
(NeurIPS 2026 IAB Workshop, GLEE Competition Track).

The agent plays the Bargaining, Negotiation and Persuasion games of
[GLEE](https://glee-competition.com/). Each workflow decomposes a game into subtasks by the
observed situation and solves each subtask with code; an optional LLM module only picks among
legal alternatives that the code computes. The workflows were revised from execution feedback
with self-evolution built on [OpenProgram](https://github.com/fzkuji-neo/OpenProgram).

This snapshot is the final workflow revision reported in the paper (`a1875a0`).

## Layout

- `glee_agent/`: game parsing, situation classification, per-family workflows (`workflows/`, `families/`) and the optional LLM judge.
- `glee_runtime/`: game queue, move scheduling and networking.
- `scripts/`: workflow evolution (`evolve.py`), replay diagnosis, reference building and model fitting.
- `docs/`: method notes and workflow diagrams. `WORKFLOW.md` summarizes the decision pipeline.
- `tests/`: unit tests.

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # set GLEE_API_KEY
PYTHONPATH=vendor:. python run_agent.py
```

Set `GLEE_USE_OPENPROGRAM=1` to run the agentic functions through OpenProgram. The optional LLM
judge (`GLEE_LLM_JUDGE=1`) reads `XAI_API_KEY` or OpenProgram's xAI credentials.

## Tests

```bash
PYTHONPATH=vendor:. pytest -q
```
