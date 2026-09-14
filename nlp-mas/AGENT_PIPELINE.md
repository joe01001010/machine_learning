# Calculator agent pipeline

## Run in WSL

```bash
conda activate nlp-mas
cd ~/git/machine_learning/nlp-mas
./serve-models.sh Qwen/Qwen3-4B-Instruct-2507 Qwen/Qwen3-4B-Instruct-2507 Qwen/Qwen3-4B-Instruct-2507
```

Wait for all three servers to be ready. In another terminal:

```bash
conda activate nlp-mas
cd ~/git/machine_learning/nlp-mas
python validate_requirements.py calculator_v1_requirements.jsonl calculator_v1/compute.py
```

Optional flags: `--scope FILE`, `--output DIRECTORY`, `--max-tokens 768`,
`--max-turns 12`, `--calculator-timeout 5`, `--http-timeout 180`.
The three `--endpoints` can override the local ports 8001, 8002 and 8003.

The client uses Python's standard library plus `jsonschema`, already present in
the inspected nlp-mas environment. Template building uses its installed
`transformers` package and the model tokenizer. No agent framework is required.

## Roles as tools

The requirements engineer starts the run by calling `create_workspace`.
The application then supplies each requirement with the approved scope:

1. Requirements engineer calls `submit_to_test_engineer(decision, rationale, tests)`.
   The application stores the decision and test list. For ACCEPT, this handler
   invokes the test-engineer conversation; other decisions are recorded as NOT_TESTED.
2. Test engineer calls `run_calculator(test_case_id)` for each test. The host retrieves
   the immutable approved input, launches the specified calculator, and saves evidence.
3. After every test has evidence, the tester calls `verification_engineer()`.
4. That tool invokes the analysis agent once per test, supplying its requirement,
   scope, definition and evidence. The analyst calls `submit_verdict(...)` to save
   PASS, FAIL or INCONCLUSIVE and its explanation.

Agents generate actual function tool calls. Python dispatches them and returns
tool-role messages linked by tool_call_id. Calls between roles are implemented as
Python functions that invoke the respective local vLLM server. The models themselves
do not perform HTTP requests or filesystem operations. The servers still provide
the inference API behind these functions.

The application enforces stage order, allowed tool names, argument schemas, and
approved test IDs. The model chooses the decision, test content, execution request,
and verdict. It does not choose arbitrary next roles or filesystem paths. This is
a constrained agent-as-tool workflow, not unrestricted agent planning.

## Server-side prompts

Your installed vLLM 0.29.0 supports `--chat-template` but not `vllm serve --system-prompt`.
Role instructions live in `prompts/<role>.txt`. Before starting the servers,
`build_server_templates.py` loads each model's original tokenizer chat template and
prepends that role's system message, retaining the original tool-call formatting.
The client sends no system message. Restart servers after editing prompt files.

Each launch archives exact prompts and generated templates in `.agent-logs.*/templates/`.
The existing shutdown trap and GPU budgets are retained. The default context length
is raised from 2048 to 3072 to accommodate scope and tool schemas; KV cache stays at
512 MiB per server. Override using `MAX_MODEL_LEN=2048 ./serve-models.sh ...` if needed,
but insufficient context can cause HTTP errors. There is no silent context truncation.

## Artifact layout

```text
results/calculator_v1/
  verification_<random>.json
  artifacts/<same-random>/
    manifest.json
    calculator_source.json
    requirements.json
    scope.json
    trace.jsonl
    decisions/REQ-001.json
    evidence/REQ-001-T01.json
    verdicts/REQ-001-T01.json
```

All artifact names are chosen by the application. The create tool can only create
the current run directory beneath the user-configured output root. Typed submission
tools save artifacts automatically; there is no unrestricted shell or write-file tool.
The tester can request `read_artifact` for its authorized decision artifact. Other
needed records are supplied directly to each conversation, without exposing reference
answers or unrelated run files.

Application-created files use exclusive creation and are never overwritten by tools.
This prevents accidental agent overwrites; it is not cryptographic tamper protection
against other programs or manual file edits. The source snapshot and SHA-256 hashes
identify the calculator's Python files. Source changes during a run abort execution.

Trace logs preserve the sent messages, returned tool calls, server usage fields when
present, tool results, and errors. Inference context is compact: one requirement/test
and current application state plus the last tool exchange. Full conversation history
is on disk, not permanently retained inside the model server.

## Interpreting reports

`workflow_status: COMPLETED` means the pipeline finished, not that software passed.
Inspect `requirements[].verdicts[]`. Rejected or unclear requirements are NOT_TESTED.
A subprocess error is evidence, not automatically an orchestration failure. Tool
timeouts/launch errors and unavailable or truncated evidence require INCONCLUSIVE.
Evidence records preserve stdout including its newline; comparison removes only one
trailing newline as defined by the server instructions.

The supplied scope expects integer results as integer strings. The inspected
calculator uses Python true division, so `4 / 2` prints `2.0`, not `2`. An analyst
may correctly flag that mismatch. Fractional formatting and division by zero are
unspecified; invented exact outcomes for them should not be accepted as justified.

The application does not replace the analyst's substantive judgments with a second
calculator or a deterministic pass/fail comparator. Grade those judgments independently
for your research. Passing generated examples is not exhaustive requirement coverage.

On a downstream failure, completed artifacts are preserved and a report marks ERROR
with incomplete requirement IDs. The script exits nonzero. Re-run to start a new ID;
automatic resume is not implemented. An initial failure before create_workspace cannot
produce a run artifact. No agent-success claims are synthesized for failed stages.

## Validation

```bash
python -m unittest -v test_pipeline.py
```

Tests use scripted model responses and real subprocess execution. They cover role
handoffs, ID linkage, preserved output, rejection and clarification paths, crashes,
timeouts, unauthorized tools, duplicate requirements, source changes, and partial
failure. They do not measure Qwen's task accuracy or replace a live vLLM smoke test.
