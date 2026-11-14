# Multi-agent software build experiment

Give the system a customer prompt. Four separate local model servers turn it into requirements, design tests, write a program, execute it, and independently verify the results. Failures return to the coder for a bounded repair cycle. The test engineer handles both test design and execution.

`experiment.py` is the build-and-repair workflow. `validate_requirements.py` is the earlier calculator-validation workflow; it uses a different role arrangement and does not implement this build loop.

## Demo the feedback loops

From the project directory in WSL, run the deterministic demo in an environment with Python, jsonschema, and GCC.

```bash
python demo_feedback.py
```

The demo uses small calculator and rideshare scenarios with scripted agent responses and real test execution. It exercises the same workflow as a live run and makes the feedback loops repeatable for a presentation. Run `python demo_feedback.py calculator` or `python demo_feedback.py rideshare` for just one scenario. Results go under `results-demo/<scenario>/` by default. The manifest records `agent_transport` as `injected` for the demo and `http` for live runs. Use live runs to evaluate model performance.

1. Follow the first requirements proposal, the verifier's revision request, and the next proposal for the same source block. Compare them with the original excerpt and the accepted requirements.
2. Follow the test plan revision. The new calculator test exercises precedence with `2+3*4`. The new rideshare test sets the minimum fare to `20.00` to exercise the fare floor.
3. Follow a failing test into a repair. Compare its expected and actual results in round 0, then open the evidence for the same test in round 1. The test and its expected result stay fixed while the source changes.

For live requirements generation from the project prompts, start the model servers as shown below and run these commands in a second WSL terminal.

```bash
python experiment.py --requirements-only --prompt-file calculator_spec.txt
python experiment.py --language c --requirements-only --prompt-file rideshare_experiment_prompt.txt
```

These runs use the requirements and verification endpoints. They stop after requirements review, so `PASS` means the requirements were finalized. Add `--review-requirements` to an ordinary build command to include the same review before test design, implementation, and repair.

The terminal prints each run's result directory. Use the saved artifacts to present what happened.

| Artifact | What to show |
| --- | --- |
| `customer_prompt.json` and `requirements_sources.json` | The complete input and the numbered excerpts given to the agents |
| `requirements_review/block-NNN/proposal-0.json` and `review-0.json` | The first proposed requirements and the verifier's decision and feedback. Block numbering starts at `001`. |
| `requirements_review/block-NNN/proposal-1.json` and `review-1.json` | The revision and its review, when the first proposal needed changes |
| `requirements.json` | The finalized requirements and their source quotes |
| `test_design/R001/` | The proposed tests and verifier feedback before the test plan was frozen |
| `test_plan.json` and `rounds/000/evidence/`, `rounds/001/evidence/` | The frozen test and the observed results before and after a repair in a build run |
| `report.json` and `metrics.html` | The run outcome and the metrics available for that run |

### Explain how you cover the prompt

Start by defining the scope with the customer. Include actors and inputs, normal flows, limits and error paths, outputs and formats, and exclusions. Include examples where a rule can be read in different ways. For the calculator, separate exact terminating decimals, repeating cycles, and precision for irrational results. For rideshare, specify the authentication, fare and minimum-fare rules, which trips and surveys count in totals, and what each organization must report.

Preserve the original wording and record any clarifications separately. The controller splits the prompt into numbered excerpts and links requirements to their supporting text. In review mode, the verifier checks each block for missing obligations, unsupported additions, and ambiguous intent. Its feedback goes to the requirements engineer for another draft of the same block. Each block has up to three proposals. A material question is recorded for the customer. Revise the prompt with the answer and rerun to keep each input and result auditable.

Use an independently written checklist of expected obligations to measure completeness within the stated scope. Divide the number of obligations correctly expressed in the final requirements by the number on the checklist. Keep source-processing progress and reviewer acceptance separate from this coverage measure. For quality scoring, use the independent annotations described under [Automatic research reporting](#automatic-research-reporting). Track model calls, latency, and available token counts alongside quality. Keep the scripted demo and live measurements separate.

For a concrete calculator checklist, the consolidated specification has these 20 obligations. Score each item separately, even when a single requirement expresses several.

| Area | Obligations |
| --- | --- |
| Interface and errors (4) | Python, one plain-text expression passed as a command-line argument, plain-text result, error response for invalid input |
| Operators and grammar (12) | Addition, subtraction, multiplication, division, parentheses, PEMDAS, `^`, `**`, fractional exponents, modulo, floor division, decimal operands |
| Signs and precision (4) | Negative operands, exact terminating decimals where possible, nonrepeating prefix plus one full repeating cycle for repeating rational answers, 17 significant digits for nonrepeating irrational results from fractional powers |

For rideshare, map every branch to the actor that triggers it and the resulting state or output. For example, check alphabetical order for rider selection and for administrator summaries as two separate obligations. Check rejected inputs and canceled trips against the organization totals, and check summary-file contents separately from the console output. List unresolved decisions before treating the prompt as ready for an end-to-end test.

To measure the effect of requirements feedback, run the same build prompt and model settings with and without `--review-requirements`. Score both against the same independently prepared reference and compare the number of calls and time spent. Report the sample size and keep incomplete runs in the outcome counts.

## Start the model servers

In the project's WSL environment:

```bash
conda activate nlp-mas-serving
./serve-models.sh Eslzzyl/Qwen3-4B-Instruct-2507-AWQ
```

Keep this terminal open. The launcher starts the servers one at a time and waits for readiness. The roles and ports are:

| Role | Port | Work |
| --- | --- | --- |
| requirements-engineer | 8001 | Extract requirements from customer intent |
| test-engineer | 8002 | Design tests, then request host execution |
| coder | 8003 | Create source and repair demonstrated defects |
| verification-engineer | 8004 | Review requirements when enabled, review test plans, and assess execution evidence |

Prompts are read from `prompts/` when the servers start and saved with that run's templates under `.agent-logs.*`. Restart the model servers after changing those prompt files. A running server retains the instructions with which it started. Stop the launcher with Ctrl+C to stop its entire server group before starting a replacement.

The example launch above uses four independent 4-bit model copies. The launcher defaults to BF16 activations, a 4096-token context, and a dynamically scaled FP8 cache of 320 MiB per instance. Environment overrides include `MAX_MODEL_LEN`, `KV_CACHE_MEMORY_BYTES`, `KV_CACHE_DTYPE`, `ATTENTION_BACKEND`, `GPU_MEMORY_UTILIZATION`, and `STARTUP_TIMEOUT`. With an explicit cache size, vLLM uses that byte count to size the cache.

### Give the coder more context

For the Qwen3-4B AWQ setup above, you can give the coder a 6144-token context and a 480 MiB cache. Stop the current launcher before running this in the server terminal.

```bash
CODER_MAX_MODEL_LEN=6144 CODER_KV_CACHE_MEMORY_BYTES=503316480 \
  MAX_NUM_BATCHED_TOKENS=256 \
  ./serve-models.sh Eslzzyl/Qwen3-4B-Instruct-2507-AWQ
```

The larger cache reserves 160 MiB more GPU memory than the default. The other roles inherit the original 4096-token context and 320 MiB cache. The smaller processing batches trade prompt throughput for less work per batch. Omit `MAX_NUM_BATCHED_TOKENS` to keep the default 512-token batch size if GPU memory allows it. The launcher prints each role's context and cache settings on startup.

## Build a calculator from a prompt

In a second terminal with the same environment active:

```bash
python experiment.py --prompt-file calculator_spec.txt --calculator-reference --python-entry-function main
```

`calculator_spec.txt` is an **assistant-prepared consolidated specification**, derived from the actual calculator request and the user's answers. It separates the terminating-decimal, repeating-cycle, and irrational-precision rules to make the intended build explicit. It is an interpreted specification, not a verbatim user dataset.

`calculator_experiment_prompt.txt` preserves the **raw user request and clarification record** separately. It requests arithmetic with PEMDAS, `^` and `**` exponents including fractional exponents, modulo and floor division, plain-text results/errors, exact terminating decimals, nonrepeating digits plus one repeating cycle for rational results, and 17 significant digits for irrational results. Use `--prompt-file calculator_experiment_prompt.txt` instead when evaluating intake directly from that unmodified wording. Keep these two prompt conditions distinct when reporting experimental results. The controller's interface conventions are recorded separately from user intent.

The optional `calculator_build_prompt.txt` is a separate, simpler **example customer prompt** derived from the original calculator example. It excludes additional behaviors and is not the specification for the actual calculator experiment. Select it explicitly with `--prompt-file calculator_build_prompt.txt` if you want that smaller demonstration; leave `--calculator-reference` disabled for its restricted scope.

You can also provide text directly:

```bash
python experiment.py --prompt "Build a command-line program that prints Hello followed by a newline and exits with status 0."
```

With neither prompt option, the program asks for a prompt on standard input. The default language is Python and the default entrypoint is `main.py`. With no `--python-entry-function`, Python runs as an ordinary script. The recommended calculator command selects explicit `main()` function invocation.

To use the existing calculator as the starting source rather than create a new one:

```bash
python experiment.py --prompt-file calculator_spec.txt --calculator-reference --python-entry-function main \
  --seed-dir calculator_v1 --entrypoint compute.py
```

Only Python source, or C source and headers for a C run, is copied from the seed into the new run's app directory. The original `calculator_v1` directory remains unchanged.

## Explicit Python entry function

`--python-entry-function main` uses the immutable `run_python_app.py` invocation adapter. It loads the generated module under `__application__`, then calls its zero-argument `main()` function with the saved arguments available through `sys.argv`. The harness interprets an integer return as the exit status, treats `None` as success, and preserves `SystemExit`. It contains no calculator mathematics.

The generated module must define the configured function and avoid calling it unconditionally at module scope. A normal `if __name__ == "__main__"` guard does not run while the adapter loads the module. This invocation mode is explicit and recorded in the manifest; ordinary Python script mode and C invocation remain available.

## What the loop does

1. Intake numbers original excerpts such as `[S001]`, each at most 200 characters. `customer_prompt` contains only the CURRENT block of up to three excerpts; `full_prompt_context` supplies background. One compact submission covers the entire block with up to three shall-statements and REQUIRED `complete=true`. False is invalid. Texts are limited to 300 characters, IDs continue from `next_requirement_id`, and supporting `source_id` values are restricted to the current block. The host copies original quotations exactly. Headings or already-covered material use an empty list with true. With review disabled, each valid submission advances the block. With review enabled, the verifier must accept the proposal before the block advances. The combined list freezes after all blocks advance and requires at least one grounded requirement.
2. The test engineer proposes 1-3 tests per requirement before the coder implements the program. Tests specify arguments, stdin, expected stdout, and the expected exit code. With `--calculator-reference`, an independent mathematical reference checks expected values; invalid-input requirements receive a malformed-input hint and an exit-code schema restricted to 1. Contradictory expectations are rejected for the model to revise; the host does not rewrite them. The independent verifier reviews scope, coverage, and grounding before tests are frozen. Rejected reviews permit at most three proposals per requirement; tool/turn limits also apply.
3. The coder writes source inside the run's app directory. The `write_file` tool requires `content` as an array of source lines, preserving leading indentation; the host joins lines with newlines. Subsequent chunks use append mode. Each write includes `complete`: true on the final chunk hands off immediately for build/testing; false means more chunks or files remain. `finish_build` is an alternative handoff. The host prepares the executable program and saves a source snapshot.
4. The test engineer requests execution of the saved tests. The host records actual process output and exit codes.
5. The verifier assesses each frozen test and actual execution. Its raw verdict is retained. A separate `controller_assessment` routes a completed stdout/exit-code mismatch to repair even if the model calls it INCONCLUSIVE. An observed mismatch is not missing execution evidence. Every test runs again after each successful build; build failures return directly to the coder.

An output-truncated response executes no tool action and saves no partial batch. If intake truncates before a block's first proposal, the source-block size and requirement batch limit shrink to one. All unconsumed excerpts remain queued. A reviewed revision keeps the same source block. The controller supplies stage-specific shortening guidance and stops INCOMPLETE after three consecutive truncated responses; existing turn and run-wide call limits still apply. Accepted requirement batches remain archived even when intake cannot finish.

Repeated requirement text, compared after whitespace normalization, is reused without adding another fact. In a mixed batch, all new facts are retained with contiguous canonical IDs while earlier IDs stay unchanged. The batch log preserves submitted requirements, duplicate aliases in `duplicates`, and `id_assignments`. Invalid IDs, unknown source IDs, false completion, and truncated responses cannot partially commit a batch or advance its block.

Requirements and expected tests stay fixed during repairs. Code cannot redefine success by changing expected results. Missing material intent pauses the run for user input. Repairs stop at the configured limit. A program test timeout or genuinely incomplete execution evidence remains INCONCLUSIVE and stops with NEEDS_USER; the current workflow does not automatically repair hangs. Requirements intake can ask about actual missing intent. Generic reference-free planning can also ask. Reference-backed test design/review offers only reference and submission/review tools; known arithmetic facts are computed rather than reopened as user questions. Complete execution verification receives no `ask_user` tool. Passing the generated tests establishes those tested cases, not exhaustive correctness.

Stdout comparison removes at most one terminal LF or CRLF from each expected and actual string; all other characters and whitespace remain significant. Exit codes must match exactly. Raw stdout is always archived. Presence of one terminal newline is outside this comparison's coverage.

The coder can page through source and failure feedback. Each model call receives current controller state, the most recent action result, the step number, and recent action summaries; full exchanges are archived in the trace. Source writes return the current tail. The controller keeps recent source pages in context and refreshes them after a write. The read tools offer unread source and feedback pages. No-op writes are rejected. Python builds use the standard library only. C builds use the C standard library and libm.

On WSL, when `bwrap` is available, build and test subprocesses see a read-only host filesystem, a writable app working directory, and separate process/network namespaces with no network access. Without it, the recorded isolation mode is `process_limits_only`. Executions also have time, memory, and output limits. Each evidence record identifies the isolation mode used.

## Adaptive test drafting

For tests with agent-authored expectations, the first truncated test-design response switches that and later requirements to bounded drafting. The truncated response executes no tool action. The host first requests the test count and metadata: arguments, exit code, line counts, line endings, and final-newline flags. It then requests exact stdin/stdout lines in bounded arrays, controlling their order and chunk size. Blank lines and tabs are preserved; further truncation reduces chunk size without consuming unfinished content.

Metadata and line chunks use separate model calls, so bounded drafting can need more turns than a single submission. The default is 32 turns per agent stage, recorded as `max_turns` in the manifest. Explicit lower limits remain respected, and the run-wide default stays at 200 model calls.

Only complete drafts enter normal submission checks. The independent reviewer then receives one full case at a time plus the other cases' plan metadata; every case must be accepted before the plan is frozen. Tests need documented exact console behavior. Missing expected wording requires clarification rather than invented output.

Draft events are archived at `test_drafts/Rnnn/attempt-N/eventNNN.json`. The run saves its `test_drafts.py` helper and records its hash in `test_drafting_protocol.json`. This fallback uses the same experiment command and model servers, with no model restart. An earlier INCOMPLETE run stays incomplete; a retry creates a separate run and must establish its own results.

## Run the original calculator evaluation

Use the basic reference for the original seeded calculator and its integer-operand prompt.

```bash
python experiment.py --language python --entrypoint main.py --seed-dir calculator_v1 \
  --prompt-file calculator_v1_prompt.txt --calculator-basic-reference
```

In this mode, the test engineer supplies one arithmetic expression as the only command-line argument. The controller independently computes exact expected output, supplies empty stdin, and archives the result in `test_reference/`. The verifier still reviews coverage before tests freeze. The basic profile extracts the product requirements above the `Examples of required behavior:` heading. The full prompt remains available to test design and review. The basic reference uses SymPy in the controller environment. It covers integer operands, parentheses, and binary addition, subtraction, multiplication, and division with terminating decimal results. Out-of-scope inputs are rejected during test drafting. The full calculator specification uses the separate profile below.

The terminal prints rejected review feedback and the requirement ID. Inspect `test_design/RNNN/proposal-N.json` and the matching `review-N.json` in the run directory to compare a proposal with its review.

## Optional calculator reference profile

`--calculator-reference` is off by default. Enable it for the full calculator experiment shown above. It supplies `reference_math` during test design/review and independently checks the proposed expected stdout and exit code before accepting a plan. Checks, including rejected expectations, are archived under `test_reference/`.

Each agent stage permits at most three explicit `reference_math` queries before removing that tool to bound query loops. Automatic reference validation still checks every proposed case. Rejection feedback identifies the input expression, proposed output, and computed reference so corrections stay linked to the right test.

The reference lives in `calculator_oracle.py` and uses SymPy in the controller environment. It computes expected mathematics only: it is not the generated calculator, is not copied into the app, and does not execute the program under test. The generated Python application still uses the standard library only.

This profile covers the full declared arithmetic grammar: addition, subtraction, multiplication, division, parentheses, `^`/`**` powers, Python-style modulo and floor division, decimal literals, and negative numbers. Exact rational results retain terminating decimals or the nonrepeating digits plus one complete repeating cycle. Irrational results use 17 significant digits with half-even rounding and unnecessary trailing zeros removed.

Powers follow the principal-power real-number profile: a non-real principal result is invalid input. For example, `(-8)^(1/3)` is invalid in this profile. Syntax/domain errors produce the declared error response. Limits on input/AST size, numeric magnitude, powers, output size, and numerical evaluation raise `ValueError` as reference limitations; they do not establish additional invalid-input requirements for the generated app.

Use this flag only when those mathematical/domain conventions match the experiment. A restricted two-integer addition smoke test, a greeting program, the simpler calculator example, a C application, or a custom grammar must leave it disabled. The full reference assumes the listed operators/domain are supported, which can contradict a restricted application's intended invalid inputs. Keep reference-assisted and reference-free conditions separate in experimental reporting.

## Results and limits

Each invocation creates a unique run directory under `results/` by default. It contains `customer_prompt.json`, `manifest.json`, `report.json`, and `trace.jsonl`. Successful intake adds `requirements.json`. A build run also saves its current source in `app/` and saves `test_plan.json` once tests are frozen. `controller_source.py` snapshots the controller used to start that run, with its SHA-256 recorded as `controller_sha256` in the manifest. Enabled reference/function modes also snapshot `calculator_oracle.py` and `run_python_app.py`; their hashes are recorded as `calculator_oracle_sha256` and `python_launcher_sha256`. The original unnumbered prompt remains in `customer_prompt.json`, and `requirements_sources.json` maps source IDs to original excerpts. Accepted intake batches are archived under sequential `requirements_batches/batch-NNN.json` names, recording source IDs, original submissions, duplicate aliases, ID assignments, `block_complete`, and `intake_complete`. The report records `requirements_staged_count` as batches are accepted. The final `requirements.json` is written only when the last source block is complete, so partial intake does not launch later stages. Test design proposals and independent reviews are archived under paths such as `test_design/R001/proposal-0.json` and `test_design/R001/review-0.json`. Numbered directories such as `rounds/000/` contain `source/`, `build.json`, `summary.json`, and per-test `evidence/` and `verdicts/`. A run stopped before a stage completes may not contain that stage's files. Review the report alongside its recorded evidence. Per-test verdict files preserve the model's assessment; round summaries/report entries can also carry a separate `controller_assessment` and repair feedback preserves `model_assessment`. Do not conflate these when scoring verifier accuracy.

The final report identifies PASS, FAIL, NEEDS_USER, INCOMPLETE, or ERROR and records its reason. For a build run, PASS applies to the final version's frozen tests. For requirements-only, PASS records acceptance of all source blocks. A budget limit produces INCOMPLETE; unresolved failures at the repair limit produce FAIL.

To use the recommended function-mode Python calculator directly, replace `REPLACE_WITH_RUN_ID` with the folder named in the report:

```bash
python run_python_app.py --function main "results/REPLACE_WITH_RUN_ID/app/main.py" "2 + 3 * 4"
```

For a run using the default ordinary-script mode, invoke the source directly with `python "results/REPLACE_WITH_RUN_ID/app/main.py" "2 + 3 * 4"`. Check `python_entry_function` in the manifest to identify the invocation mode.

For `--output` or `--entrypoint` overrides, adjust that path. The `app/` directory contains the most recently written version, which can still have recorded failures. For an exact archived version, use its `rounds/NNN/source/` entrypoint and the run's saved launcher when function mode applies.

Useful options:

| Option | Default | Purpose |
| --- | --- | --- |
| `--output` | `results` | Parent directory for unique run folders |
| `--language` | `python` | Program language: `python` or `c` |
| `--requirements-only` | Off | Review and finalize requirements using the requirements and verification endpoints |
| `--review-requirements` | Off for builds | Enable requirements review before designing tests and building |
| `--calculator-basic-reference` | Off | Compute expectations for the original integer-operand calculator prompt |
| `--calculator-reference` | Off | Enable the full Python calculator mathematical reference profile |
| `--entrypoint` | `main.py` / `main.c` | Main source file for the selected language |
| `--python-entry-function` | None | Invoke the named zero-argument Python function through the fixed launcher |
| `--seed-dir` | None | Existing source directory to copy into the run |
| `--max-repairs` | `3` | Repair attempts after the initial build |
| `--max-turns` | `32` | Tool turns allowed per agent stage |
| `--max-model-calls` | `200` | Run-wide model-call limit |
| `--max-tokens` | `1024` | Maximum tokens per model response |
| `--timeout` | `5` | Seconds allowed per program execution |
| `--http-timeout` | `180` | Seconds allowed per model request |
| `--endpoints` | Ports `8001`-`8004` | Four chat-completion endpoints in table order |

The host counts input tokens against each role's serving context and reduces the response budget below `--max-tokens` when needed, keeping a 32-token margin. If fewer than 128 response tokens remain, the run ends INCOMPLETE and prints the measured input size and context limit. Increase the role's serving context when the input itself exceeds that limit.

Run `python experiment.py --help` for the current CLI. Record the exact model checkpoint/revision, weight and cache quantization, context limit, prompt, and repair limits when comparing experiments.

## C rideshare experiment

The C adapter compiles C source with `gcc` and links `libm`, then supplies the saved command-line arguments and stdin to each execution. Use `--language c`; the default entrypoint becomes `main.c`. An existing project can be copied with `--seed-dir` and its main file selected with `--entrypoint`.

C test `args` must contain actual process arguments, or an empty list; interactive responses belong in `stdin` in their documented order. Expected stdout needs a documented console transcript. Include typed-input echo only if the application itself prints it. Missing prompt wording or input order requires clarification. The C interface supplies no calculator-specific result or error conventions.

`rideshare_experiment_prompt.txt` defines the administrator and rider flows, fare calculation, optional ratings, reporting, and the deterministic `--test-minutes` fixture for the C experiment. For a build from the supplied code, use `--seed-dir rideshare_v1 --entrypoint rideshare.c`. Record or clarify the expected console transcripts before freezing tests. Treat report-file contents as unverified unless the evidence includes the files and the expected contents.

### Automatic research reporting

Every completed pipeline invocation now writes `metrics.html` (offline dashboard),
`metrics.json`, `metrics.csv`, and `research_summary.json` inside its result directory.
This includes PASS, FAIL, ERROR, INCOMPLETE, NEEDS_USER, and handled Ctrl+C exits.
No additional model calls are made. Forced process termination cannot run cleanup;
use the offline command below afterward. Reporting errors are recorded separately
in `metrics_error.json` and never change the pipeline outcome.

The dashboard shows each agent's calls, truncations, rejected actions, reported
input/output tokens, and request latency, plus build success and frozen-test
agreement. The **Prompt processing and feedback** section shows the share of
excerpts processed, requirements and test plan review counts, and the share of
blocks or plans accepted after revision. The excerpt count measures intake
progress. Independent requirement labels measure semantic coverage.

Summary success rates include failed and incomplete runs within each scope.
The dashboard groups runs with the same source of agent responses and shows that source at the top.
Requirements-only runs and end-to-end runs have separate success rates and
are grouped by language and entrypoint. The JSON records usage coverage so
reported token totals can be interpreted when some calls lack usage data.

Prepare researcher annotations independently of the agents to measure
requirements precision, recall, and F1. Label frozen tests for validity and
execution records for correctness to measure final-code pass rate and verifier
failure-detection precision, recall, F1, and false-pass rate. Label overall
task success separately. Missing metrics display
N/A. Verifier FAIL is the positive class; INCONCLUSIVE verdicts are reported as
abstentions and excluded from precision/recall/F1. False-pass rate is incorrect
executions called PASS divided by independently labeled incorrect executions with
verdicts. Label coverage and raw confusion counts are preserved.

Copy `evaluation_labels.template.json` to a separate file. Replace null test and
execution labels with booleans, or remove unreviewed entries. Supply all extracted
requirement IDs plus reference IDs to enable requirement scores. Requirement
matches must be one-to-one; label unsupported or duplicate extractions null.
Never use an agent's own assessment as the independent annotation.

Example annotation structure (replace example IDs with those from your run):

```json
{
  "reference_requirements": ["G001", "G002"],
  "requirement_matches": {"R001": "G001", "R002": null},
  "test_validity": {"R001-T01": true},
  "execution_correctness": {"actual-execution-id": false},
  "task_success": false,
  "mutations": [{"id": "swap-plus-minus", "eligible": true, "killed": true}]
}
```

Mutation score uses supplied mutation outcomes; this reporter does not introduce
or execute mutations. Mark equivalent/invalid mutants ineligible; uncovered valid
mutants remain eligible survivors. Repair success currently uses pipeline outcomes
and is labeled as such, rather than claiming independent correctness.

Optional pipeline flag: `--evaluation-labels /absolute/path/labels.json`.
More commonly, annotate after the run and regenerate the report offline:

```bash
python research_metrics.py /home/weibelj/git/machine_learning/nlp-mas/results/RUN_ID --labels /absolute/path/labels.json
```

Omit `--labels` to report an existing failed/interrupted run without annotations.
Reports include a snapshot and SHA-256 of the reporting code. The aggregate is a
snapshot of sibling directories containing `report.json`, not a curated benchmark;
keep separate result directories for distinct research conditions or exclude pilot
runs when preparing the final study.
