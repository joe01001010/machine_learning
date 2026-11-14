#!/usr/bin/env python3
"""Four-agent, evidence-backed build/test/repair experiments."""
import argparse
import ast
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid

from jsonschema import validate, ValidationError

ROLES = ('requirements-engineer', 'test-engineer', 'coder', 'verification-engineer')
BASE = Path(__file__).resolve().parent


def now():
    return datetime.now(timezone.utc).isoformat()


def obj(properties):
    return {'type': 'object', 'properties': properties,
            'required': list(properties), 'additionalProperties': False}


def text_schema(**kwargs):
    return {'type': 'string', **kwargs}


def output_text(value):
    """Ignore one terminal line ending, preserving every other character."""
    return value[:-2] if value.endswith('\r\n') else value[:-1] if value.endswith('\n') else value


def tool(name, description, properties, handler):
    return ({'type': 'function', 'function': {'name': name, 'description': description,
             'parameters': obj(properties), 'strict': True}}, handler)


class NeedsUser(Exception):
    pass


class BudgetExceeded(Exception):
    pass


@dataclass
class Config:
    prompt: str
    output: Path
    language: str = 'python'
    entrypoint: str | None = None
    seed_dir: Path | None = None
    max_repairs: int = 3
    max_turns: int = 32
    max_model_calls: int = 200
    max_tokens: int = 1024
    timeout: float = 5
    http_timeout: float = 180
    calculator_reference: bool = False
    python_entry_function: str | None = None
    evaluation_labels: Path | None = None
    endpoints: list = field(default_factory=lambda: [
        f'http://127.0.0.1:{p}/v1/chat/completions' for p in range(8001, 8005)])


class Experiment:
    def __init__(self, config, transport=None):
        self.config = config
        if not config.prompt.strip() or len(config.prompt) > 20000:
            raise ValueError('Provide a nonempty prompt of at most 20000 characters')

        if config.language not in ('python', 'c') or len(config.endpoints) != 4:
            raise ValueError('Choose python or c and provide four model endpoints')

        for name in ('max_turns', 'max_model_calls', 'max_tokens', 'timeout', 'http_timeout'):
            if getattr(config, name) <= 0:
                raise ValueError(f'{name} must be positive')

        if config.max_repairs < 0:
            raise ValueError('max_repairs cannot be negative')

        if config.python_entry_function and (config.language != 'python' or not config.python_entry_function.isidentifier()):
            raise ValueError('python-entry-function requires Python and a valid function name')

        self.entrypoint = config.entrypoint or ('main.py' if config.language == 'python' else 'main.c')
        self.extensions = {'.py'} if config.language == 'python' else {'.c', '.h'}
        self.run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ-') + uuid.uuid4().hex[:8]
        self.root = Path(config.output).resolve() / self.run_id
        self.app = self.root / 'app'
        self.app.mkdir(parents=True, exist_ok=False)
        self.source_path(self.entrypoint)
        self.transport = transport or self.http
        self.use_tokenize = False
        self.reference = None
        self.oracle_sha256 = None
        self._compact_test_io = False

        if config.calculator_reference:
            if config.language != 'python':
                raise ValueError('The calculator reference profile is for the Python calculator experiment')

            from calculator_oracle import reference
            self.reference = reference
            oracle = (BASE / 'calculator_oracle.py').read_bytes()
            (self.root / 'calculator_oracle.py').write_bytes(oracle)
            self.oracle_sha256 = hashlib.sha256(oracle).hexdigest()

        self.launcher = None
        self.launcher_sha256 = None

        if config.python_entry_function:
            self.launcher = self.root / 'run_python_app.py'
            harness = (BASE / 'run_python_app.py').read_bytes()
            self.launcher.write_bytes(harness)
            self.launcher_sha256 = hashlib.sha256(harness).hexdigest()

        self.model_calls = 0
        self.requirements = []
        self.tests = []
        self.report = {'run_id': self.run_id, 'started_at': now(), 'status': 'INCOMPLETE',
                       'rounds': [], 'questions': [], 'run_directory': str(self.root)}
        controller = Path(__file__).read_bytes()
        (self.root / 'controller_source.py').write_bytes(controller)
        self.save('customer_prompt.json', {'text': config.prompt, 'source': 'user'})
        self.save('manifest.json', {'run_id': self.run_id, 'roles': list(ROLES),
            'controller_sha256': hashlib.sha256(controller).hexdigest(), 'python': sys.version,
            'endpoints': config.endpoints, 'language': config.language, 'entrypoint': self.entrypoint,
            'max_repairs': config.max_repairs, 'max_turns': config.max_turns, 'max_model_calls': config.max_model_calls,
            'max_tokens': config.max_tokens, 'temperature': 0,
            'calculator_reference': config.calculator_reference,
            'python_entry_function': config.python_entry_function,
            'python_launcher_sha256': self.launcher_sha256,
            'calculator_oracle_sha256': self.oracle_sha256,
            'interface': self.interface(), 'seed_dir': str(config.seed_dir) if config.seed_dir else None})

        if config.seed_dir:
            source = Path(config.seed_dir).resolve(strict=True)
            if not source.is_dir() or source == self.app or self.root.is_relative_to(source):
                raise ValueError('seed-dir must be a separate source directory')

            for path in sorted(source.rglob('*')):
                if path.is_file() and not path.is_symlink() and path.suffix in self.extensions:
                    rel = path.relative_to(source)
                    target = self.source_path(str(rel))
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(path, target)

            self.save('seed_manifest.json', {'source': str(source), 'source_sha256': self.hashes(self.app)})

    def interface(self):
        interface = {'language': self.config.language, 'entrypoint': self.entrypoint,
            'invocation': (f'The fixed launcher imports {self.entrypoint} and calls {self.config.python_entry_function}() once; sys.argv contains the entrypoint and saved test arguments.'
                           if self.config.python_entry_function else f'python {self.entrypoint} <expression>') if self.config.language == 'python'
                          else 'compiled program with argument list and/or scripted stdin',
            'dependencies': 'Python standard library only' if self.config.language == 'python'
                            else 'C standard library and libm; compiled with gcc -std=c11',
            'comparison': 'stdout compared exactly after removing one terminal LF or CRLF from each string; preserve all other whitespace. Exit code compared exactly.',
            'calculator_output_convention': 'Plain decimal notation; omit unnecessary leading/trailing zeros and decimal point for integers; append one newline. Use round-half-even when a significant-digit limit applies.',
            'invalid_input_convention': 'For a requested generic error response use Error: invalid input\n and exit 1, unless the customer specifies otherwise.'}

        if self.config.language == 'c':
            interface.pop('calculator_output_convention')
            interface.pop('invalid_input_convention')
            interface['test_input_convention'] = 'args contains actual process arguments only, such as ["--test-minutes", "12"], or [] for none. Put interactive responses in stdin in the documented order, one response per input line; never put scenario descriptions in args.'
            interface['console_output_convention'] = 'Expected stdout must use a documented console transcript. Do not include terminal echo of typed input unless the application explicitly prints it. Ask for missing console wording or input order before inventing expected output.'
        return interface

    def save(self, name, data):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8') as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2)
        return str(path.relative_to(self.root))

    def trace(self, data):
        with (self.root / 'trace.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'time': now(), **data}, ensure_ascii=False) + '\n')

    def source_path(self, name):
        rel = Path(name)
        if rel.is_absolute() or '..' in rel.parts or '\\' in name or rel.suffix not in self.extensions:
            raise ValueError('Use a relative application source path, without parent traversal')
        path = self.app / rel
        if not path.resolve().is_relative_to(self.app.resolve()):
            raise ValueError('Source path escapes the application directory')
        cursor = path
        while cursor != self.app:
            if cursor.is_symlink():
                raise ValueError('Source symlinks are not allowed')
            cursor = cursor.parent
        return path

    def hashes(self, directory):
        return {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(directory.rglob('*')) if p.is_file() and p.suffix in self.extensions}

    def write_file(self, path, content, mode):
        target = self.source_path(path)
        if isinstance(content, list):
            if not all(isinstance(line, str) and '\n' not in line and '\r' not in line for line in content):
                raise ValueError('Each array item must contain exactly one source line; retain its leading indentation')
            content = '\n'.join(content) + ('\n' if content else '')
        if mode not in ('replace', 'append'):
            raise ValueError('Choose replace or append')
        if len(content.encode()) > 24000:
            raise ValueError('Write at most 24000 bytes per call; use append for longer files')
        if target.exists() and ((mode == 'replace' and target.read_text(encoding='utf-8') == content)
                                or (mode == 'append' and not content)):
            raise ValueError('This write makes no source change. Implement the next missing behavior or finish_build.')
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('w' if mode == 'replace' else 'a', encoding='utf-8') as stream:
            stream.write(content)
        return {'path': path, 'characters': target.stat().st_size,
                'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                'source_tail': target.read_text(encoding='utf-8')[-1200:]}

    def read_file(self, path, offset):
        content = self.source_path(path).read_text(encoding='utf-8')
        if offset < 0 or offset > len(content):
            raise ValueError('Offset is outside the file')
        end = min(offset + 3000, len(content))
        return {'path': path, 'content': content[offset:end],
                'next_offset': end if end < len(content) else None}

    def ask_user(self, question, reason):
        self.report['questions'].append({'question': question, 'reason': reason})
        raise NeedsUser(question)

    def reference_tools(self):
        if self.reference is None:
            return []
        return [tool('reference_math', 'Compute expected calculator output independently of the generated program. Use this for arithmetic, recurring decimals, and 17-digit fractional powers.',
            {'expression': text_schema(minLength=1, maxLength=500)}, self.reference)]

    def http(self, role, body):
        endpoint = self.config.endpoints[ROLES.index(role)]
        if self.use_tokenize:
            tokenize = urllib.request.Request(endpoint.rsplit('/v1/', 1)[0] + '/tokenize',
                data=json.dumps({k: body[k] for k in ('model', 'messages', 'tools')}).encode(),
                headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(tokenize, timeout=15) as response:
                budget = json.load(response)
            available = budget['max_model_len'] - budget['count'] - 32
            if available < 128:
                raise BudgetExceeded(f'{role}: input leaves too little model context; shorten the prompt or source pages')
            body['max_tokens'] = min(body['max_tokens'], available)
        request = urllib.request.Request(self.config.endpoints[ROLES.index(role)],
            data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(request, timeout=self.config.http_timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f'{role}: HTTP {exc.code}: {exc.read().decode(errors="replace")}') from exc

    def agent(self, role, mode, context, available, done, allow_questions=True, on_truncation=None):
        last_result = None
        action_history = []
        reference_calls = 0
        consecutive_truncations = 0
        common = tool('ask_user', 'Stop for missing user intent; do not guess.',
                      {'question': text_schema(minLength=1, maxLength=1000),
                       'reason': text_schema(minLength=1, maxLength=1000)}, self.ask_user)
        for turn in range(self.config.max_turns):
            if self.model_calls >= self.config.max_model_calls:
                raise BudgetExceeded('Run-wide model call limit reached')
            current = available() + ([common] if allow_questions else [])
            if reference_calls >= 3:
                current = [item for item in current if item[0]['function']['name'] != 'reference_math']
            handlers = {definition['function']['name']: (definition, handler)
                        for definition, handler in current}
            state = {'mode': mode, **context(), 'step': turn + 1,
                     'recent_actions': action_history[-6:], 'previous_action': last_result}
            # Intake must see customer intent alone, so implementation policies
            # cannot be mistaken for additional customer requirements or quotes.
            if mode != 'requirements':
                state['interface'] = self.interface()
            body = {'model': role, 'messages': [{'role': 'user', 'content': json.dumps(state)}],
                    'tools': [definition for definition, _ in current], 'tool_choice': 'required',
                    'parallel_tool_calls': False, 'temperature': 0, 'max_tokens': self.config.max_tokens}
            self.model_calls += 1
            call_started = time.perf_counter()
            try:
                response = self.transport(role, body)
            except (Exception, KeyboardInterrupt) as exc:
                self.trace({'role': role, 'mode': mode, 'transport_error': type(exc).__name__,
                    'duration_seconds': time.perf_counter() - call_started})
                raise
            self.trace({'role': role, 'mode': mode, 'request': body, 'response': response,
                'duration_seconds': time.perf_counter() - call_started})
            choice = response['choices'][0]
            if choice.get('finish_reason') == 'length':
                consecutive_truncations += 1
                if on_truncation:
                    on_truncation()
                instructions = {
                    'requirements': 'Call submit_requirements with only ONE new requirement, brief text and a supporting source_id for the current excerpt. Continue from next_requirement_id and set complete=true; the controller advances to later excerpts automatically.',
                    'coding': 'Use smaller source chunks (under 2000 characters), then append. Set complete=false until the final chunk.',
                    'test_design': 'Use the currently offered test-drafting tool. Submit only the requested metadata or exact next lines; keep individual lines short without changing the intended input or expected values.',
                    'test_review': 'Use the offered review tool with a short decision and feedback.',
                    'verification': 'Use the offered verdict tool with a short explanation.',
                }
                last_result = {'error': 'Response exceeded output token limit; no action was executed.',
                    'instruction': instructions.get(mode, 'Use the offered tool with a shorter valid submission.')}
                if consecutive_truncations >= 3:
                    raise BudgetExceeded(f'{role}: {mode} response truncated 3 times without a completed action '
                                         f'(output budget {body["max_tokens"]} tokens); use a smaller submission or increase --max-tokens within the model context limit')
                continue
            consecutive_truncations = 0
            message = choice['message']
            calls = message.get('tool_calls') or []
            if len(calls) != 1:
                raise RuntimeError(f'{role} must return exactly one offered tool call')
            call = calls[0]
            arguments = {}
            try:
                name = call['function']['name']
                if name not in handlers:
                    raise ValueError('Tool is not available in this stage')
                definition, handler = handlers[name]
                arguments = json.loads(call['function']['arguments'])
                validate(arguments, definition['function']['parameters'])
                if name == 'reference_math':
                    reference_calls += 1
                result = handler(**arguments)
            except (ValueError, ValidationError, KeyError, FileNotFoundError) as exc:
                result = {'error': str(exc)[:1000], 'instruction': 'Correct the rejected arguments.'}
            self.trace({'role': role, 'mode': mode, 'tool': call['function']['name'], 'result': result})
            if done():
                return
            # Exact exchanges are archived above; resend compact host-owned state so
            # writing a large source file does not consume the next request's context.
            last_result = {'tool': call['function']['name'], 'result': result}
            action_history.append({'tool': call['function']['name'],
                'arguments': {k: str(v)[:120] for k, v in arguments.items()
                              if k != 'content' and not isinstance(v, (list, dict))} if isinstance(arguments, dict) else {},
                'outcome': result.get('error', 'completed')})
        raise BudgetExceeded(f'{role}: {mode} turn limit reached')

    def intake(self):
        staged = []
        finished = False
        batch_limit = 3
        batch_counter = 0
        source_cursor = 0
        source_block_size = 3
        # Use short IDs on the model-facing side. Materialize quotations from
        # the original prompt, avoiding fragile copying of Unicode/punctuation.
        sources = {}
        for line in self.config.prompt.splitlines():
            remaining = line.strip()
            while remaining:
                end = len(remaining)
                if end > 200:
                    end = remaining.rfind(' ', 0, 201)
                    if end < 1:
                        end = 200
                fragment = remaining[:end]
                sources[f'S{len(sources) + 1:03d}'] = fragment
                remaining = remaining[end:].lstrip()
        source_order = list(sources)
        numbered_prompt = '\n'.join(f'[{key}] {value}' for key, value in sources.items())
        self.save('requirements_sources.json', sources)

        def current_source_ids():
            return source_order[source_cursor:source_cursor + source_block_size]

        def total_source_blocks():
            remaining = max(0, len(source_order) - source_cursor)
            return batch_counter + (remaining + source_block_size - 1) // source_block_size

        def normalized_text(value):
            return ' '.join(value.split())

        def submit_requirements(requirements, complete):
            nonlocal finished, batch_counter, source_cursor
            block_ids = current_source_ids()
            if complete is not True:
                raise ValueError('complete must be true: each valid submission covers and automatically advances the CURRENT source block. Group its facts into at most batch_limit requirements, or use requirements=[] for headings/already-covered facts')
            if any(r['source_id'] not in block_ids for r in requirements):
                raise ValueError(f'Extract only the CURRENT source block: source_id must be one of {block_ids}')
            ids = [r['requirement_id'] for r in requirements]
            if len(ids) != len(set(ids)):
                raise ValueError('Requirement IDs must be unique within this batch')
            expected = [f'R{len(staged) + i + 1:03d}' for i in range(len(requirements))]
            if ids != expected:
                raise ValueError(f'Continue after the accepted batches: use IDs {expected} in order; do not repeat accepted requirements')
            known = {normalized_text(r['text']): r['requirement_id'] for r in staged}
            normalized, duplicates, assignments = [], [], []
            for candidate in requirements:
                text = normalized_text(candidate['text'])
                if text in known:
                    duplicates.append({'submitted_requirement_id': candidate['requirement_id'],
                        'canonical_requirement_id': known[text], 'source_id': candidate['source_id'],
                        'source_quote': sources[candidate['source_id']]})
                    continue
                canonical_id = f'R{len(staged) + len(normalized) + 1:03d}'
                normalized.append({**candidate, 'requirement_id': canonical_id,
                                   'source_quote': sources[candidate['source_id']]})
                assignments.append({'submitted_requirement_id': candidate['requirement_id'],
                                    'canonical_requirement_id': canonical_id})
                known[text] = canonical_id
            if len(staged) + len(normalized) > 64:
                raise ValueError('Intake supports at most 64 requirements in one experiment')
            accepted_ids = [r['requirement_id'] for r in normalized]
            combined = staged + normalized
            next_cursor = source_cursor + len(block_ids)
            intake_complete = next_cursor >= len(source_order)
            if intake_complete and not combined:
                raise ValueError('At least one grounded requirement is required before finishing intake')
            # Validate the whole batch before changing staged state. Truncated
            # responses never reach this handler and cannot partially commit.
            self.save(f'requirements_batches/batch-{batch_counter + 1:03d}.json',
                      {'requirements': normalized, 'source_ids': block_ids,
                       'submitted_requirements': requirements, 'duplicates': duplicates,
                       'id_assignments': assignments,
                       'source_block_index': batch_counter + 1,
                       'complete': complete, 'block_complete': complete,
                       'intake_complete': intake_complete,
                       'accepted_ids': [r['requirement_id'] for r in combined]})
            if intake_complete:
                self.save('requirements.json', combined)
                self.requirements = combined
            staged.extend(normalized)
            batch_counter += 1
            source_cursor = next_cursor
            finished = intake_complete
            self.report['requirements_staged_count'] = len(staged)
            self.report['requirements_source_blocks_completed'] = batch_counter
            self.report['requirements_source_excerpts_completed'] = min(source_cursor, len(source_order))
            action = f'Accepted {", ".join(accepted_ids)} ({len(staged)} total)' if accepted_ids else 'Skipped already-covered or heading-only source block'
            if duplicates:
                action += f'; reused {len(duplicates)} existing requirement(s)'
            progress = '; requirements finalized.' if intake_complete else '; advancing to the next source block.'
            print(f'  {action}{progress}', flush=True)
            return {'accepted': accepted_ids, 'total_accepted': len(staged), 'duplicates': duplicates,
                    'source_ids': block_ids, 'complete': complete,
                    'block_complete': complete, 'intake_complete': intake_complete,
                    'next_source_ids': current_source_ids(),
                    'next_requirement_id': None if finished else f'R{len(staged) + 1:03d}'}

        def available():
            schema = obj({'requirement_id': text_schema(enum=[
                f'R{len(staged) + i + 1:03d}' for i in range(batch_limit)]),
                'text': text_schema(minLength=1, maxLength=300),
                'source_id': text_schema(enum=current_source_ids(),
                    description='ID from the CURRENT source block supporting this requirement. The host saves its exact source_quote.')})
            return [tool('submit_requirements',
                f'Cover the entire CURRENT source block in at most {batch_limit} NEW requirements. Every valid submission automatically advances to the next block. complete must be true; it acknowledges only this block, not the full prompt. Use an empty batch for headings or already-covered facts.',
                {'requirements': {'type': 'array', 'minItems': 0, 'maxItems': batch_limit, 'items': schema},
                 'complete': {'type': 'boolean', 'enum': [True]}}, submit_requirements)]

        def shrink_batch():
            nonlocal batch_limit, source_block_size
            batch_limit = 1
            source_block_size = 1

        def context():
            block_ids = current_source_ids()
            return {'full_prompt_context': numbered_prompt,
                'customer_prompt': '\n'.join(f'[{key}] {sources[key]}' for key in block_ids),
                'source_ids': block_ids,
                'source_block': {'index': batch_counter + 1,
                                 'total': total_source_blocks(),
                                 'sources': [{'source_id': key, 'text': sources[key]} for key in block_ids]},
                'accepted_requirements': [{'requirement_id': r['requirement_id'], 'text': r['text'][:80]} for r in staged],
                'next_requirement_id': f'R{len(staged) + 1:03d}', 'batch_limit': batch_limit,
                'instruction': 'Extract requirements ONLY from customer_prompt, which is the CURRENT source block. Cover ALL its requested behavior in ONE compact submission of at most batch_limit requirements; group related facts. Every valid submission automatically advances the host. complete MUST be true: it acknowledges this block, not the full prompt. The host freezes intake only after every source excerpt has been processed. Use requirements=[] with complete=true for headings or facts already covered. Use contiguous IDs starting at next_requirement_id and current source_ids; the host copies exact quotations. Existing facts are reused automatically, not added twice. full_prompt_context is only background for clarifications. Later explicit clarifications resolve ambiguity. Do not add tool policies as customer requirements.'}

        print('Interpreting your prompt...', flush=True)
        self.agent(ROLES[0], 'requirements', context,
            available, lambda: finished, on_truncation=shrink_batch)

    def design_tests(self):
        for requirement in self.requirements:
            invalid_requirement = bool(self.reference and re.search(
                r'\binvalid\b|\berror (?:response|message|handling)\b', requirement['text'].lower()))
            schema = obj({'test_case_id': text_schema(pattern='^'+requirement['requirement_id']+r'-T0[1-3]$'),
                'args': {'type': 'array', 'maxItems': 20, 'items': text_schema(maxLength=2000)},
                'stdin': text_schema(maxLength=6000), 'expected_stdout': text_schema(maxLength=6000,
                    description='Exact plain-text result; one trailing newline is optional and ignored during comparison.'),
                'expected_exit_code': {'type': 'integer', 'minimum': 0, 'maximum': 255},
                'justification': text_schema(minLength=1, maxLength=700)})
            if invalid_requirement:
                schema['properties']['expected_exit_code'] = {'type': 'integer', 'enum': [1],
                    'description': 'This requirement is about invalid input; propose error-path tests only.'}
                schema['properties']['args']['description'] = 'Supply a malformed arithmetic expression, not a valid calculation; e.g. letters or an incomplete operator.'
            feedback = None
            for attempt in range(3):
                proposed, review, reference_checks = [], {}, []
                draft = None
                def submit_tests(tests, rationale):
                    validate(tests, {'type': 'array', 'minItems': 1, 'maxItems': 3, 'items': schema})
                    if len({t['test_case_id'] for t in tests}) != len(tests):
                        raise ValueError('Test IDs must be unique')
                    if self.reference:
                        if invalid_requirement and not any(t['expected_exit_code'] == 1 for t in tests):
                            raise ValueError('This requirement concerns invalid input: include at least one invalid-input test with the declared error response and exit 1')
                        checks = []
                        for test in tests:
                            expected = (self.reference(test['args'][0]) if len(test['args']) == 1
                                else {'expected_stdout': 'Error: invalid input\n', 'expected_exit_code': 1, 'kind': 'error'})
                            check = {'test_case_id': test['test_case_id'], 'args': test['args'],
                                'proposed_stdout': test['expected_stdout'], 'reference': expected,
                                'matches': output_text(test['expected_stdout']) == output_text(expected['expected_stdout'])
                                           and test['expected_exit_code'] == expected['expected_exit_code']}
                            checks.append(check)
                        self.save(f'test_reference/{requirement["requirement_id"]}/{uuid.uuid4().hex}.json', checks)
                        wrong = [check for check in checks if not check['matches']]
                        if wrong:
                            raise ValueError('Expected values contradict the independent mathematical reference: ' + json.dumps(wrong)[:1600])
                        reference_checks.extend(checks)
                    proposed.extend([{**t, 'requirement_id': requirement['requirement_id']} for t in tests])
                    self.save(f'test_design/{requirement["requirement_id"]}/proposal-{attempt}.json',
                              {'tests': proposed, 'rationale': rationale})
                    return {'proposed': [t['test_case_id'] for t in tests]}
                def compact_drafting():
                    nonlocal draft
                    if draft is not None:
                        draft.shrink_chunk()
                        return
                    from test_drafts import TestDraft
                    if not self._compact_test_io:
                        source = (BASE / 'test_drafts.py').read_bytes()
                        (self.root / 'test_drafts.py').write_bytes(source)
                        self.save('test_drafting_protocol.json', {'protocol': 'bounded-lines-v1',
                            'activation': 'test_design output truncation',
                            'sha256': hashlib.sha256(source).hexdigest()})
                    self._compact_test_io = True
                    draft = TestDraft(requirement['requirement_id'], submit_tests, self.save,
                        f'test_drafts/{requirement["requirement_id"]}/attempt-{attempt}')
                if self._compact_test_io:
                    compact_drafting()
                def drafting_context():
                    state = {'customer_prompt': self.config.prompt,
                    'requirement': requirement, 'review_feedback': feedback,
                    'test_kind': 'INVALID_INPUT: exercise malformed expressions and the declared error response, not valid arithmetic' if invalid_requirement else 'behavior described by this requirement',
                    'instruction': 'Design 1-3 independent tests for THIS requirement before seeing code. Check arithmetic carefully. expected_stdout is the exact plain-text result; a single final line ending is optional. Apply independent reviewer feedback if present.'}
                    if draft is not None:
                        state['test_draft'] = draft.context()
                        state['instruction'] += (' Use only the current phase tool. Plan the smallest adequate scenarios; declare exact line counts, then provide only the requested next lines. '
                            'Each array item is a single line without a newline character. Do not repeat a credential or invent console wording. '
                            'Ask the user if exact expected console output is not specified. The host assembles and submits only complete tests for independent review.')
                    return state
                def drafting_tools():
                    return (draft.tools() if draft is not None else [tool('submit_tests', 'Propose tests for independent review before freezing.',
                        {'tests': {'type': 'array', 'minItems': 1, 'maxItems': 3, 'items': schema},
                         'rationale': text_schema(minLength=1, maxLength=700)}, submit_tests)]) + self.reference_tools()
                self.agent(ROLES[1], 'test_design', drafting_context, drafting_tools,
                    lambda: bool(proposed), allow_questions=self.reference is None,
                    on_truncation=compact_drafting)
                # Once tests require chunking, review one complete case at a time
                # rather than putting every transcript back into one model context.
                groups = [[test] for test in proposed] if draft is not None else [proposed]
                decisions = []
                for index, group in enumerate(groups):
                    case_review = {}
                    def review_tests(decision, feedback):
                        case_review.update(decision=decision, feedback=feedback)
                        return case_review
                    self.agent(ROLES[3], 'test_review', lambda: {'customer_prompt': self.config.prompt,
                        'requirement': requirement, 'tests': group,
                        'test_plan_overview': [{k: t[k] for k in ('test_case_id', 'args', 'justification')} for t in proposed],
                        'mathematical_reference_checks': [c for c in reference_checks if c['test_case_id'] in {t['test_case_id'] for t in group}],
                        'instruction': 'Review test design only; no software has run. Check actual coverage of THIS requirement and customer intent plus declared interface. Review the complete cases in tests; the overview supplies the other planned scenarios for coverage. When reference checks are supplied, those computed numeric values take precedence over mental arithmetic; focus on scope/coverage. REVISE unsupported tests with specific feedback; ACCEPT only justified tests.'},
                        lambda: [tool('review_tests', 'Review proposed tests before implementation begins.',
                            {'decision': text_schema(enum=['ACCEPT', 'REVISE']),
                             'feedback': text_schema(minLength=1, maxLength=1000)}, review_tests)] + self.reference_tools(),
                        lambda: bool(case_review), allow_questions=self.reference is None)
                    decisions.append(dict(case_review))
                    if draft is not None:
                        self.save(f'test_design/{requirement["requirement_id"]}/review-{attempt}-case-{index+1}.json', case_review)
                    if case_review['decision'] != 'ACCEPT':
                        break
                review.update(decision='ACCEPT' if len(decisions) == len(groups) and all(r['decision'] == 'ACCEPT' for r in decisions) else 'REVISE',
                    feedback='; '.join(r['feedback'] for r in decisions))
                self.save(f'test_design/{requirement["requirement_id"]}/review-{attempt}.json', review)
                if review['decision'] == 'ACCEPT':
                    self.tests.extend(proposed)
                    break
                feedback = review['feedback']
            else:
                raise BudgetExceeded(f'{requirement["requirement_id"]}: test plan review limit reached')
        self.plan_hash = hashlib.sha256(json.dumps(self.tests, sort_keys=True).encode()).hexdigest()
        self.save('test_plan.json', {'tests': self.tests, 'sha256': self.plan_hash})

    def coding(self, number, feedback):
        finished = []
        read_versions = set()
        consumed_feedback = {0}
        def read_source(path, offset):
            source = self.source_path(path)
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            key = (path, digest, offset)
            if key in read_versions:
                raise ValueError('This source page was already read and has not changed. Write the next source chunk or finish_build; do not reread it.')
            if offset % 3000:
                raise ValueError('Start at offset 0 and then follow next_offset (3000 characters per page)')
            result = self.read_file(path, offset)
            read_versions.add(key)
            return result
        def finish_build(summary):
            if not self.source_path(self.entrypoint).is_file():
                raise ValueError(f'Create the entrypoint {self.entrypoint} before finishing')
            finished.append(summary)
            return {'ready': True}
        def write_and_handoff(path, content, mode, complete):
            result = self.write_file(path, content, mode)
            if complete:
                finish_build(f'Coder submitted {path} and requested build/test execution')
                result['submitted_for_testing'] = True
            return result
        print(f'Building version {number}...' if number == 0 else f'Repairing version {number}...', flush=True)
        def read_feedback(offset):
            if offset < 0 or offset > len(feedback):
                raise ValueError('Invalid feedback offset')
            if offset in consumed_feedback:
                raise ValueError('This feedback page was already provided; use the recorded failure to repair the source')
            consumed_feedback.add(offset)
            return {'items': feedback[offset:offset+1],
                    'next_offset': offset+1 if offset+1 < len(feedback) else None}
        def available():
            items = [tool('write_file', 'Write source as an ARRAY OF LINES preserving indentation. Set complete=true on the final source chunk to submit the program for build and testing immediately; false if more files/chunks remain.',
                    {'path': text_schema(maxLength=200),
                     'content': {'type': 'array', 'minItems': 1, 'maxItems': 80,
                                 'items': text_schema(maxLength=1000)},
                     'mode': text_schema(enum=['replace', 'append']),
                     'complete': {'type': 'boolean'}}, write_and_handoff),
                tool('finish_build', 'Hand this code version to the test engineer.',
                    {'summary': text_schema(minLength=1, maxLength=800)}, finish_build)]
            hashes = self.hashes(self.app)
            remaining = [offset for name, digest in hashes.items()
                for offset in range(0, max(1, len(self.source_path(name).read_text(encoding='utf-8'))), 3000)
                if (name, digest, offset) not in read_versions]
            if remaining:
                items.insert(0, tool('read_file', 'Read an unread application source page; do not repeat unchanged pages.',
                    {'path': text_schema(maxLength=200), 'offset': {'type': 'integer', 'enum': sorted(set(remaining))}}, read_source))
            if len(consumed_feedback) < len(feedback):
                items.insert(0, tool('read_feedback', 'Read remaining failure pages; page 0 is already in context.',
                    {'offset': {'type': 'integer', 'enum': [i for i in range(len(feedback)) if i not in consumed_feedback]}},
                    read_feedback))
            return items
        self.agent(ROLES[2], 'coding', lambda: {'round': number, 'customer_prompt': self.config.prompt,
            'requirements': [{'requirement_id': r['requirement_id'], 'text': r['text']} for r in self.requirements],
            'files': list(self.hashes(self.app)), 'feedback': feedback[:1], 'feedback_count': len(feedback),
            'instruction': ('Build from scratch on round 0 unless seed files exist; repair failures later. '
                + (f'Define {self.config.python_entry_function}() with no parameters; the fixed launcher calls it with sys.argv set. Return an integer status or use sys.exit. Do not call it unconditionally at module scope. '
                   if self.config.python_entry_function else 'The entrypoint must execute its logic when run: call main() if you define a Python main function. ')
                + 'Write an array of source lines preserving indentation. Use complete=true on the LAST write to submit for build/test; false only when more chunks remain. Use append for later chunks. finish_build is an alternative handoff.')},
            available, lambda: bool(finished))
        return finished[0]

    def process(self, command, cwd, stdin='', timeout=None):
        started = time.perf_counter()
        result = {'command': command, 'started_at': now(), 'return_code': None,
                  'stdout': '', 'stderr': '', 'execution_status': 'LAUNCH_ERROR', 'output_truncated': False}
        env = {'PATH': os.environ.get('PATH', '/usr/bin:/bin'), 'LANG': 'C.UTF-8',
               'PYTHONIOENCODING': 'utf-8', 'PYTHONDONTWRITEBYTECODE': '1'}
        # Generated programs get a disposable writable directory, a read-only
        # host filesystem, no network, and their own process namespace on WSL.
        isolation = shutil.which('bwrap') if os.name == 'posix' else None
        actual_command = command
        if isolation:
            actual_command = [isolation, '--unshare-pid', '--unshare-net', '--die-with-parent',
                '--ro-bind', '/', '/', '--tmpfs', '/tmp', '--bind', str(cwd), str(cwd),
                '--proc', '/proc', '--dev', '/dev', '--chdir', str(cwd), '--', *command]
        result['isolation'] = 'bwrap' if isolation else 'process_limits_only'
        timeout = timeout or self.config.timeout
        def limits():
            import resource
            resource.setrlimit(resource.RLIMIT_CPU, (math.ceil(timeout)+1, math.ceil(timeout)+2))
            resource.setrlimit(resource.RLIMIT_FSIZE, (1048576, 1048576))
            resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
        with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err, tempfile.TemporaryFile() as inp:
            inp.write(stdin.encode('utf-8'))
            inp.seek(0)
            child = None
            try:
                child = subprocess.Popen(actual_command, cwd=cwd, env=env, stdin=inp, stdout=out, stderr=err,
                    start_new_session=(os.name == 'posix'), preexec_fn=limits if os.name == 'posix' else None)
                result['return_code'] = child.wait(timeout=timeout)
                result['execution_status'] = 'COMPLETED'
            except subprocess.TimeoutExpired:
                result['execution_status'] = 'TIMEOUT'
            except OSError as exc:
                result['stderr'] = str(exc)
            finally:
                if child is not None:
                    if os.name == 'posix':
                        try:
                            os.killpg(child.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    elif child.poll() is None:
                        child.kill()
                    child.wait()
                    result['return_code'] = child.returncode
                for key, stream in [('stdout', out), ('stderr', err)]:
                    stream.seek(0)
                    data = stream.read(1048577)
                    result['output_truncated'] |= len(data) >= 1048576
                    result[key] += data[:1048576].decode('utf-8', errors='replace')
        result['duration_ms'] = round((time.perf_counter()-started)*1000, 3)
        return result

    def build(self, directory):
        if self.config.language == 'python':
            errors = []
            for p in sorted(directory.rglob('*.py')):
                try:
                    source = p.read_text(encoding='utf-8')
                    compile(source, str(p), 'exec')
                    if self.config.python_entry_function and p == directory / self.entrypoint:
                        tree = ast.parse(source)
                        if not any(isinstance(n, ast.FunctionDef) and n.name == self.config.python_entry_function for n in tree.body):
                            errors.append(f'Define a top-level {self.config.python_entry_function}() function for the selected launcher interface')
                except (SyntaxError, ValueError) as exc:
                    errors.append(str(exc))
            return {'success': not errors, 'stdout': '', 'stderr': '\n'.join(errors)}
        command = ['gcc', '-std=c11', '-O0', '-Wall', *[str(p) for p in sorted(directory.rglob('*.c'))],
                   '-lm', '-o', str(directory / 'program')]
        result = self.process(command, directory, timeout=30)
        return {**result, 'success': result['execution_status'] == 'COMPLETED' and result['return_code'] == 0}

    def execute(self, test, snapshot, hashes):
        with tempfile.TemporaryDirectory(prefix='nlp-mas-test-') as temp:
            work = Path(temp) / 'app'
            shutil.copytree(snapshot, work)
            launcher = self.launcher
            if launcher:
                # /tmp is replaced inside the execution namespace. Keep a fresh
                # copy beside this disposable app so temporary output roots work
                # too; the archived launcher remains outside generated source.
                launcher = work / f'.controller_entry_{uuid.uuid4().hex}.py'
                shutil.copyfile(self.launcher, launcher)
                launcher.chmod(0o444)
            command = ([sys.executable, '-B', str(launcher), '--function', self.config.python_entry_function,
                        str(work / self.entrypoint)] if self.config.python_entry_function
                       else [sys.executable, '-B', str(work / self.entrypoint)] if self.config.language == 'python'
                       else [str(work / 'program')]) + test['args']
            record = self.process(command, work, test['stdin'])
        record.update(execution_id=uuid.uuid4().hex, requirement_id=test['requirement_id'],
            test_case_id=test['test_case_id'], args=test['args'], stdin=test['stdin'], source_sha256=hashes)
        record['host_match'] = (record['execution_status'] == 'COMPLETED' and not record['output_truncated']
            and output_text(record['stdout']) == output_text(test['expected_stdout'])
            and record['return_code'] == test['expected_exit_code'])
        return record

    def verify(self, test, execution):
        requirement = next(r for r in self.requirements if r['requirement_id'] == test['requirement_id'])
        def submit_verdict(verdict, criteria_justified, explanation):
            if not criteria_justified and verdict != 'INCONCLUSIVE':
                raise ValueError('Unjustified criteria require INCONCLUSIVE')
            infrastructure = execution['execution_status'] != 'COMPLETED' or execution['output_truncated']
            if infrastructure and verdict != 'INCONCLUSIVE':
                raise ValueError('Incomplete execution evidence requires INCONCLUSIVE')
            if criteria_justified and not infrastructure:
                expected = 'PASS' if execution['host_match'] else 'FAIL'
                if verdict != expected:
                    raise ValueError(f'Exact output and exit-code comparison requires {expected}')
            result = {'verdict': verdict, 'criteria_justified': criteria_justified, 'explanation': explanation}
            decision.update(result)
            return result
        decision = {}
        evidence = {key: execution[key] for key in ('execution_id', 'execution_status', 'return_code',
                                                  'stdout', 'stderr', 'host_match', 'output_truncated')}
        for key in ('stdout', 'stderr'):
            if len(evidence[key]) > 1800:
                evidence[key] = evidence[key][:1800] + '[truncated in model context; full output archived]'
        self.agent(ROLES[3], 'verification', lambda: {'customer_prompt': self.config.prompt,
            'requirement': requirement, 'test': test, 'execution': evidence,
            'mathematical_reference': (self.reference(test['args'][0]) if self.reference and len(test['args']) == 1 else None),
            'instruction': 'This test plan was independently reviewed before coding. Classify this recorded execution: a completed process with empty or wrong stdout is an observed FAILURE, not missing evidence. Criteria justification concerns the requested expectation, not whether the code met it. host_match is a mechanical comparison, not a coverage claim.'},
            lambda: [tool('submit_verdict', 'Assess requirement/test alignment and observed execution.',
                {'verdict': text_schema(enum=['PASS', 'FAIL', 'INCONCLUSIVE']),
                 'criteria_justified': {'type': 'boolean'},
                 'explanation': text_schema(minLength=1, maxLength=900)}, submit_verdict)], lambda: bool(decision),
            allow_questions=not (execution['execution_status'] == 'COMPLETED' and not execution['output_truncated']))
        return decision

    def run(self):
        status, reason = 'INCOMPLETE', ''
        try:
            if self.transport == self.http:
                servers = []
                for role, endpoint in zip(ROLES, self.config.endpoints):
                    url = endpoint.rsplit('/v1/', 1)[0] + '/v1/models'
                    with urllib.request.urlopen(url, timeout=10) as response:
                        models = json.load(response)
                    if role not in [m['id'] for m in models['data']]:
                        raise RuntimeError(f'{role} endpoint serves a different role')
                    self.use_tokenize |= any('max_model_len' in model for model in models['data'])
                    servers.append({'role': role, 'endpoint': endpoint, 'models': models})
                self.save('server_models.json', servers)
            self.intake()
            print('Designing and freezing tests...', flush=True)
            self.design_tests()
            feedback = []
            for number in range(self.config.max_repairs + 1):
                summary = self.coding(number, feedback)
                hashes = self.hashes(self.app)
                folder = self.root / 'rounds' / f'{number:03d}'
                snapshot = folder / 'source'
                shutil.copytree(self.app, snapshot)
                build = self.build(snapshot)
                row = {'round': number, 'summary': summary, 'source_sha256': hashes,
                       'build': build, 'tests': []}
                self.report['rounds'].append(row)
                self.save(f'rounds/{number:03d}/build.json', build)
                feedback = []
                if not build['success']:
                    feedback.append({'kind': 'BUILD_FAILURE', 'stderr': build['stderr'][:3000]})
                else:
                    print(f'Testing version {number}: {len(self.tests)} tests...', flush=True)
                    for test in self.tests:
                        executions = []
                        def run_test(test_case_id):
                            record = self.execute(test, snapshot, hashes)
                            self.save(f'rounds/{number:03d}/evidence/{test_case_id}.json', record)
                            executions.append(record)
                            return {k: record[k] for k in ('execution_id', 'return_code', 'execution_status', 'host_match')}
                        self.agent(ROLES[1], 'test_execution', lambda: {'round': number, 'test': test,
                            'instruction': 'Execute this frozen test using run_test; never replace its input or output.'},
                            lambda: [tool('run_test', 'Run this test and record actual process output.',
                                {'test_case_id': text_schema(enum=[test['test_case_id']])}, run_test)],
                            lambda: bool(executions))
                        result = {'requirement_id': test['requirement_id'], 'test_case_id': test['test_case_id'],
                                  'execution': executions[0], 'verdict': None}
                        row['tests'].append(result)
                        result['verdict'] = self.verify(test, executions[0])
                        self.save(f'rounds/{number:03d}/verdicts/{test["test_case_id"]}.json', result['verdict'])
                        if result['verdict']['verdict'] != 'PASS':
                            assessment = result['verdict']
                            observed_failure = (executions[0]['execution_status'] == 'COMPLETED'
                                and not executions[0]['output_truncated'] and not executions[0]['host_match'])
                            if observed_failure:
                                # Preserve the model's assessment verbatim, but do not
                                # let it hide a demonstrated mismatch as missing evidence.
                                assessment = {'verdict': 'FAIL', 'criteria_justified': True,
                                    'explanation': 'The completed process did not meet the previously frozen test expectation.'}
                            result['controller_assessment'] = assessment
                            feedback.append({'requirement_id': test['requirement_id'],
                                'test_case_id': test['test_case_id'], 'args': test['args'], 'stdin': test['stdin'],
                                'expected_stdout': test['expected_stdout'], 'expected_exit_code': test['expected_exit_code'],
                                'actual_stdout': executions[0]['stdout'][:1200], 'actual_stderr': executions[0]['stderr'][:1200],
                                'actual_exit_code': executions[0]['return_code'], 'assessment': assessment,
                                'model_assessment': result['verdict']})
                        print(f'  {test["test_case_id"]}: {result["verdict"]["verdict"]}', flush=True)
                self.save(f'rounds/{number:03d}/summary.json', row)
                if not feedback:
                    status = 'PASS'
                    reason = 'Every frozen test passed on the final source version; this is not exhaustive verification.'
                    break
                if any(item.get('assessment', {}).get('verdict') == 'INCONCLUSIVE' for item in feedback):
                    status, reason = 'NEEDS_USER', 'Resolve unclear test criteria or incomplete execution evidence.'
                    self.report['questions'].append({'question': reason, 'details': feedback})
                    break
            else:
                status, reason = 'FAIL', 'Repair limit reached; unresolved failures are recorded.'
        except NeedsUser as exc:
            status, reason = 'NEEDS_USER', str(exc)
        except BudgetExceeded as exc:
            status, reason = 'INCOMPLETE', str(exc)
        except KeyboardInterrupt:
            status, reason = 'INCOMPLETE', 'Interrupted by user'
        except Exception as exc:
            status, reason = 'ERROR', f'{type(exc).__name__}: {exc}'
        self.report.update(status=status, reason=reason, finished_at=now(), model_calls=self.model_calls,
                           requirements=self.requirements, test_count=len(self.tests))
        self.save('report.json', self.report)
        # Reporting is offline and must not replace the pipeline's outcome.
        try:
            from research_metrics import generate
            generate(self.root, self.config.evaluation_labels)
            print(f'Research metrics: {self.root / "metrics.html"}', flush=True)
        except Exception as exc:
            self.save('metrics_error.json', {'error': f'{type(exc).__name__}: {exc}'})
            try:
                generate(self.root, ignore_labels=True)
            except Exception:
                pass
            print(f'Research reporting issue: {self.root / "metrics_error.json"}', flush=True)
        print(f'{status}: {reason}\nReport: {self.root / "report.json"}', flush=True)
        return self.report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    prompts = parser.add_mutually_exclusive_group()
    prompts.add_argument('--prompt')
    prompts.add_argument('--prompt-file', type=Path)
    parser.add_argument('--output', type=Path, default=BASE / 'results')
    parser.add_argument('--language', choices=['python', 'c'], default='python')
    parser.add_argument('--entrypoint')
    parser.add_argument('--seed-dir', type=Path)
    parser.add_argument('--max-repairs', type=int, default=3)
    parser.add_argument('--max-turns', type=int, default=32)
    parser.add_argument('--max-model-calls', type=int, default=200)
    parser.add_argument('--max-tokens', type=int, default=1024)
    parser.add_argument('--timeout', type=float, default=5)
    parser.add_argument('--http-timeout', type=float, default=180)
    parser.add_argument('--evaluation-labels', type=Path, help='Independent researcher annotations for quality metrics; optional')
    parser.add_argument('--calculator-reference', action='store_true', help='Check calculator test expectations against an independent mathematical reference')
    parser.add_argument('--python-entry-function', help='Use the fixed launcher to call this generated Python function, e.g. main')
    parser.add_argument('--endpoints', nargs=4, default=Config('', Path('.')).endpoints)
    args = parser.parse_args()
    prompt = args.prompt_file.read_text(encoding='utf-8-sig') if args.prompt_file else args.prompt
    if prompt is None:
        prompt = input('Describe the application you want the agents to build: ')

    options = vars(args)
    options.pop('prompt_file')
    options['prompt'] = prompt

    try:
        report = Experiment(Config(**options)).run()
    except ValueError as exc:
        parser.error(str(exc))
    return 0 if report['status'] == 'PASS' else 2 if report['status'] == 'NEEDS_USER' else 1


if __name__ == '__main__':
    raise SystemExit(main())
