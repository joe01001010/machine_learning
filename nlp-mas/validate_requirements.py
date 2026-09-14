#!/usr/bin/env python
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone

from jsonschema import validate, ValidationError

BASE = Path(__file__).resolve().parent
ROLES = ('requirements-engineer', 'test-engineer', 'verification-engineer')


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def obj(properties):
    return {'type': 'object', 'properties': properties,
            'required': list(properties), 'additionalProperties': False}


def string(**extra):
    return {'type': 'string', **extra}


def tool(name, description, schema, handler):
    return {'type': 'function', 'function': {'name': name, 'description': description,
            'parameters': schema}}, handler


class Pipeline:
    def __init__(self, args, transport=None):
        self.args = args
        self.transport = transport or self.http
        self.calculator = args.calculator.resolve(strict=True)
        self.scope = args.scope.read_text(encoding='utf-8')
        self.requirements = [json.loads(line) for line in
                             args.requirements.read_text(encoding='utf-8').splitlines() if line.strip()]
        ids = []
        for req in self.requirements:
            if not isinstance(req, dict) or not isinstance(req.get('text'), str):
                raise ValueError('Each requirement needs requirement_id and text strings')

            rid = req.get('requirement_id', '')
            if not isinstance(rid, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', rid):
                raise ValueError(f'Invalid requirement ID: {rid!r}')

            ids.append(rid)

        if not ids or len(set(ids)) != len(ids):
            raise ValueError('Requirements must be nonempty and have unique IDs')

        self.run_id = uuid.uuid4().hex[:12]
        self.root = args.output.resolve()
        self.pool = self.root / 'artifacts' / self.run_id
        self.report_path = self.root / f'verification_{self.run_id}.json'
        self.rows = []
        self.artifacts = {}
        self.created = False


    def create_workspace(self):
        if not self.created:
            self.pool.mkdir(parents=True, exist_ok=False)
            self.created = True
            self.save('requirements.json', self.requirements)
            self.save('scope.json', {'text': self.scope})
            hashes = {str(p.relative_to(self.calculator.parent)): digest(p)
                      for p in sorted(self.calculator.parent.rglob('*.py'))}
            self.source_hashes = hashes
            self.save('calculator_source.json', {name: (self.calculator.parent / name).read_text(encoding='utf-8')
                                                 for name in hashes})
            self.save('manifest.json', {'run_id': self.run_id, 'created_at': now(),
                      'calculator_path': str(self.calculator), 'python': sys.version,
                      'source_sha256': hashes, 'models': list(ROLES),
                      'endpoints': self.args.endpoints, 'temperature': self.args.temperature,
                      'max_tokens': self.args.max_tokens,
                      'note': 'Server configuration and exact prompts are archived in .agent-logs.*'})
        return {'workspace': str(self.pool), 'run_id': self.run_id}


    def save(self, name, data):
        # Only host-defined names reach this method; agents cannot choose filesystem paths.
        path = self.pool / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        self.artifacts[name] = path
        return name


    def log(self, event):
        with (self.pool / 'trace.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps({'time': now(), **event}, ensure_ascii=False) + '\n')


    def http(self, role, body):
        endpoint = self.args.endpoints[ROLES.index(role)]
        request = urllib.request.Request(endpoint, data=json.dumps(body).encode(),
                                         headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(request, timeout=self.args.http_timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f'{role}: HTTP {exc.code}: {exc.read().decode(errors="replace")}') from exc


    def agent(self, role, context, available, done, allowed_reads=()):
        """Bounded tool loop with compact, application-owned state on every turn.

        Histories are persisted in trace.jsonl; only the last tool exchange plus
        current state is resent, so old evidence does not fill the 2K context.
        """
        last_exchange = []
        for turn in range(self.args.max_turns):
            tools = available()
            if allowed_reads:
                def read(artifact_id):
                    value = json.loads(self.artifacts[artifact_id].read_text(encoding='utf-8'))
                    return {'artifact_id': artifact_id, 'data': value}
                tools.append(tool('read_artifact', 'Read an authorized shared artifact.',
                                  obj({'artifact_id': string(enum=list(allowed_reads))}), read))

            definitions = [item[0] for item in tools]
            handlers = {item[0]['function']['name']: item for item in tools}
            messages = [{'role': 'user', 'content': json.dumps(context(), ensure_ascii=False)}] + last_exchange
            body = {'model': role, 'messages': messages, 'tools': definitions,
                    'tool_choice': 'required', 'parallel_tool_calls': False,
                    'temperature': self.args.temperature, 'max_tokens': self.args.max_tokens}
            # First create-workspace call precedes disk logging; retain it afterward.
            response = self.transport(role, body)
            choice = response['choices'][0]
            message = choice['message']
            if self.created:
                self.log({'role': role, 'request': body, 'response': response})

            if choice.get('finish_reason') == 'length':
                raise RuntimeError(f'{role}: truncated response; raise --max-tokens/context length')

            calls = message.get('tool_calls') or []
            if len(calls) != 1:
                raise RuntimeError(f'{role}: expected exactly one tool call, received {len(calls)}')

            call = calls[0]
            name = call['function']['name']
            try:
                if name not in handlers:
                    raise ValueError('Tool is not authorized for the current stage')

                definition, handler = handlers[name]
                arguments = json.loads(call['function']['arguments'])
                validate(arguments, definition['function']['parameters'])
                result = handler(**arguments)

            except (ValidationError, ValueError, KeyError) as exc:
                result = {'error': str(exc)[:500], 'instruction': 'Correct the tool arguments.'}

            if self.created:
                if name == 'create_workspace':
                    self.log({'role': role, 'request': body, 'response': response})

                self.log({'role': role, 'tool_call_id': call['id'], 'tool': name, 'result': result})

            if done():
                return result

            last_exchange = [
                {'role': 'assistant', 'content': message.get('content'), 'tool_calls': calls},
                {'role': 'tool', 'tool_call_id': call['id'], 'content': json.dumps(result, ensure_ascii=False)},
            ]
        raise RuntimeError(f'{role}: tool-call limit reached')


    def execute(self, req, test):
        rid, tid = req['requirement_id'], test['test_case_id']
        record = {'execution_id': str(uuid.uuid4()), 'requirement_id': rid,
                  'test_case_id': tid, 'input': test['input'], 'started_at': now(),
                  'calculator_path': str(self.calculator), 'error': None}
        current = {str(p.relative_to(self.calculator.parent)): digest(p)
                   for p in sorted(self.calculator.parent.rglob('*.py'))}
        if current != self.source_hashes:
            raise RuntimeError('Calculator source changed during the run; restart validation')

        record['source_sha256'] = current
        started = time.perf_counter()
        with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
            process = None
            try:
                process = subprocess.Popen([sys.executable, str(self.calculator), test['input']],
                    stdout=stdout, stderr=stderr, cwd=self.calculator.parent,
                    start_new_session=(os.name == 'posix'))
                code = process.wait(timeout=self.args.calculator_timeout)
                record.update(return_code=code, execution_status='COMPLETED' if code == 0 else 'ERROR')

            except subprocess.TimeoutExpired:
                if os.name == 'posix':
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()

                process.wait()
                record.update(return_code=process.returncode, execution_status='TIMEOUT', error='Execution timed out')

            except OSError as exc:
                record.update(return_code=None, execution_status='LAUNCH_ERROR', error=str(exc))

            except BaseException:
                if process is not None and process.poll() is None:
                    if os.name == 'posix':
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                    process.wait()
                raise

            record['output_truncated'] = False
            for name, stream in [('stdout', stdout), ('stderr', stderr)]:
                stream.seek(0)
                data = stream.read(1_048_577)
                record['output_truncated'] |= len(data) > 1_048_576
                record[name] = data[:1_048_576].decode('utf-8', errors='replace')

        record['duration_ms'] = round((time.perf_counter() - started) * 1000, 3)
        return record


    def verify(self, req, decision, test, evidence, evidence_ref):
        verdict = {}


        def submit_verdict(criteria_justified, verdict_value, explanation):
            if not criteria_justified and verdict_value != 'INCONCLUSIVE':
                raise ValueError('Unjustified success criteria require INCONCLUSIVE')

            if (evidence['execution_status'] in ('TIMEOUT', 'LAUNCH_ERROR') or evidence['output_truncated']) and verdict_value != 'INCONCLUSIVE':
                raise ValueError('Incomplete/infrastructure evidence requires INCONCLUSIVE')

            verdict.update(requirement_id=req['requirement_id'], test_case_id=test['test_case_id'],
                           execution_id=evidence['execution_id'], evidence_artifact=evidence_ref,
                           criteria_justified=criteria_justified, verdict=verdict_value, explanation=explanation)
            ref = self.save(f'verdicts/{test["test_case_id"]}.json', verdict)
            return {'artifact_id': ref, 'verdict': verdict_value}


        def available():
            return [tool('submit_verdict', 'Save the verdict linked to this test and execution.', obj({
                'criteria_justified': {'type': 'boolean'},
                'verdict_value': string(enum=['PASS', 'FAIL', 'INCONCLUSIVE']),
                'explanation': string(minLength=1, maxLength=1500)}), submit_verdict)]


        # Large output remains on disk. Never silently show a clipped record as complete.
        visible = dict(evidence)
        for field in ('stdout', 'stderr'):
            if len(visible[field]) > 1800:
                visible[field] = visible[field][:1800]
                visible['output_truncated'] = True

        evidence = visible
        self.agent(ROLES[2], lambda: {'scope': self.scope, 'requirement': req,
            'test_definition': test, 'evidence': evidence}, available, lambda: bool(verdict))
        return verdict


    def tester(self, req, decision, decision_ref):
        tests = {t['test_case_id']: t for t in decision['tests']}
        evidence, refs, verdicts = {}, {}, []
        delegated = False


        def run_calculator(test_case_id):
            if test_case_id not in evidence:
                record = self.execute(req, tests[test_case_id])
                refs[test_case_id] = self.save(f'evidence/{test_case_id}.json', record)
                evidence[test_case_id] = record
            return {'artifact_id': refs[test_case_id], 'test_case_id': test_case_id,
                    'execution_status': evidence[test_case_id]['execution_status']}


        def verification_engineer():
            nonlocal delegated
            try:
                for tid, test in tests.items():
                    verdicts.append(self.verify(req, decision, test, evidence[tid], refs[tid]))

            except Exception as exc:
                raise RuntimeError(f'Analysis handoff failed: {exc}') from exc

            delegated = True
            return {'reviewed_tests': len(verdicts)}


        def available():
            pending = [tid for tid in tests if tid not in evidence]
            if pending:
                return [tool('run_calculator', 'Execute an approved test and save real execution evidence.',
                             obj({'test_case_id': string(enum=pending)}), run_calculator)]

            return [tool('verification_engineer', 'Delegate all saved evidence to the analysis agent.',
                         obj({}), verification_engineer)]

        self.agent(ROLES[1], lambda: {'requirement_id': req['requirement_id'],
                   'test_artifact': decision_ref, 'approved_tests': decision['tests'],
                   'executed_test_ids': list(evidence)}, available, lambda: delegated,
                   allowed_reads=[decision_ref])

        return verdicts


    def requirement(self, req):
        completed = False
        test_schema = obj({'test_case_id': string(pattern='^' + re.escape(req['requirement_id']) + r'-T0[1-3]$'),
                           'input': string(minLength=1, maxLength=500),
                           'expected_output': string(maxLength=500),
                           'justification': string(minLength=1, maxLength=700)})


        def submit_to_test_engineer(decision, rationale, tests):
            nonlocal completed
            if (decision == 'ACCEPT') != bool(tests):
                raise ValueError('ACCEPT needs 1-3 tests; other decisions require no tests')

            ids = [t['test_case_id'] for t in tests]
            if len(ids) != len(set(ids)):
                raise ValueError('Duplicate test IDs')

            record = {'requirement_id': req['requirement_id'], 'decision': decision,
                      'rationale': rationale, 'tests': tests}
            ref = self.save(f'decisions/{req["requirement_id"]}.json', record)
            row = {**record, 'decision_artifact': ref, 'verdicts': [], 'workflow_status': 'IN_PROGRESS'}
            self.rows.append(row)
            if tests:
                try:
                    row['verdicts'] = self.tester(req, record, ref)
                except Exception as exc:
                    raise RuntimeError(f'Test handoff failed: {exc}') from exc

            row['workflow_status'] = 'COMPLETED' if tests else 'NOT_TESTED'
            completed = True

            return {'decision_artifact': ref, 'reviewed_tests': len(row['verdicts'])}


        self.agent(ROLES[0], lambda: {'scope': self.scope, 'requirement': req},
            lambda: [tool('submit_to_test_engineer', 'Save the decision and delegate accepted tests to the tester.',
            obj({'decision': string(enum=['ACCEPT', 'REJECT', 'NEEDS_CLARIFICATION']),
            'rationale': string(minLength=1, maxLength=1000),
            'tests': {'type': 'array', 'items': test_schema, 'maxItems': 3}}), submit_to_test_engineer)],
            lambda: completed)


    def run(self):
        self.agent(ROLES[0], lambda: {'task': 'Create the shared artifact workspace for this validation run.'},
            lambda: [tool('create_workspace', 'Create the application-scoped shared directory.',
                          obj({}), self.create_workspace)], lambda: self.created)
        error = None

        try:
            for req in self.requirements:
                print(f'Validating {req["requirement_id"]} ...', flush=True)
                self.requirement(req)
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
            self.log({'workflow_error': error})

        report = {'run_id': self.run_id, 'workflow_status': 'ERROR' if error else 'COMPLETED',
                  'error': error, 'artifact_directory': str(self.pool), 'requirements': self.rows,
                  'not_completed': [r['requirement_id'] for r in self.requirements
                    if not any(row['requirement_id'] == r['requirement_id'] and row['workflow_status'] != 'IN_PROGRESS'
                               for row in self.rows)],
                  'interpretation': 'PASS applies only to generated tests, not exhaustive requirement coverage.'}

        with self.report_path.open('x', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f'Report: {self.report_path}', flush=True)
        if error:
            raise RuntimeError(error)

        return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('requirements', type=Path)
    p.add_argument('calculator', type=Path)
    p.add_argument('--scope', type=Path, default=BASE / 'calculator_v1_scope.txt')
    p.add_argument('--output', type=Path, default=BASE / 'results/calculator_v1')
    p.add_argument('--endpoints', nargs=3, default=[f'http://127.0.0.1:{port}/v1/chat/completions' for port in (8001,8002,8003)])
    p.add_argument('--temperature', type=float, default=0)
    p.add_argument('--max-tokens', type=int, default=768)
    p.add_argument('--max-turns', type=int, default=12)
    p.add_argument('--http-timeout', type=float, default=180)
    p.add_argument('--calculator-timeout', type=float, default=5)
    args = p.parse_args()

    try:
        Pipeline(args).run()
    except Exception as exc:
        print(f'Validation failed: {exc}', file=sys.stderr)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
