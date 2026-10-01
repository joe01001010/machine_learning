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
ROLES = ('requirements-engineer', 'test-case-engineer', 'test-engineer', 'verification-engineer')


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
        if len(args.endpoints) != len(ROLES):
            raise ValueError('Provide four endpoints in requirements, test-case, test, verification order')

        for name, default in [('max_turns', 12), ('max_model_calls', 160), ('max_consultation_depth', 3), ('max_revisions', 3)]:
            if getattr(args, name, default) < 1:
                raise ValueError(f'{name} must be positive')

        self.args = args
        self.transport = transport or self.http
        self.calculator = args.calculator.resolve(strict=True)

        prompt = getattr(args, 'prompt', None)
        prompt_file = getattr(args, 'prompt_file', None)

        if prompt is None and prompt_file is None:
            raise ValueError('Provide exactly one of --prompt or --prompt-file')

        elif prompt is None:
            self.customer_prompt = Path(prompt_file).read_text(encoding='utf-8-sig')

        else:
            self.customer_prompt = prompt

        if not self.customer_prompt.strip():
            raise ValueError('Customer prompt must not be empty')

        if len(self.customer_prompt) > 16000:
            raise ValueError('Customer prompt exceeds 16000 characters; shorten it without losing requirements')

        self.scope = self.customer_prompt
        self.requirements = []
        self.intake_active = False
        self.intake_complete = False
        self.run_id = uuid.uuid4().hex[:12]
        self.root = args.output.resolve()
        self.pool = self.root / 'artifacts' / self.run_id
        self.report_path = self.root / f'verification_{self.run_id}.json'
        self.rows = []
        self.artifacts = {}
        self.created = False
        self.escalations = []
        self.escalated = False
        self.pending_revision = None
        self.reviewing_revision = False
        self.revisions = []
        self.registry = {}
        self.active_requirement = None
        self.model_calls = 0
        self.consultation_stack = []
        self.communication = {role: [] for role in ROLES}


    def create_workspace(self):
        if not self.created:
            self.pool.mkdir(parents=True, exist_ok=False)
            (self.pool / 'communication').mkdir()
            self.created = True

            self.save('customer_prompt.json', {'text': self.customer_prompt, 'source': 'user'})
            self.save('scope.json', {'text': self.scope, 'source': 'customer_prompt.json', 'authority': 'user'})

            hashes = []
            for p in sorted(self.calculator.parent.rglob('*.py')):
                temp_key = str(p.relative_to(self.calculator.parent))
                hashes.append({temp_key: digest(p)})

            self.source_hashes = hashes

            for name in hashes:
                self.save('calculator_source.json', {name: (self.calculator.parent / name).read_text(encoding='utf-8')})

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


    def read_shared(self, artifact_id, offset=0):
        if artifact_id not in self.artifacts:
            raise ValueError('Unknown artifact ID; use list_artifacts')
        content = self.artifacts[artifact_id].read_text(encoding='utf-8')
        if offset < 0 or offset > len(content):
            raise ValueError('Invalid character offset')
        end = min(offset + 2000, len(content))
        return {'artifact_id': artifact_id, 'content': content[offset:end],
                'next_offset': end if end < len(content) else None,
                'total_characters': len(content)}

    def create_artifact(self, role, title, content, purpose, supersedes=''):
        if supersedes and not supersedes.startswith('communication/artifacts/'):
            raise ValueError('Only agent-authored artifacts can be revised')
        if supersedes:
            if supersedes not in self.artifacts:
                raise ValueError('Unknown superseded artifact')
            old = json.loads(self.artifacts[supersedes].read_text())
            if old['author'] != role:
                raise ValueError('Create your own review artifact instead of revising another author')
        ref = self.save(f'communication/artifacts/{uuid.uuid4().hex}.json',
            {'author': role, 'created_at': now(), 'title': title, 'purpose': purpose,
             'content': content, 'supersedes': supersedes or None,
             'provenance': 'agent-authored; not execution evidence'})
        self.communication[role].append(ref)
        return {'artifact_id': ref}

    def send_message(self, sender, recipient, message, artifact_ids, context):
        if recipient not in ROLES or recipient == sender:
            raise ValueError('Choose another team role')
        if any(ref not in self.artifacts for ref in artifact_ids):
            raise ValueError('Unknown attachment; use list_artifacts')
        if len(self.consultation_stack) >= getattr(self.args, 'max_consultation_depth', 3):
            raise ValueError('Consultation depth limit reached; reply or mark unresolved')
        request_id = uuid.uuid4().hex
        request_ref = self.save(f'communication/messages/{request_id}.json',
            {'sender': sender, 'recipient': recipient, 'message': message,
             'artifact_ids': artifact_ids, 'created_at': now(),
             'parent_message': self.consultation_stack[-1] if self.consultation_stack else None,
             'context': context})
        self.communication[sender].append(request_ref)
        self.communication[recipient].append(request_ref)
        reply = {}

        def respond(message, artifact_ids):
            if any(ref not in self.artifacts for ref in artifact_ids):
                raise ValueError('Unknown reply attachment')
            ref = self.save(f'communication/replies/{request_id}.json',
                {'sender': recipient, 'recipient': sender, 'in_reply_to': request_ref,
                 'message': message, 'artifact_ids': artifact_ids, 'created_at': now()})
            reply.update(message=message, artifact_ids=artifact_ids, reply_artifact=ref)
            self.communication[sender].append(ref)
            self.communication[recipient].append(ref)
            return reply

        self.consultation_stack.append(request_ref)
        try:
            self.agent(recipient, lambda: {
                'mode': 'consultation', 'from': sender, 'message': message,
                'attachments': artifact_ids, 'request_artifact': request_ref,
                'scope_artifact': 'scope.json',
                'instruction': 'Read attachments/context as needed. Consult peers or create artifacts, then reply_to_message. Advice does not change approved tests or evidence.'},
                lambda: [tool('reply_to_message', 'Return an answer to the requesting team.',
                    obj({'message': string(minLength=1, maxLength=2000),
                         'artifact_ids': {'type': 'array', 'items': string(), 'maxItems': 8}}), respond)],
                lambda: bool(reply))
        except Exception as exc:
            self.save(f'communication/errors/{request_id}.json',
                      {'request_artifact': request_ref, 'error': str(exc), 'created_at': now()})
            raise RuntimeError(f'Consultation with {recipient} failed: {exc}') from exc
        finally:
            self.consultation_stack.pop()
        return {'request_artifact': request_ref, **reply,
                'revision_requested': self.pending_revision['request_id'] if self.pending_revision else None}

    def shared_tools(self, role, context):
        def listing(offset):
            refs = list(self.artifacts)
            if offset < 0 or offset > len(refs):
                raise ValueError('Invalid list offset')
            page = refs[offset:offset + 20]
            items = []
            for ref in page:
                item = {'artifact_id': ref}
                if ref.startswith('communication/artifacts/'):
                    data = json.loads(self.artifacts[ref].read_text())
                    item.update({key: data[key] for key in ('author', 'title', 'purpose', 'supersedes')})
                items.append(item)
            return {'artifacts': items,
                    'next_offset': offset + 20 if offset + 20 < len(refs) else None}

        revision_tools = []
        if self.active_requirement and not self.reviewing_revision:
            revision_tools = [tool('request_revision',
                'Pause this requirement and request owner-reviewed rework of a CURRENT artifact. Software defects remain FAIL.',
                obj({'target': string(), 'reason': string(minLength=1, maxLength=1500),
                     'requested_change': string(minLength=1, maxLength=1500),
                     'evidence_refs': {'type': 'array', 'items': string(), 'maxItems': 8}}),
                lambda **kw: self.request_revision(role, **kw))]
        if self.active_requirement or self.intake_active:
            revision_tools.append(tool('escalate_to_user', 'Stop this requirement for a user decision; never assume approval.',
                obj({'question': string(minLength=1, maxLength=2000),
                     'reason': string(minLength=1, maxLength=1500),
                     'artifact_ids': {'type': 'array', 'items': string(), 'maxItems': 8}}),
                lambda **kw: self.escalate_to_user(role, **kw)))
        return revision_tools + [
            tool('send_message', 'Ask another team and receive its reply. Attach artifact IDs.',
                 obj({'recipient': string(enum=[r for r in ROLES if r != role]),
                      'message': string(minLength=1, maxLength=2000),
                      'artifact_ids': {'type': 'array', 'items': string(), 'maxItems': 8}}),
                 lambda recipient, message, artifact_ids: self.send_message(
                     role, recipient, message, artifact_ids, context())),
            tool('create_artifact', 'Save a shared note, draft, review or final output. Revisions create new records.',
                 obj({'title': string(minLength=1, maxLength=120),
                      'content': string(minLength=1, maxLength=16000),
                      'purpose': string(enum=['working', 'shared', 'final']),
                      'supersedes': string(maxLength=200)}),
                 lambda **kw: self.create_artifact(role, **kw)),
            tool('list_artifacts', 'List run artifacts, paginated. Start at offset 0.',
                 obj({'offset': {'type': 'integer', 'minimum': 0}}), listing),
            tool('read_artifact', 'Read any registered run artifact in 2000-character pages.',
                 obj({'artifact_id': string(), 'offset': {'type': 'integer', 'minimum': 0}}),
                 self.read_shared),
        ]

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


    def agent(self, role, context, available, done):
        """Bounded tool loop with compact, application-owned state on every turn.

        Histories are persisted in trace.jsonl; only the last tool exchange plus
        current state is resent. Full messages and artifacts remain on disk.
        """
        last_exchange = []
        invalid_attempts = {}
        for turn in range(self.args.max_turns):
            tools = available()
            if self.created:
                tools += self.shared_tools(role, context)
            if self.model_calls >= getattr(self.args, 'max_model_calls', 160):
                raise RuntimeError('Run-wide model-call limit reached')
            self.model_calls += 1
            definitions = [item[0] for item in tools]
            handlers = {item[0]['function']['name']: item for item in tools}
            messages = [{'role': 'user', 'content': json.dumps({**context(), 'recent_communication': self.communication[role][-6:],
                'workflow': self.workflow_context()}, ensure_ascii=False)}] + last_exchange
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
            validation_failed = False
            try:
                if name not in handlers:
                    raise ValueError('Tool is not authorized for the current stage')

                definition, handler = handlers[name]
                arguments = json.loads(call['function']['arguments'])
                validate(arguments, definition['function']['parameters'])
                result = handler(**arguments)

            except (ValidationError, ValueError, KeyError) as exc:
                validation_failed = True
                result = {'error': str(exc)[:2000],
                          'instruction': 'Change the rejected fields before resubmitting. Do not repeat the same arguments.'}

            if self.created:
                if name == 'create_workspace':
                    self.log({'role': role, 'request': body, 'response': response})

                self.log({'role': role, 'tool_call_id': call['id'], 'tool': name, 'result': result})

            if validation_failed:
                try:
                    canonical = json.dumps(json.loads(call['function']['arguments']), sort_keys=True)
                except (ValueError, TypeError):
                    canonical = str(call['function']['arguments'])
                signature = (name, canonical)
                invalid_attempts[signature] = invalid_attempts.get(signature, 0) + 1
                print(f'{role}: {name} rejected: {result["error"]}', flush=True)
                if invalid_attempts[signature] >= 3:
                    raise RuntimeError(
                        f'{role}: repeated invalid {name} arguments 3 times; last validation error: {result["error"]}')
            if self.escalated:
                return {'status': 'NEEDS_USER'}
            if self.pending_revision is not None and not self.reviewing_revision:
                return {'revision_requested': self.pending_revision['request_id']}
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


    def workflow_context(self):
        rid = self.active_requirement
        return {'requirement_id': rid,
                'current_artifacts': [{'artifact_id': ref, 'kind': meta['kind'],
                                       'revision': meta['revision']}
                    for ref, meta in self.registry.items()
                    if meta['requirement_id'] == rid and meta['status'] == 'CURRENT'],
                'revision_feedback': getattr(self, 'revision_feedback', None)}

    def checkpoint(self):
        self.save(f'workflow/{uuid.uuid4().hex}.json',
                  {'time': now(), 'requirement_id': self.active_requirement,
                   'artifacts': self.registry, 'revisions': self.revisions,
                   'escalations': self.escalations})

    def versioned(self, path, data, kind, stage, dependencies):
        previous = [(ref, meta) for ref, meta in self.registry.items() if meta['lineage'] == path]
        revision = len(previous) + 1
        ref = path if revision == 1 else path[:-5] + f'.r{revision}.json'
        record = {**data, 'revision': revision, 'dependencies': dependencies,
                  'supersedes': previous[-1][0] if previous else None}
        self.save(ref, record)
        for _, meta in previous:
            meta['status'] = 'STALE'
        self.registry[ref] = {'lineage': path, 'revision': revision, 'kind': kind,
            'stage': stage, 'owner': ROLES[stage], 'requirement_id': self.active_requirement,
            'status': 'CURRENT', 'dependencies': dependencies}
        self.checkpoint()
        return ref

    def revision_event(self, request, status, **details):
        request.update(status=status, **details)
        self.save(f'revisions/{request["request_id"]}/{uuid.uuid4().hex}.json',
                  {**request, 'time': now()})
        self.checkpoint()

    def request_revision(self, role, target, reason, requested_change, evidence_refs):
        if self.reviewing_revision or self.pending_revision:
            raise ValueError('Finish the current revision review first')
        meta = self.registry.get(target)
        if not meta or meta['status'] != 'CURRENT' or meta['requirement_id'] != self.active_requirement:
            raise ValueError('Target must be a CURRENT artifact of this requirement')
        if any(ref not in self.artifacts for ref in evidence_refs):
            raise ValueError('Unknown evidence reference')
        request = {'request_id': uuid.uuid4().hex, 'target': target, 'requester': role,
                   'owner': meta['owner'], 'requirement_id': self.active_requirement,
                   'reason': reason, 'requested_change': requested_change,
                   'evidence_refs': evidence_refs}
        self.revisions.append(request)
        self.pending_revision = request
        self.revision_event(request, 'OPEN')
        return {'request_id': request['request_id'], 'status': 'OPEN'}

    def escalate_to_user(self, role, question, reason, artifact_ids):
        if any(ref not in self.artifacts for ref in artifact_ids):
            raise ValueError('Unknown escalation attachment')
        record = {'author': role, 'requirement_id': self.active_requirement,
                  'question': question, 'reason': reason, 'artifact_ids': artifact_ids,
                  'created_at': now(), 'status': 'NEEDS_USER'}
        ref = self.save(f'escalations/{uuid.uuid4().hex}.json', record)
        record['artifact_id'] = ref
        self.escalations.append(record)
        self.escalated = True
        self.checkpoint()
        return {'status': 'NEEDS_USER', 'artifact_id': ref}

    def test_schema(self, req):
        return obj({'test_case_id': string(pattern='^' + re.escape(req['requirement_id']) + r'-T0[1-3]$'),
                    'input': string(minLength=1, maxLength=500),
                    'expected_output': string(maxLength=500),
                    'justification': string(minLength=1, maxLength=700)})

    def check_tests(self, tests):
        validate(tests, {'type': 'array', 'minItems': 1, 'maxItems': 3,
                         'items': self.test_schema(self.work['req'])})
        if len({test['test_case_id'] for test in tests}) != len(tests):
            raise ValueError('Duplicate test IDs')

    def verdict_schema(self):
        return obj({'criteria_justified': {'type': 'boolean'},
                    'verdict_value': string(enum=['PASS', 'FAIL', 'INCONCLUSIVE']),
                    'explanation': string(minLength=1, maxLength=1500)})

    def check_verdict(self, values, evidence):
        if not values['criteria_justified'] and values['verdict_value'] != 'INCONCLUSIVE':
            raise ValueError('Unjustified success criteria require INCONCLUSIVE')
        if (evidence['execution_status'] in ('TIMEOUT', 'LAUNCH_ERROR') or
                evidence['output_truncated'] or
                any(len(evidence.get(key, '')) > 1800 for key in ('stdout', 'stderr'))):
            if values['verdict_value'] != 'INCONCLUSIVE':
                raise ValueError('Incomplete/infrastructure evidence requires INCONCLUSIVE')

    def handle_revision(self):
        request = self.pending_revision
        target = request['target']
        meta = self.registry[target]
        count = sum(r['requirement_id'] == self.active_requirement for r in self.revisions)
        if count > getattr(self.args, 'max_revisions', 3):
            self.escalate_to_user('controller', 'Resolve repeated revision requests.',
                                  'Revision request budget exhausted.', [target])
            self.revision_event(request, 'NEEDS_USER')
            self.pending_revision = None
            return None
        owner = meta['owner']
        reviewer = request['requester'] if request['requester'] != owner else (
            ROLES[3] if owner != ROLES[3] else ROLES[0])
        stage = meta['stage']
        if stage == 0:
            changes_schema = obj({'text': string(minLength=1, maxLength=3000)})
        elif stage == 1:
            changes_schema = obj({'tests': {'type': 'array', 'minItems': 1, 'maxItems': 3,
                                            'items': self.test_schema(self.work['req'])}})
        elif stage == 2:
            changes_schema = obj({'action': string(enum=['RERUN'])})
        else:
            changes_schema = self.verdict_schema()
        feedback = None
        accepted = None
        self.reviewing_revision = True
        try:
            for attempt in range(getattr(self.args, 'max_revisions', 3)):
                proposal = {}
                review = {}

                def submit_revision(request_id, base_artifact, changes, rationale):
                    if request_id != request['request_id'] or base_artifact != target:
                        raise ValueError('Proposal must match this request and base artifact')
                    if self.registry[target]['status'] != 'CURRENT':
                        raise ValueError('Proposal base is stale')
                    validate(changes, changes_schema)
                    if stage == 1:
                        self.check_tests(changes['tests'])
                    if stage == 3:
                        old = json.loads(self.artifacts[target].read_text())
                        ev = json.loads(self.artifacts[old['evidence_artifact']].read_text())
                        self.check_verdict(changes, ev)
                    data = {'request_id': request_id, 'base_artifact': base_artifact,
                            'author': owner, 'changes': changes, 'rationale': rationale}
                    ref = self.save(f'revisions/{request_id}/proposal-{attempt + 1}.json', data)
                    proposal.update(**data, artifact_id=ref)
                    self.revision_event(request, 'PROPOSED', proposal_artifact=ref)
                    return {'proposal_artifact': ref}

                self.agent(owner, lambda: {'mode': 'revision_proposal', 'request': request,
                    'base_artifact': target, 'scope': self.scope, 'requirement': self.work['req'],
                    'feedback': feedback,
                    'instruction': 'Submit a justified correction. Do not weaken valid tests to hide software defects. Escalate if user intent is missing.'},
                    lambda: [tool('submit_revision', 'Propose a replacement against the exact current base.',
                        obj({'request_id': string(), 'base_artifact': string(), 'changes': changes_schema,
                             'rationale': string(minLength=1, maxLength=1500)}), submit_revision)],
                    lambda: bool(proposal))
                if self.escalated:
                    break

                def review_revision(proposal_artifact, decision, rationale):
                    if proposal_artifact != proposal['artifact_id']:
                        raise ValueError('Review the current proposal only')
                    if self.registry[target]['status'] != 'CURRENT':
                        raise ValueError('Proposal base is stale')
                    review.update(decision=decision, rationale=rationale)
                    self.save(f'revisions/{request["request_id"]}/review-{attempt + 1}.json',
                              {'reviewer': reviewer, 'proposal_artifact': proposal_artifact, **review})
                    self.revision_event(request, decision, reviewer=reviewer)
                    return review

                self.agent(reviewer, lambda: {'mode': 'revision_review', 'request': request,
                    'proposal': proposal, 'scope': self.scope, 'requirement': self.work['req'],
                    'instruction': 'Check against approved scope, not observed software output. ACCEPT only justified changes; otherwise CHANGES_REQUESTED or escalate_to_user.'},
                    lambda: [tool('review_revision', 'Review another team proposal.',
                        obj({'proposal_artifact': string(),
                             'decision': string(enum=['ACCEPT', 'CHANGES_REQUESTED']),
                             'rationale': string(minLength=1, maxLength=1500)}), review_revision)],
                    lambda: bool(review))
                if self.escalated:
                    break
                if review['decision'] == 'ACCEPT':
                    accepted = proposal
                    break
                feedback = review['rationale']

            if not accepted and not self.escalated:
                self.escalate_to_user('controller', 'Resolve the disputed revision.',
                                      'Proposal review budget exhausted.', [target])
            if self.escalated:
                self.revision_event(request, 'NEEDS_USER')
                return None
            if stage == 0:
                self.escalate_to_user(owner, 'Approve or clarify the proposed requirement change.',
                    'Approved requirements cannot be changed by agent agreement.',
                    [target, accepted['artifact_id']])
                self.revision_event(request, 'NEEDS_USER')
                return None

            self.revision_feedback = accepted
            for ref, item in self.registry.items():
                if item['requirement_id'] == self.active_requirement and item['stage'] >= stage:
                    # Verdict-only changes affect just the target; other verdicts remain current.
                    if stage != 3 or ref == target:
                        item['status'] = 'STALE'
            self.work['row']['workflow_status'] = 'IN_PROGRESS'
            if stage <= 2:
                self.work['row']['verdicts'] = []
                self.work['evidence'] = {}
            if stage == 1:
                self.store_decision('ACCEPT', accepted['rationale'], accepted['changes']['tests'])
            if stage == 3:
                old = json.loads(self.artifacts[target].read_text())
                tid = old['test_case_id']
                self.store_verdict(tid, accepted['changes'])
            self.revision_event(request, 'APPLIED')
            return 2 if stage <= 2 else 4
        finally:
            self.reviewing_revision = False
            self.pending_revision = None

    def store_decision(self, decision, rationale, tests):
        req = self.work['req']
        record = {'requirement_id': req['requirement_id'], 'decision': decision,
                  'rationale': rationale, 'tests': tests}
        ref = self.versioned(f'decisions/{req["requirement_id"]}.json', record, 'tests', 1,
                             [self.work['requirement_ref'], self.work['review_ref']])
        self.work.update(decision=record, decision_ref=ref)
        self.work['row'].update(**record, decision_artifact=ref, verdicts=[])

    def store_verdict(self, tid, values):
        evidence_ref = self.work['evidence'][tid]
        evidence = json.loads(self.artifacts[evidence_ref].read_text())
        self.check_verdict(values, evidence)
        verdict = {'requirement_id': self.active_requirement, 'test_case_id': tid,
                   'execution_id': evidence['execution_id'], 'evidence_artifact': evidence_ref,
                   'criteria_justified': values['criteria_justified'],
                   'verdict': values['verdict_value'], 'explanation': values['explanation']}
        ref = self.versioned(f'verdicts/{tid}.json', verdict, 'verdict', 3,
                             [self.work['decision_ref'], evidence_ref])
        verdict['verdict_artifact'] = ref
        rows = self.work['row']['verdicts']
        rows[:] = [v for v in rows if v['test_case_id'] != tid] + [verdict]
        return {'artifact_id': ref, 'verdict': values['verdict_value']}

    def requirement_review(self):
        finished = False
        req = self.work['req']
        def submit_to_test_case_engineer(decision, rationale):
            nonlocal finished
            ref = self.versioned(f'requirement_reviews/{req["requirement_id"]}.json',
                {'requirement': req, 'decision': decision, 'rationale': rationale},
                'requirement_review', 0, [self.work['requirement_ref']])
            self.work.update(review_ref=ref, requirement_decision=decision)
            self.work['row'].update(decision=decision, rationale=rationale, decision_artifact=ref)
            finished = True
            return {'artifact_id': ref, 'decision': decision}
        self.agent(ROLES[0], lambda: {'scope': self.scope, 'requirement': req},
            lambda: [tool('submit_to_test_case_engineer', 'Save the requirement review.',
                obj({'decision': string(enum=['ACCEPT', 'REJECT', 'NEEDS_CLARIFICATION']),
                     'rationale': string(minLength=1, maxLength=1000)}), submit_to_test_case_engineer)],
            lambda: finished)

    def design_tests(self, req):
        finished = False
        def submit_to_test_engineer(decision, rationale, tests):
            nonlocal finished
            if (decision == 'ACCEPT') != bool(tests):
                raise ValueError('ACCEPT requires tests; other decisions require none')
            if tests:
                self.check_tests(tests)
            self.store_decision(decision, rationale, tests)
            finished = True
            return {'decision_artifact': self.work['decision_ref']}
        self.agent(ROLES[1], lambda: {'scope': self.scope, 'requirement': req},
            lambda: [tool('submit_to_test_engineer', 'Save tests for controller-scheduled execution.',
                obj({'decision': string(enum=['ACCEPT', 'REJECT', 'NEEDS_CLARIFICATION']),
                     'rationale': string(minLength=1, maxLength=1000),
                     'tests': {'type': 'array', 'maxItems': 3, 'items': self.test_schema(req)}}),
                submit_to_test_engineer)], lambda: finished)

    def tester(self):
        tests = {t['test_case_id']: t for t in self.work['decision']['tests']}
        finished = False
        def run_calculator(test_case_id):
            record = self.execute(self.work['req'], tests[test_case_id])
            ref = self.versioned(f'evidence/{test_case_id}.json', record, 'execution', 2,
                                 [self.work['decision_ref']])
            self.work['evidence'][test_case_id] = ref
            return {'artifact_id': ref, 'execution_status': record['execution_status']}
        def verification_engineer():
            nonlocal finished
            finished = True
            return {'status': 'READY_FOR_VERIFICATION'}
        def available():
            pending = [tid for tid in tests if tid not in self.work['evidence']]
            if pending:
                return [tool('run_calculator', 'Execute an approved test and save actual output.',
                             obj({'test_case_id': string(enum=pending)}), run_calculator)]
            return [tool('verification_engineer', 'Schedule verification of saved evidence.', obj({}),
                         verification_engineer)]
        self.agent(ROLES[2], lambda: {'requirement_id': self.active_requirement,
                   'test_artifact': self.work['decision_ref'], 'approved_tests': list(tests.values()),
                   'executed_test_ids': list(self.work['evidence'])}, available, lambda: finished)

    def verify(self):
        for test in self.work['decision']['tests']:
            if any(v['test_case_id'] == test['test_case_id'] for v in self.work['row']['verdicts']):
                continue
            evidence = json.loads(self.artifacts[self.work['evidence'][test['test_case_id']]].read_text())
            for field in ('stdout', 'stderr'):
                if len(evidence[field]) > 1800:
                    evidence[field] = evidence[field][:1800]
                    evidence['output_truncated'] = True
            finished = False
            def submit_verdict(**values):
                nonlocal finished
                result = self.store_verdict(test['test_case_id'], values)
                finished = True
                return result
            self.agent(ROLES[3], lambda: {'scope': self.scope, 'requirement': self.work['req'],
                       'test_definition': test, 'evidence': evidence},
                lambda: [tool('submit_verdict', 'Save assessment of this test and execution.',
                              self.verdict_schema(), submit_verdict)], lambda: finished)
            if self.pending_revision or self.escalated:
                return

    def requirement(self, req):
        self.active_requirement = req['requirement_id']
        self.escalated = False
        self.revision_feedback = None
        row = {'requirement_id': self.active_requirement, 'workflow_status': 'IN_PROGRESS',
               'tests': [], 'verdicts': []}
        self.rows.append(row)
        self.work = {'req': req, 'row': row, 'evidence': {}}
        self.work['requirement_ref'] = self.versioned(
            f'requirements/{self.active_requirement}.json', req, 'requirement', 0, ['scope.json'])
        stage = 0
        while stage < 4:
            if stage == 0:
                self.requirement_review()
            elif stage == 1:
                self.design_tests(req)
            elif stage == 2:
                self.tester()
            else:
                self.verify()
            if self.escalated:
                row['workflow_status'] = 'NEEDS_USER'
                break
            if self.pending_revision:
                stage = self.handle_revision()
                if self.escalated:
                    row['workflow_status'] = 'NEEDS_USER'
                    break
                # Verdict revision might be requested before other tests have verdicts.
                if stage == 4:
                    current_tids = {v['test_case_id'] for v in row['verdicts']}
                    required_tids = {t['test_case_id'] for t in self.work['decision']['tests']}
                    if current_tids != required_tids:
                        stage = 3
                continue
            decision = self.work.get('requirement_decision') if stage == 0 else self.work.get('decision', {}).get('decision')
            if stage < 2 and decision != 'ACCEPT':
                row['workflow_status'] = 'NOT_TESTED'
                break
            stage += 1
        else:
            row['workflow_status'] = 'COMPLETED'
            for request in self.revisions:
                if request['requirement_id'] == self.active_requirement and request['status'] == 'APPLIED':
                    self.revision_event(request, 'RESOLVED')
        if self.pending_revision:
            self.revision_event(self.pending_revision, 'NEEDS_USER')
            self.pending_revision = None
        self.checkpoint()
        self.active_requirement = None

    def generate_requirements(self):
        requirement_schema = obj({
            'text': string(minLength=10,
                           maxLength=1500,
                           description='One testable obligation using the word shall. Write the requirement here, not in source_quote.'),
            'source_quote': string(minLength=1,
                                   maxLength=1500,
                                   description='A short verbatim excerpt copied from customer_prompt, with exact spelling and punctuation. Do not summarize or rewrite.')})


        def submit_requirements(requirements):
            errors = []
            normalized = [item['text'].strip().casefold() for item in requirements]
            if len(set(normalized)) != len(normalized):
                errors.append('Duplicate requirements; submit distinct obligations.')
            for index, item in enumerate(requirements):
                if not re.search(r'\bshall\b', item['text'], flags=re.IGNORECASE):
                    errors.append(f'requirements[{index}].text must contain a shall statement.')
                if item['source_quote'] not in self.customer_prompt:
                    errors.append(f'requirements[{index}].source_quote must be an exact excerpt copied from customer_prompt; paraphrased summaries are invalid.')
            if errors:
                raise ValueError(' '.join(errors))
            generated = [{'requirement_id': f'REQ-{index:03d}', **item,
                          'source_artifact': 'customer_prompt.json',
                          'provenance': 'agent-generated interpretation of user directions'}
                         for index, item in enumerate(requirements, 1)]
            self.save('requirements.json', generated)
            self.requirements = generated
            self.intake_complete = True
            return {'artifact_id': 'requirements.json',
                    'requirement_ids': [item['requirement_id'] for item in generated]}

        self.intake_active = True
        try:
            self.agent(ROLES[0], lambda: {
                'mode': 'requirements_intake', 'customer_prompt': self.customer_prompt,
                'customer_prompt_artifact': 'customer_prompt.json',
                'execution_interface': 'The configured calculator is called as: python compute.py <one expression argument>. The host records stdout, stderr and return code.',
                'instruction': 'Generate distinct testable shall requirements covering user directions. Preserve requested behavior even if the implementation lacks it. Do not invent output formatting, supported operators, bounds or error policies. Include a literal source_quote for each requirement. Escalate material ambiguities or directions that cannot be assessed with the supplied interface.'},
                lambda: [tool('submit_requirements', 'Save generated requirements grounded in the customer prompt.',
                    obj({'requirements': {'type': 'array', 'minItems': 1, 'maxItems': 30,
                                          'items': requirement_schema}}), submit_requirements)],
                lambda: self.intake_complete)
        finally:
            self.intake_active = False


    def run(self):
        self.create_workspace()
        error = None

        try:
            self.generate_requirements()
            for req in self.requirements:
                print(f'Validating {req["requirement_id"]} ...', flush=True)
                self.requirement(req)

        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
            self.log({'workflow_error': error})

        unresolved = [r for r in self.revisions if r['status'] != 'RESOLVED']
        status = 'ERROR' if error else ('NEEDS_USER' if self.escalations or unresolved else 'COMPLETED')

        report = {'run_id': self.run_id, 'workflow_status': status, 'error': error,
            'artifact_directory': str(self.pool), 'communication_directory': str(self.pool / 'communication'),
            'model_calls': self.model_calls, 'requirements': self.rows,
            'intake_status': 'COMPLETED' if self.intake_complete else ('NEEDS_USER' if self.escalations else 'ERROR'),
            'customer_prompt_artifact': 'customer_prompt.json',
            'generated_requirements_artifact': 'requirements.json' if self.intake_complete else None,
            'revision_requests': self.revisions, 'escalations': self.escalations,
            'artifact_status': self.registry,
            'not_completed': [r['requirement_id'] for r in self.requirements if not any(
                row['requirement_id'] == r['requirement_id'] and row['workflow_status'] in ('COMPLETED', 'NOT_TESTED')
                for row in self.rows)],
            'interpretation': 'PASS applies only to current generated tests, not exhaustive requirement coverage.'}

        with self.report_path.open('x', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f'Report: {self.report_path}', flush=True)

        if error:
            raise RuntimeError(error)

        return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('calculator', type=Path)
    intake = p.add_mutually_exclusive_group(required=True)
    intake.add_argument('--prompt', help='Customer directions as text')
    intake.add_argument('--prompt-file', type=Path, help='UTF-8 text file containing customer directions')
    p.add_argument('--output', type=Path, default=BASE / 'results/calculator_v1')
    p.add_argument('--endpoints', nargs=4, default=[f'http://127.0.0.1:{port}/v1/chat/completions' for port in (8001,8002,8003,8004)])
    p.add_argument('--temperature', type=float, default=0)
    p.add_argument('--max-tokens', type=int, default=768)
    p.add_argument('--max-turns', type=int, default=12)
    p.add_argument('--max-revisions', type=int, default=3)
    p.add_argument('--max-model-calls', type=int, default=160)
    p.add_argument('--max-consultation-depth', type=int, default=3)
    p.add_argument('--http-timeout', type=float, default=180)
    p.add_argument('--calculator-timeout', type=float, default=5)
    args = p.parse_args()

    try:
        report = Pipeline(args).run()
        if report['workflow_status'] == 'NEEDS_USER':
            raise SystemExit(2)
    except Exception as exc:
        print(f'Validation failed: {exc}', file=sys.stderr)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
