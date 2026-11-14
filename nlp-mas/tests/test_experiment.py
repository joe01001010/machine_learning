"""Behavioral checks for the four-agent build/test/repair experiment.

Models are replaced by a scripted transport; generated applications are real
subprocesses. No running model servers are required by this suite.
"""
from contextlib import contextmanager
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
import re
from pathlib import Path
import shutil
import tempfile
import threading
import unittest

from experiment import BudgetExceeded, Config, Experiment


PROMPT = (
    'Build a Python command-line calculator that adds two integers supplied as '
    'one expression argument. Print the integer result and a newline; exit 0.'
)
REQUIREMENTS = [{
    'requirement_id': 'R001',
    'text': 'The calculator shall add two integer operands from an expression argument.',
    'source_quote': 'adds two integers supplied as one expression argument',
}]
TESTS = [
    {'test_case_id': 'R001-T01', 'args': ['2+3'], 'stdin': '',
     'expected_stdout': '5\n', 'expected_exit_code': 0,
     'justification': 'Two integer operands must be added and printed as an integer.'},
    {'test_case_id': 'R001-T02', 'args': ['4+7'], 'stdin': '',
     'expected_stdout': '11\n', 'expected_exit_code': 0,
     'justification': 'A second input checks that the sum is computed rather than hard-coded.'},
]
BROKEN = 'print(0)\n'
CORRECT = "import sys\nprint(sum(int(part) for part in sys.argv[1].split('+')))\n"


def completion(name, arguments):
    if name == 'submit_requirements':
        arguments = {**arguments, 'complete': arguments.get('complete', True)}
    if name == 'write_file':
        if isinstance(arguments.get('content'), str):
            arguments = {**arguments, 'content': arguments['content'].splitlines()}
        arguments = {**arguments, 'complete': arguments.get('complete', False)}
    return {'choices': [{'finish_reason': 'tool_calls', 'message': {
        'role': 'assistant', 'content': None,
        'tool_calls': [{'id': 'test-call', 'type': 'function', 'function': {
            'name': name, 'arguments': json.dumps(arguments)}}],
    }}]}


class ScriptedTeam:
    """A small team that builds a real faulty app, then repairs it from feedback."""
    def __init__(self, *, always_broken=False, always_pass=False, clarify=False,
                 invalid_source_quote=False, never_finish=False, program=None,
                 source_path="main.py", requirements=None, tests=None, auto_handoff=True):
        self.auto_handoff = auto_handoff
        self.always_broken = always_broken
        self.always_pass = always_pass
        self.clarify = clarify
        self.invalid_source_quote = invalid_source_quote
        self.never_finish = never_finish
        self.program = program
        self.source_path = source_path
        self.requirements = REQUIREMENTS if requirements is None else requirements
        self.tests = TESTS if tests is None else tests
        self.calls = []
        self.coding_rounds = []
        self.written_rounds = set()
        self.execution_rounds = []
        self.test_index = 0
        self.round = 0
        self.last_feedback = []

    def __call__(self, role, body):
        context = json.loads(body['messages'][0]['content'])
        tools = {t['function']['name']: t['function'] for t in body['tools']}
        self.calls.append((role, context, tuple(tools)))
        if self.clarify and 'ask_user' in tools:
            return completion('ask_user', {'question': 'Which operations should the calculator support?',
                                           'reason': 'The requested operations are unspecified.'})
        if 'submit_requirements' in tools:
            sources = dict(re.findall(r'^\[(S\d+)\] (.*)$', context['customer_prompt'], re.M))
            requirements = []
            for requirement in self.requirements:
                matches = [source_id for source_id, excerpt in sources.items()
                           if requirement['source_quote'] in excerpt or excerpt in requirement['source_quote']]
                assert matches or self.invalid_source_quote, 'Fixture quote must match a current original source fragment'
                requirements.append({'requirement_id': requirement['requirement_id'], 'text': requirement['text'],
                                     'source_id': 'S999' if self.invalid_source_quote else matches[0]})
            return completion('submit_requirements', {'requirements': requirements})
        if 'submit_tests' in tools:
            return completion('submit_tests', {'tests': self.tests, 'rationale': 'Checks the requested addition interface.'})
        if 'review_tests' in tools:
            return completion('review_tests', {
                'decision': 'ACCEPT',
                'feedback': 'The proposed inputs and exact expected outputs follow the requested arithmetic and interface.',
            })
        if 'write_file' in tools:
            self.round = context['round']
            self.last_feedback = context.get('feedback', [])
            if self.round not in self.written_rounds or self.never_finish:
                self.written_rounds.add(self.round)
                self.coding_rounds.append(context)
                source = self.program(self.round) if callable(self.program) else self.program
                if source is None:
                    source = BROKEN if self.always_broken or len(self.written_rounds) == 1 else CORRECT
                return completion('write_file', {'path': self.source_path, 'content': source, 'mode': 'replace',
                                                 'complete': self.auto_handoff and not self.never_finish})
            return completion('finish_build', {'summary': 'Build ready for execution by the test engineer.'})
        if 'run_test' in tools:
            schema = tools['run_test']['parameters']['properties']['test_case_id']
            candidates = schema.get('enum')
            tid = candidates[0] if candidates else self.tests[self.test_index % len(self.tests)]['test_case_id']
            self.test_index += 1
            self.execution_rounds.append((self.round, tid))
            return completion('run_test', {'test_case_id': tid})
        if 'submit_verdict' in tools:
            execution = context['execution']
            incomplete = execution['execution_status'] != 'COMPLETED' or execution['output_truncated']
            passing = self.always_pass or execution['host_match']
            verdict = 'PASS' if passing else 'INCONCLUSIVE' if incomplete else 'FAIL'
            return completion('submit_verdict', {
                'verdict': verdict, 'criteria_justified': True,
                'explanation': 'Observed execution matches the approved expected output.' if passing
                               else 'Execution is incomplete.' if incomplete
                               else 'Observed output does not equal the expected sum.',
            })
        raise AssertionError(f'Unexpected model task {role}: {tuple(tools)}; context={context}')


class PlanReviewTeam(ScriptedTeam):
    def __init__(self, *, always_revise=False):
        super().__init__(program=CORRECT)
        self.always_revise = always_revise
        self.design_attempts = 0
        self.review_attempts = 0
        self.accepted = False
        self.experiment = None
        self.plan_before_coding = None

    def __call__(self, role, body):
        tools = {tool['function']['name'] for tool in body['tools']}
        if 'submit_tests' in tools:
            self.design_attempts += 1
            state = json.loads(body['messages'][0]['content'])
            if self.design_attempts == 1:
                assert not state.get('previous_tests')
                tests = [dict(test) for test in TESTS]
            else:
                assert state['review_feedback'] == 'For input 2+3, the requested arithmetic sum is 5, not 999.'
                previous = state['previous_tests']
                assert [test['args'] for test in previous] == [['2+3'], ['4+7']]
                assert [test['expected_stdout'] for test in previous] == ['999\n', '11\n']
                tests = [{key: test[key] for key in TESTS[0]} for test in previous]
                tests[0]['expected_stdout'] = '5\n'
            if self.design_attempts == 1 or self.always_revise:
                tests[0]['expected_stdout'] = '999\n'
            return completion('submit_tests', {'tests': tests, 'rationale': 'Proposed arithmetic checks.'})
        if 'review_tests' in tools:
            self.review_attempts += 1
            revise = self.always_revise or self.design_attempts == 1
            self.accepted = not revise
            return completion('review_tests', {
                'decision': 'REVISE' if revise else 'ACCEPT',
                'feedback': 'For input 2+3, the requested arithmetic sum is 5, not 999.' if revise
                            else 'The corrected test plan follows the customer prompt.',
            })
        if 'write_file' in tools:
            assert self.accepted, 'Code generation began before an independent test-plan acceptance'
            current = (self.experiment.root / 'test_plan.json').read_bytes()
            if self.plan_before_coding is None:
                self.plan_before_coding = current
            assert current == self.plan_before_coding, 'Frozen tests changed during coding'
        return super().__call__(role, body)


class ReferenceTeam(ScriptedTeam):
    def __init__(self, *, always_wrong=False):
        prompt = ('Build a Python calculator that computes irrational powers rounded to '
                  '17 significant digits using round-half-even, and prints the result followed by a newline.')
        requirement = {'requirement_id': 'R001',
                       'text': 'The calculator shall round irrational powers to 17 significant digits.',
                       'source_quote': '17 significant digits using round-half-even'}
        test = {**TESTS[0], 'args': ['2^0.5'], 'expected_stdout': '1.414213562373095\n',
                'justification': 'Square root of two requires irrational-result rounding.'}
        super().__init__(program="print('1.414213562373095')\n", requirements=[requirement], tests=[test])
        self.prompt = prompt
        self.always_wrong = always_wrong
        self.submissions = 0
        self.reference_roles = []
        self.design_reference = None
        self.review_reference = None
        self.review_count = 0
        self.rejections = []

    def __call__(self, role, body):
        context = json.loads(body['messages'][0]['content'])
        tools = {tool['function']['name'] for tool in body['tools']}
        previous = context.get('previous_action') or {}
        if 'submit_tests' in tools:
            assert 'reference_math' in tools
            if previous.get('result', {}).get('error'):
                self.rejections.append(previous['result']['error'])
            if previous.get('tool') == 'reference_math':
                self.design_reference = previous['result']
            if self.submissions and not self.design_reference and not self.always_wrong:
                self.reference_roles.append(role)
                return completion('reference_math', {'expression': '2^0.5'})
            test = dict(self.tests[0])
            if self.always_wrong or not self.design_reference:
                test['expected_stdout'] = '1.4142135623730951\n'
            else:
                test['expected_stdout'] = self.design_reference['expected_stdout']
                test['expected_exit_code'] = self.design_reference['expected_exit_code']
            self.submissions += 1
            return completion('submit_tests', {'tests': [test], 'rationale': 'Proposed irrational-power rounding check.'})
        if 'review_tests' in tools:
            self.review_count += 1
            if self.always_wrong:
                return completion('review_tests', {'decision': 'ACCEPT',
                                                   'feedback': 'Accept this proposal even if its arithmetic is wrong.'})
            assert all(check['matches'] for check in context['mathematical_reference_checks'])
            assert 'reference_math' in tools
            if previous.get('tool') == 'reference_math':
                self.review_reference = previous['result']
            if self.review_reference is None:
                self.reference_roles.append(role)
                return completion('reference_math', {'expression': '2^0.5'})
            return completion('review_tests', {'decision': 'ACCEPT', 'feedback': 'Scope and independent numerical reference agree.'})
        return super().__call__(role, body)


@contextmanager
def model_http_server(transport, *, tokenize=None, model_metadata=None):
    """Local OpenAI-shaped stub exercising the real urllib transport."""
    errors = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            try:
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                result = tokenize(body) if self.path == '/tokenize' and tokenize else transport(body['model'], body)
                payload = json.dumps(result).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except Exception as exc:
                errors.append(exc)
                self.send_error(500)

        def do_GET(self):
            if self.path != '/v1/models' or model_metadata is None:
                self.send_error(404)
                return
            payload = json.dumps(model_metadata).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = HTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}/v1/chat/completions'
    finally:
        server.shutdown()
        thread.join(timeout=3)
        server.server_close()
    if errors:
        raise AssertionError(f'Local model stub failed: {errors}')


class TruncatedCoder(ScriptedTeam):
    def __init__(self, *, always=False):
        super().__init__(program=CORRECT, auto_handoff=False)
        self.always = always
        self.requests = []
        self.experiment = None

    def __call__(self, role, body):
        state = json.loads(body['messages'][0]['content'])
        self.requests.append(state)
        if len(self.requests) == 1 or self.always:
            response = completion('write_file', {
                'path': 'main.py', 'content': '# TRUNCATED_SENTINEL\n' * 300,
                'mode': 'replace',
            })
            response['choices'][0]['finish_reason'] = 'length'
            return response
        if len(self.requests) == 2:
            assert not (self.experiment.app / 'main.py').exists(), 'Truncated write was executed'
            assert 'no action was executed' in state['previous_action']['error']
            assert 'TRUNCATED_SENTINEL' not in json.dumps(state), 'Truncated payload leaked into retry context'
        return super().__call__(role, body)


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.output = Path(self.temp.name) / 'results'

    def tearDown(self):
        self.temp.cleanup()

    def experiment(self, transport, **options):
        return Experiment(Config(prompt=PROMPT, output=self.output, **options), transport=transport)

    def test_failed_build_is_repaired_and_all_unchanged_tests_are_rerun(self):
        team = ScriptedTeam()
        exp = self.experiment(team)
        report = exp.run()
        self.assertEqual(report['status'], 'PASS', report)
        self.assertEqual(len(report['rounds']), 2, report)
        first, repaired = report['rounds']
        self.assertEqual({t['test_case_id'] for t in first['tests']}, {'R001-T01', 'R001-T02'})
        self.assertEqual({t['test_case_id'] for t in repaired['tests']}, {'R001-T01', 'R001-T02'})
        self.assertEqual([t['execution']['stdout'] for t in first['tests']], ['0\n', '0\n'])
        self.assertEqual([t['execution']['stdout'] for t in repaired['tests']], ['5\n', '11\n'])
        self.assertNotEqual(first['source_sha256'], repaired['source_sha256'])
        self.assertTrue(team.last_feedback, 'The repair must receive recorded failure feedback.')
        self.assertEqual((exp.app / 'main.py').read_text(), CORRECT)
        for round_record in report['rounds']:
            for test in round_record['tests']:
                self.assertEqual(test['requirement_id'], 'R001')
                self.assertEqual(test['execution']['source_sha256'], round_record['source_sha256'])
                self.assertEqual(test['execution']['return_code'], 0)
        self.assertTrue(any(role == 'requirements-engineer' for role, _, _ in team.calls))
        self.assertTrue(any(role == 'test-engineer' for role, _, _ in team.calls))
        self.assertTrue(any(role == 'coder' for role, _, _ in team.calls))
        self.assertTrue(any(role == 'verification-engineer' for role, _, _ in team.calls))

    def test_compile_failure_is_repaired_before_any_test_execution(self):
        team = ScriptedTeam(program=lambda number: 'def broken(:\n' if number == 0 else CORRECT)
        report = self.experiment(team, max_repairs=1).run()
        self.assertEqual(report['status'], 'PASS', report)
        initial, repaired = report['rounds']
        self.assertFalse(initial['build']['success'])
        self.assertTrue(initial['build']['stderr'])
        self.assertFalse(initial['tests'])
        self.assertTrue(repaired['build']['success'])
        self.assertEqual(len(repaired['tests']), 2)
        self.assertTrue(any(item['kind'] == 'BUILD_FAILURE' for item in team.last_feedback))
        self.assertEqual({number for number, _ in team.execution_rounds}, {1})

    def test_rejected_test_plan_is_corrected_and_accepted_before_coding_then_frozen(self):
        team = PlanReviewTeam()
        exp = self.experiment(team, max_repairs=0)
        team.experiment = exp
        report = exp.run()
        self.assertEqual(report['status'], 'PASS', report)
        self.assertEqual(team.design_attempts, 2)
        self.assertEqual(team.review_attempts, 2)
        self.assertTrue(team.accepted)
        plan_path = exp.root / 'test_plan.json'
        self.assertEqual(plan_path.read_bytes(), team.plan_before_coding)
        plan = json.loads(plan_path.read_text())
        self.assertEqual([test['expected_stdout'] for test in plan['tests']], ['5\n', '11\n'])
        self.assertEqual(report['test_count'], 2)
        self.assertEqual(len(report['rounds']), 1)

    def test_repeated_independent_test_plan_rejection_stops_before_coding(self):
        team = PlanReviewTeam(always_revise=True)
        exp = self.experiment(team)
        team.experiment = exp
        report = exp.run()
        self.assertEqual(report['status'], 'INCOMPLETE', report)
        self.assertEqual(team.design_attempts, 3)
        self.assertEqual(team.review_attempts, 3)
        self.assertFalse(team.accepted)
        self.assertFalse(report['rounds'])
        self.assertFalse(team.execution_rounds)
        self.assertFalse((exp.app / 'main.py').exists())
        self.assertFalse((exp.root / 'test_plan.json').exists())

    def test_max_repairs_bounds_failed_runs(self):
        team = ScriptedTeam(always_broken=True)
        report = self.experiment(team, max_repairs=1).run()
        self.assertEqual(report['status'], 'FAIL', report)
        self.assertEqual(len(report['rounds']), 2)
        self.assertEqual(len(team.written_rounds), 2)
        self.assertTrue(all(t['execution']['stdout'] == '0\n'
                            for r in report['rounds'] for t in r['tests']))

    def test_verifier_cannot_turn_observed_failure_into_pass(self):
        team = ScriptedTeam(always_broken=True, always_pass=True)
        report = self.experiment(team, max_repairs=0, max_turns=3).run()
        self.assertNotEqual(report['status'], 'PASS', report)
        self.assertTrue(report['rounds'], report)
        self.assertTrue(any(t['execution']['stdout'] == '0\n'
                            for r in report['rounds'] for t in r.get('tests', [])), report)

    def test_clarification_stops_before_application_is_executed(self):
        team = ScriptedTeam(clarify=True)
        report = self.experiment(team).run()
        self.assertEqual(report['status'], 'NEEDS_USER', report)
        self.assertFalse(report['rounds'])
        self.assertFalse(team.execution_rounds)
        self.assertFalse((self.output / 'main.py').exists())

    def test_requirements_must_quote_the_actual_customer_prompt(self):
        team = ScriptedTeam(invalid_source_quote=True)
        report = self.experiment(team, max_turns=2, max_model_calls=4).run()
        self.assertNotEqual(report['status'], 'PASS', report)
        self.assertFalse(team.execution_rounds)
        self.assertFalse(team.coding_rounds)

    def test_model_turn_limit_stops_a_coder_that_never_finishes(self):
        team = ScriptedTeam(never_finish=True)
        report = self.experiment(team, max_turns=3, max_model_calls=20).run()
        self.assertIn(report['status'], ('INCOMPLETE', 'ERROR'), report)
        self.assertFalse(team.execution_rounds)
        self.assertLessEqual(len(team.coding_rounds), 3)

    def test_run_wide_model_call_limit_applies_across_agent_roles(self):
        team = ScriptedTeam()
        report = self.experiment(team, max_model_calls=2).run()
        self.assertIn(report['status'], ('INCOMPLETE', 'ERROR'), report)
        self.assertLessEqual(len(team.calls), 2)
        self.assertFalse(team.execution_rounds)

    def test_coder_writes_are_confined_to_application_sources(self):
        exp = self.experiment(ScriptedTeam())
        outside = Path(self.temp.name) / 'outside.py'
        for path in ('../outside.py', str(outside), '../report.json', 'main.sh'):
            with self.subTest(path=path):
                with self.assertRaises((ValueError, RuntimeError, PermissionError)):
                    exp.write_file(path, 'print(1)\n', 'replace')
        self.assertFalse(outside.exists())

    @unittest.skipUnless(shutil.which('gcc'), 'C adapter requires gcc')
    def test_c_adapter_compiles_and_executes_frozen_scripted_stdin_tests(self):
        prompt = ('Build a C program that reads two integers from standard input and prints '
                  'their sum as an integer followed by a newline, then exits 0.')
        requirements = [{
            'requirement_id': 'R001',
            'text': 'The program shall read two integers from stdin and print their sum.',
            'source_quote': 'reads two integers from standard input',
        }]
        tests = [{**test, 'args': [], 'stdin': stdin} for test, stdin in zip(TESTS, ('2 3\n', '4 7\n'))]
        source = ('#include <stdio.h>\nint main(void) { long a,b; '
                  'if (scanf("%ld %ld", &a, &b) != 2) return 1; '
                  'printf("%ld\\n", a+b); return 0; }\n')
        team = ScriptedTeam(program=source, source_path='main.c', requirements=requirements, tests=tests)
        exp = Experiment(Config(prompt=prompt, output=self.output, language='c', max_repairs=0), transport=team)
        report = exp.run()
        self.assertEqual(report['status'], 'PASS', report)
        self.assertEqual(len(report['rounds']), 1)
        self.assertTrue(report['rounds'][0]['build']['success'])
        executions = [test['execution'] for test in report['rounds'][0]['tests']]
        self.assertEqual([e['stdin'] for e in executions], ['2 3\n', '4 7\n'])
        self.assertEqual([e['stdout'] for e in executions], ['5\n', '11\n'])
        self.assertTrue(all(e['return_code'] == 0 for e in executions))
        self.assertTrue((exp.root / 'rounds/000/source/program').is_file())

    def test_timed_out_application_is_recorded_and_cannot_pass(self):
        team = ScriptedTeam(program='while True:\n    pass\n')
        report = self.experiment(team, max_repairs=0, timeout=0.05).run()
        self.assertEqual(report['status'], 'NEEDS_USER', report)
        executions = [test['execution'] for test in report['rounds'][0]['tests']]
        self.assertEqual(len(executions), 2)
        self.assertTrue(all(e['execution_status'] == 'TIMEOUT' for e in executions), executions)
        self.assertTrue(all(e['return_code'] < 0 and not e['host_match'] for e in executions), executions)

    def test_excessive_application_output_is_bounded_and_cannot_pass(self):
        team = ScriptedTeam(program="print('x' * 2_000_000)\n")
        report = self.experiment(team, max_repairs=0).run()
        self.assertEqual(report['status'], 'NEEDS_USER', report)
        executions = [test['execution'] for test in report['rounds'][0]['tests']]
        self.assertTrue(all(e['output_truncated'] for e in executions), executions)
        self.assertTrue(all(len(e['stdout'].encode()) <= 1048576 for e in executions))
        self.assertTrue(all(not e['host_match'] for e in executions))

    def test_each_execution_uses_an_isolated_copy_of_the_frozen_source_snapshot(self):
        source = ("from pathlib import Path\nimport sys\n"
                  "Path(__file__).write_text('print(999)\\n')\n"
                  "print(sum(int(part) for part in sys.argv[1].split('+')))\n")
        team = ScriptedTeam(program=source)
        exp = self.experiment(team, max_repairs=0)
        report = exp.run()
        self.assertEqual(report['status'], 'PASS', report)
        self.assertEqual((exp.app / 'main.py').read_text(), source)
        snapshot = exp.root / 'rounds/000/source/main.py'
        self.assertEqual(snapshot.read_text(), source)
        expected_hash = hashlib.sha256(source.encode()).hexdigest()
        round_record = report['rounds'][0]
        self.assertEqual(round_record['source_sha256']['main.py'], expected_hash)
        executions = [test['execution'] for test in round_record['tests']]
        self.assertEqual([e['stdout'] for e in executions], ['5\n', '11\n'])
        self.assertTrue(all(e['source_sha256']['main.py'] == expected_hash for e in executions))
        self.assertNotEqual(executions[0]['execution_id'], executions[1]['execution_id'])

    @unittest.skipUnless(os.name == 'posix' and shutil.which('bwrap'), 'Read-only host isolation requires bwrap')
    def test_generated_application_cannot_write_to_host_files(self):
        # Place the sentinel outside /tmp, which is separately hidden by bwrap.
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            target = Path(directory) / 'host-sentinel.txt'
            target.write_text('original')
            source = ('from pathlib import Path\nimport errno, sys\n'
                      f'target = Path({str(target)!r})\n'
                      "assert target.read_text() == 'original'\n"
                      'try:\n'
                      "    target.write_text('tampered')\n"
                      'except OSError as exc:\n'
                      '    assert exc.errno in (errno.EROFS, errno.EACCES, errno.EPERM)\n'
                      'else:\n'
                      "    raise RuntimeError('host write unexpectedly succeeded')\n"
                      "print(sum(int(part) for part in sys.argv[1].split('+')))\n")
            report = self.experiment(ScriptedTeam(program=source), max_repairs=0).run()
            self.assertEqual(report['status'], 'PASS', report)
            self.assertEqual(target.read_text(), 'original')
            executions = [test['execution'] for test in report['rounds'][0]['tests']]
            self.assertTrue(all(e['isolation'] == 'bwrap' for e in executions))
            self.assertEqual([e['stdout'] for e in executions], ['5\n', '11\n'])

    def test_http_truncated_tool_response_is_not_executed_and_retry_context_is_compact(self):
        team = TruncatedCoder()
        with model_http_server(team) as endpoint:
            exp = Experiment(Config(prompt=PROMPT, output=self.output, endpoints=[endpoint] * 4))
            team.experiment = exp
            summary = exp.coding(0, [])
        self.assertTrue(summary)
        self.assertEqual(len(team.requests), 3)
        self.assertEqual((exp.app / 'main.py').read_text(), CORRECT)
        self.assertLess(len(json.dumps(team.requests[1])), 4000)
        self.assertEqual(exp.model_calls, 3)

    def test_http_repeated_truncation_stops_at_turn_limit_without_writing_source(self):
        team = TruncatedCoder(always=True)
        with model_http_server(team) as endpoint:
            exp = Experiment(Config(prompt=PROMPT, output=self.output, endpoints=[endpoint] * 4,
                                    max_turns=3, max_model_calls=10))
            team.experiment = exp
            with self.assertRaises(BudgetExceeded):
                exp.coding(0, [])
        self.assertEqual(len(team.requests), 3)
        self.assertEqual(exp.model_calls, 3)
        self.assertFalse((exp.app / 'main.py').exists())

    def test_read_once_coder_is_guided_to_change_or_finish_and_noop_writes_do_not_reenable_reads(self):
        requests = []
        def read_first_coder(role, body):
            state = json.loads(body['messages'][0]['content'])
            tools = {tool['function']['name'] for tool in body['tools']}
            requests.append(state)
            step = len(requests)
            if step in (1, 5):
                self.assertIn('read_file', tools)
                return completion('read_file', {'path': 'main.py', 'offset': 0})
            self.assertNotIn('read_file', tools, 'Unchanged source must not be offered for repeated reading')
            if step == 2:
                self.assertEqual(state['source_pages'][0]['content'], BROKEN)
                return completion('write_file', {'path': 'main.py', 'content': BROKEN, 'mode': 'replace'})
            if step == 3:
                self.assertIn('no source change', state['previous_action']['result']['error'].lower())
                return completion('write_file', {'path': 'main.py', 'content': '', 'mode': 'append'})
            if step == 4:
                # An empty array is rejected by the tool schema before writing.
                self.assertTrue(state['previous_action']['result'].get('error'))
                self.assertEqual((exp.app / 'main.py').read_text(), BROKEN)
                return completion('write_file', {'path': 'main.py', 'content': CORRECT, 'mode': 'replace'})
            self.assertEqual(step, 6, 'Coder consumed excess turns without progressing')
            self.assertEqual(state['source_pages'][0]['content'], CORRECT)
            return completion('finish_build', {'summary': 'Read changed source once; implementation is ready.'})

        exp = self.experiment(read_first_coder, max_turns=8)
        exp.write_file('main.py', BROKEN, 'replace')
        exp.coding(1, [])
        self.assertEqual(len(requests), 6)
        self.assertEqual(exp.model_calls, 6)
        self.assertEqual((exp.app / 'main.py').read_text(), CORRECT)
        self.assertEqual([action['tool'] for action in requests[-1]['recent_actions']],
                         ['read_file', 'write_file', 'write_file', 'write_file', 'read_file'])
        self.assertTrue(all(len(state['recent_actions']) <= 6 for state in requests))

    def test_coder_retains_recent_source_pages_and_offers_only_unread_files(self):
        compute = 'from parser import parse\n'
        parser = 'def parse(expression):\n    return expression.strip()\n'
        corrected = compute + "print(parse('1 + 2'))\n"

        def coder(role, body):
            self.assertEqual(role, 'coder')
            state = json.loads(body['messages'][0]['content'])
            tools = {tool['function']['name']: tool['function'] for tool in body['tools']}
            read_tool = tools.get('read_file')
            readable = set(read_tool['parameters']['properties']['path']['enum']) if read_tool else set()
            pages = {page['path']: page['content'] for page in state.get('source_pages', [])}
            step = state['step']
            if step == 1:
                self.assertEqual(readable, {'compute.py', 'parser.py'})
                return completion('read_file', {'path': 'compute.py', 'offset': 0})
            if step == 2:
                self.assertEqual(readable, {'parser.py'})
                self.assertEqual(pages, {'compute.py': compute})
                return completion('read_file', {'path': 'parser.py', 'offset': 0})
            if step == 3:
                self.assertEqual(readable, set())
                self.assertEqual(pages, {'compute.py': compute, 'parser.py': parser})
                previous_read = state['previous_action']['result']
                self.assertEqual(previous_read['path'], 'parser.py')
                self.assertIsNone(previous_read['next_offset'])
                self.assertNotIn('content', previous_read)
                self.assertEqual(json.dumps(state).count(json.dumps(parser)), 1)
                return completion('write_file', {'path': './compute.py', 'content': corrected, 'mode': 'replace'})
            if step == 4:
                self.assertEqual(readable, {'compute.py'})
                self.assertEqual(pages, {'parser.py': parser})
                return completion('read_file', {'path': 'compute.py', 'offset': 0})
            self.assertEqual(step, 5)
            self.assertEqual(readable, set())
            self.assertEqual(pages, {'compute.py': corrected, 'parser.py': parser})
            return completion('finish_build', {'summary': 'Both source files inspected.'})

        exp = self.experiment(coder, entrypoint='compute.py')
        exp.write_file('compute.py', compute, 'replace')
        exp.write_file('parser.py', parser, 'replace')
        exp.coding(0, [])
        self.assertEqual(exp.model_calls, 5)
        self.assertEqual((exp.app / 'compute.py').read_text(), corrected)
        trace = (json.loads(line) for line in (exp.root / 'trace.jsonl').read_text().splitlines())
        reads = [(event['result']['path'], event['result']['content']) for event in trace
                 if event.get('tool') == 'read_file']
        self.assertEqual(reads, [('compute.py', compute), ('parser.py', parser), ('compute.py', corrected)])

    def test_output_comparison_ignores_only_one_terminal_line_ending_and_preserves_raw_evidence(self):
        cases = [
            ('5\n', '5', True),
            ('5\r\n', '5\n', True),
            ('5', '5\n', True),
            ('5\n\n', '5\n', False),
            ('5\n', '5\n\n', False),
            (' 5\n', '5\n', False),
            ('5 \n', '5\n', False),
            ('5\n6\n', '56\n', False),
            ('5\n6\n', '5\n6', True),
            ('5\r', '5', False),
        ]
        exp = self.experiment(ScriptedTeam())
        for index, (actual, expected, match) in enumerate(cases):
            with self.subTest(actual=actual, expected=expected):
                source = f'import sys\nsys.stdout.write({actual!r})\n# case {index}\n'
                exp.write_file('main.py', source, 'replace')
                test = {**TESTS[0], 'requirement_id': 'R001', 'expected_stdout': expected}
                hashes = exp.hashes(exp.app)
                evidence = exp.execute(test, exp.app, hashes)
                self.assertEqual(evidence['host_match'], match, evidence)
                self.assertEqual(evidence['stdout'], actual, 'Raw output must not be normalized in evidence')
                self.assertEqual(evidence['return_code'], 0)
                self.assertEqual(evidence['source_sha256'], hashes)

    def test_repeated_large_invalid_test_submissions_do_not_accumulate_in_model_context(self):
        contexts = []
        large_test = {**TESTS[0], 'args': ['a' * 1900] * 20,
                      'expected_stdout': 'z' * 5000 + '\n', 'justification': 'j' * 600}
        def invalid_designer(role, body):
            contexts.append(json.loads(body['messages'][0]['content']))
            # Structurally valid, but duplicate IDs must be rejected before review.
            return completion('submit_tests', {'tests': [large_test, dict(large_test)], 'rationale': 'r' * 600})
        exp = self.experiment(invalid_designer, max_turns=10)
        exp.requirements = REQUIREMENTS
        with self.assertRaises(BudgetExceeded):
            exp.design_tests()
        self.assertEqual(len(contexts), 10)
        self.assertLess(max(len(json.dumps(state)) for state in contexts), 10000)
        for state in contexts:
            self.assertLessEqual(len(state['recent_actions']), 6)
            for action in state['recent_actions']:
                self.assertNotIn('tests', action['arguments'])
                self.assertTrue(all(not isinstance(value, (list, dict)) and len(str(value)) <= 120
                                    for value in action['arguments'].values()))
        self.assertFalse(exp.tests)
        self.assertFalse((exp.root / 'test_plan.json').exists())
        self.assertFalse((exp.app / 'main.py').exists())

    def test_http_tokenization_reserves_context_and_reduces_output_budget_before_generation(self):
        tokenizations, generations = [], []
        metadata = {'data': [{'id': role, 'max_model_len': 4096} for role in (
            'requirements-engineer', 'test-engineer', 'coder', 'verification-engineer')]}
        def tokenize(body):
            tokenizations.append(body)
            return {'count': 3800, 'max_model_len': 4096}
        def clarify(role, body):
            generations.append(body)
            return completion('ask_user', {'question': 'Choose calculator formatting.',
                                           'reason': 'Formatting needs a user decision.'})
        with model_http_server(clarify, tokenize=tokenize, model_metadata=metadata) as endpoint:
            exp = Experiment(Config(prompt=PROMPT, output=self.output, endpoints=[endpoint] * 4,
                                    max_tokens=1024))
            report = exp.run()
        self.assertEqual(report['status'], 'NEEDS_USER', report)
        self.assertTrue(exp.use_tokenize)
        self.assertEqual(len(tokenizations), 1)
        self.assertEqual(set(tokenizations[0]), {'model', 'messages', 'tools'})
        self.assertEqual(len(generations), 1)
        self.assertEqual(generations[0]['max_tokens'], 264)
        self.assertTrue(all(tool['function']['strict'] for tool in generations[0]['tools']))
        self.assertTrue((exp.root / 'server_models.json').is_file())

    def test_http_tokenization_stops_before_generation_when_input_leaves_insufficient_context(self):
        generations = []
        metadata = {'data': [{'id': role, 'max_model_len': 4096} for role in (
            'requirements-engineer', 'test-engineer', 'coder', 'verification-engineer')]}
        def generation(role, body):
            generations.append(body)
            raise AssertionError('Generation must not be requested when only 64 tokens remain')
        with model_http_server(generation, tokenize=lambda body: {'count': 4000, 'max_model_len': 4096},
                               model_metadata=metadata) as endpoint:
            exp = Experiment(Config(prompt=PROMPT, output=self.output, endpoints=[endpoint] * 4))
            report = exp.run()
        self.assertEqual(report['status'], 'INCOMPLETE', report)
        self.assertIn('4,000 input tokens in a 4,096-token context', report['reason'])
        self.assertIn('less than 128 response tokens', report['reason'])
        self.assertFalse(generations)
        self.assertFalse(report['rounds'])
        self.assertFalse((exp.app / 'main.py').exists())

    def test_coder_line_array_preserves_indentation_and_builds_an_executable_calculator(self):
        source = ("import sys\n"
                  "def add(expression):\n"
                  "    left, right = expression.split('+')\n"
                  "    return int(left) + int(right)\n"
                  "if __name__ == '__main__':\n"
                  "    print(add(sys.argv[1]))\n")
        team = ScriptedTeam(program=source)
        exp = self.experiment(team, max_repairs=0)
        report = exp.run()
        self.assertEqual(report['status'], 'PASS', report)
        self.assertEqual((exp.app / 'main.py').read_text(), source)
        self.assertEqual([test['execution']['stdout'] for test in report['rounds'][0]['tests']],
                         ['5\n', '11\n'])
        events = [json.loads(line) for line in (exp.root / 'trace.jsonl').read_text().splitlines()]
        written = []
        for event in events:
            response = event.get('response', {})
            for choice in response.get('choices', []):
                for call in choice['message'].get('tool_calls') or []:
                    if call['function']['name'] == 'write_file':
                        written.append(json.loads(call['function']['arguments'])['content'])
        self.assertEqual(written, [source.splitlines()])
        self.assertEqual(written[0][2], "    left, right = expression.split('+')")

    def test_basic_calculator_profile_freezes_only_valid_cli_cases_with_computed_outputs(self):
        expression = '(5 + 3) * (2 + 4)'
        requirement_text = 'Support arithmetic expressions with parentheses.'
        prompt = (f'{requirement_text}\n\nExamples of required behavior:\n'
                  f'"{expression}" produces "48".\nGenerate and review test plans before coding.')
        test = {'test_case_id': 'R001-T01', 'args': [expression],
                'justification': 'Checks grouping before multiplication.'}
        proposals = [
            {**test, 'stdin': expression},
            {**test, 'args': ['(5 + 3)', '*', '(2 + 4)']},
            {**test, 'args': ['1 / 3']},
            test,
        ]
        stages = []

        def team(role, body):
            state = json.loads(body['messages'][0]['content'])
            stages.append(state['mode'])
            if state['mode'] == 'requirements':
                self.assertEqual(role, 'requirements-engineer')
                self.assertEqual(state['full_prompt_context'], f'[S001] {requirement_text}')
                return completion('submit_requirements', {'requirements': [
                    {'requirement_id': 'R001', 'text': requirement_text, 'source_id': 'S001'},
                ]})
            self.assertEqual(state['customer_prompt'], prompt)
            if state['mode'] == 'test_design':
                self.assertEqual(role, 'test-engineer')
                if state['step'] > 1:
                    self.assertIn('error', state['previous_action']['result'])
                return completion('submit_tests', {'tests': [proposals[state['step'] - 1]],
                                                   'rationale': 'Check exact arithmetic on one argument.'})
            self.assertEqual(role, 'verification-engineer')
            self.assertEqual(state['tests'], [{**test, 'stdin': '', 'expected_stdout': '48\n',
                                               'expected_exit_code': 0, 'requirement_id': 'R001'}])
            check, = state['mathematical_reference_checks']
            self.assertEqual(check['reference']['expected_stdout'], '48\n')
            return completion('review_tests', {'decision': 'ACCEPT', 'feedback': 'Covers precedence.'})

        exp = Experiment(Config(prompt=prompt, output=self.output,
                                calculator_basic_reference=True), transport=team)
        exp.intake()
        exp.design_tests()

        self.assertEqual(stages, ['requirements'] + ['test_design'] * 4 + ['test_review'])
        self.assertEqual(exp.requirements[0]['source_quote'], requirement_text)
        plan = json.loads((exp.root / 'test_plan.json').read_text())
        self.assertEqual(plan['tests'], [{**test, 'stdin': '', 'expected_stdout': '48\n',
                                          'expected_exit_code': 0, 'requirement_id': 'R001'}])

    def test_reference_profile_rejects_wrong_17_digit_values_then_allows_correction_before_freezing(self):
        team = ReferenceTeam()
        exp = Experiment(Config(prompt=team.prompt, output=self.output, calculator_reference=True,
                                max_repairs=0), transport=team)
        report = exp.run()
        self.assertEqual(report['status'], 'PASS', report)
        self.assertEqual(team.submissions, 2)
        self.assertTrue(any('contradict' in error for error in team.rejections))
        self.assertEqual(team.reference_roles, ['test-engineer', 'verification-engineer'])
        self.assertEqual(team.design_reference['expected_stdout'], '1.414213562373095\n')
        plan = json.loads((exp.root / 'test_plan.json').read_text())
        self.assertEqual(plan['tests'][0]['expected_stdout'], '1.414213562373095\n')
        self.assertEqual(report['rounds'][0]['tests'][0]['execution']['stdout'], '1.414213562373095\n')
        checks = [check for path in (exp.root / 'test_reference/R001').glob('*.json')
                  for check in json.loads(path.read_text())]
        self.assertEqual(sorted(check['matches'] for check in checks), [False, True])
        self.assertTrue(json.loads((exp.root / 'manifest.json').read_text())['calculator_reference'])

    def test_reference_profile_prevents_an_accepting_reviewer_from_freezing_wrong_math(self):
        team = ReferenceTeam(always_wrong=True)
        exp = Experiment(Config(prompt=team.prompt, output=self.output, calculator_reference=True,
                                max_turns=3), transport=team)
        report = exp.run()
        self.assertEqual(report['status'], 'INCOMPLETE', report)
        self.assertEqual(team.submissions, 3)
        self.assertEqual(team.review_count, 0, 'An incorrect numerical proposal must not reach acceptance')
        self.assertFalse(report['rounds'])
        self.assertFalse((exp.root / 'test_plan.json').exists())
        self.assertFalse((exp.app / 'main.py').exists())
        paths = list((exp.root / 'test_reference/R001').glob('*.json'))
        self.assertEqual(len(paths), 3)
        self.assertTrue(all(not check['matches'] for path in paths for check in json.loads(path.read_text())))

    def test_reference_profile_requires_invalid_input_coverage_for_error_handling_requirements(self):
        contexts = []
        requirement = {'requirement_id': 'R001', 'text': 'The calculator shall reject invalid input with an error response.',
                       'source_quote': 'invalid input'}
        prompt = 'Reject invalid input with Error: invalid input and exit 1.'
        def designer(role, body):
            context = json.loads(body['messages'][0]['content'])
            tools = {tool['function']['name'] for tool in body['tools']}
            contexts.append(context)
            if 'submit_tests' in tools:
                test = dict(TESTS[0]) if context['previous_action'] is None else {
                    **TESTS[0], 'args': ['2++'], 'expected_stdout': 'Error: invalid input\n', 'expected_exit_code': 1}
                return completion('submit_tests', {'tests': [test], 'rationale': 'Check required invalid-input handling.'})
            return completion('review_tests', {'decision': 'ACCEPT', 'feedback': 'An invalid expression is exercised.'})
        exp = Experiment(Config(prompt=prompt, output=self.output, calculator_reference=True), transport=designer)
        exp.requirements = [requirement]
        exp.design_tests()
        self.assertEqual(len(contexts), 3)
        self.assertTrue(contexts[1]['previous_action']['result'].get('error'))
        self.assertIn('INVALID_INPUT', contexts[1]['test_kind'])
        self.assertEqual(exp.tests[0]['args'], ['2++'])
        self.assertEqual(exp.tests[0]['expected_exit_code'], 1)
        self.assertTrue((exp.root / 'test_plan.json').is_file())

    def test_complete_source_handoff_runs_real_tests_without_an_extra_finish_call(self):
        team = ScriptedTeam(program=CORRECT)
        exp = self.experiment(team, max_repairs=0)
        report = exp.run()
        self.assertEqual(report['status'], 'PASS', report)
        self.assertEqual(len(team.coding_rounds), 1)
        coder_calls = [call for call in team.calls if call[0] == 'coder']
        self.assertEqual(len(coder_calls), 1)
        events = [json.loads(line) for line in (exp.root / 'trace.jsonl').read_text().splitlines()]
        tool_events = [event['tool'] for event in events if event.get('role') == 'coder' and 'tool' in event]
        self.assertEqual(tool_events, ['write_file'])
        self.assertEqual([test['execution']['stdout'] for test in report['rounds'][0]['tests']], ['5\n', '11\n'])

    def test_observed_failure_is_repaired_while_raw_inconclusive_model_assessment_is_preserved(self):
        raw = {'verdict': 'INCONCLUSIVE', 'criteria_justified': False,
               'explanation': 'I am uncertain about the criteria even though the process completed.'}
        class UncertainVerifier(ScriptedTeam):
            def __call__(self, role, body):
                context = json.loads(body['messages'][0]['content'])
                tools = {tool['function']['name'] for tool in body['tools']}
                if 'submit_verdict' in tools and not context['execution']['host_match']:
                    return completion('submit_verdict', raw)
                return super().__call__(role, body)
        team = UncertainVerifier()
        exp = self.experiment(team, max_repairs=1)
        report = exp.run()
        self.assertEqual(report['status'], 'PASS', report)
        self.assertEqual(len(report['rounds']), 2)
        first, repaired = report['rounds']
        self.assertTrue(all(test['verdict'] == raw for test in first['tests']))
        self.assertTrue(all(test['controller_assessment']['verdict'] == 'FAIL' for test in first['tests']))
        self.assertEqual([test['execution']['stdout'] for test in first['tests']], ['0\n', '0\n'])
        self.assertEqual([test['execution']['stdout'] for test in repaired['tests']], ['5\n', '11\n'])
        self.assertEqual(team.last_feedback[0]['model_assessment'], raw)
        self.assertEqual(team.last_feedback[0]['assessment']['verdict'], 'FAIL')
        archived = json.loads((exp.root / 'rounds/000/verdicts/R001-T01.json').read_text())
        self.assertEqual(archived, raw)
        self.assertFalse(report['questions'])

    def test_function_entry_mode_invokes_main_once_preserves_negative_args_and_honors_integer_return(self):
        prompt = ('Build a Python calculator that adds two integers supplied as one expression argument. '
                  'Print the integer result and a newline, then exit with status 7.')
        requirements = [{**REQUIREMENTS[0],
                         'text': 'The calculator shall add two integer operands and exit with status 7.'}]
        tests = [{**TESTS[0], 'expected_exit_code': 7},
                 {**TESTS[1], 'args': ['-4+7'], 'expected_stdout': '3\n', 'expected_exit_code': 7}]
        source = ('import sys\n'
                  'calls = 0\n'
                  'def main():\n'
                  '    global calls\n'
                  '    calls += 1\n'
                  '    if calls != 1: raise RuntimeError("main called more than once")\n'
                  '    left, right = sys.argv[1].split("+")\n'
                  '    print(int(left) + int(right))\n'
                  '    return 7\n')
        team = ScriptedTeam(program=source, requirements=requirements, tests=tests)
        exp = Experiment(Config(prompt=prompt, output=self.output, python_entry_function='main',
                                max_repairs=0), transport=team)
        report = exp.run()
        self.assertEqual(report['status'], 'PASS', report)
        self.assertEqual(len(report['rounds']), 1)
        executions = [test['execution'] for test in report['rounds'][0]['tests']]
        self.assertEqual([record['stdout'] for record in executions], ['5\n', '3\n'])
        self.assertEqual([record['return_code'] for record in executions], [7, 7])
        self.assertEqual(executions[1]['args'], ['-4+7'])
        self.assertTrue(all('--function' in record['command'] and 'main' in record['command']
                            for record in executions))
        self.assertEqual((exp.app / 'main.py').read_text(), source)
        manifest = json.loads((exp.root / 'manifest.json').read_text())
        self.assertEqual(manifest['python_entry_function'], 'main')
        self.assertEqual(manifest['python_launcher_sha256'],
                         hashlib.sha256((exp.root / 'run_python_app.py').read_bytes()).hexdigest())

    def test_function_entry_mode_reports_missing_main_as_build_feedback_then_repairs(self):
        missing = 'def calculate():\n    print(5)\n    return 0\n'
        fixed = ('import sys\n'
                 'def main():\n'
                 '    print(sum(int(part) for part in sys.argv[1].split("+")))\n'
                 '    return 0\n')
        team = ScriptedTeam(program=lambda number: missing if number == 0 else fixed)
        exp = self.experiment(team, python_entry_function='main', max_repairs=1)
        report = exp.run()
        self.assertEqual(report['status'], 'PASS', report)
        initial, repaired = report['rounds']
        self.assertFalse(initial['build']['success'])
        self.assertIn('top-level main()', initial['build']['stderr'])
        self.assertFalse(initial['tests'])
        self.assertTrue(repaired['build']['success'])
        self.assertEqual([test['execution']['stdout'] for test in repaired['tests']], ['5\n', '11\n'])
        self.assertEqual({number for number, _ in team.execution_rounds}, {1})
        self.assertEqual(team.last_feedback[0]['kind'], 'BUILD_FAILURE')
        self.assertIn('main()', team.last_feedback[0]['stderr'])
        self.assertEqual((exp.root / 'rounds/000/source/main.py').read_text(), missing)
        self.assertEqual((exp.root / 'rounds/001/source/main.py').read_text(), fixed)

    def test_source_symlink_cannot_escape_application_directory(self):
        exp = self.experiment(ScriptedTeam())
        exp.app.mkdir(parents=True, exist_ok=True)
        outside = Path(self.temp.name) / 'outside.py'
        outside.write_text('original\n')
        (exp.app / 'main.py').symlink_to(outside)
        with self.assertRaises((ValueError, RuntimeError, PermissionError)):
            exp.write_file('main.py', 'changed\n', 'replace')
        self.assertEqual(outside.read_text(), 'original\n')


if __name__ == '__main__':
    unittest.main()
