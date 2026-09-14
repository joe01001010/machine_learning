"""Contract tests use scripted model responses but real calculator subprocesses."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest

from validate_requirements import Pipeline


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.calc = self.root / 'calc' / 'compute.py'
        self.calc.parent.mkdir()
        self.calc.write_text('import sys\nprint(8)\n', encoding='utf-8')
        reqs = self.root / 'requirements.jsonl'
        reqs.write_text(json.dumps({'requirement_id': 'REQ-001', 'text': 'Add integers'})+'\n', encoding='utf-8')
        scope = self.root / 'scope.txt'
        scope.write_text('Integer addition, exact string output.', encoding='utf-8')
        self.args = argparse.Namespace(calculator=self.calc, requirements=reqs, scope=scope,
            output=self.root/'results', endpoints=['unused']*3, temperature=0, max_tokens=768,
            max_turns=12, http_timeout=1, calculator_timeout=.5)
        self.called = []
        self.decision = 'ACCEPT'
        self.verdict = 'PASS'

    def transport(self, role, body):
        self.assertFalse(any(m['role'] == 'system' for m in body['messages']))
        fn = body['tools'][0]['function']
        name = fn['name']
        self.called.append(name)
        args = {}
        if name == 'submit_to_test_engineer':
            args = {'decision': self.decision, 'rationale': 'Reviewed scope.', 'tests': []}
            if self.decision == 'ACCEPT':
                args['tests'] = [{'test_case_id': 'REQ-001-T01', 'input': '5 + 3',
                                  'expected_output': '8', 'justification': 'Checks addition.'}]
        elif name == 'run_calculator':
            args = {'test_case_id': fn['parameters']['properties']['test_case_id']['enum'][0]}
        elif name == 'submit_verdict':
            args = {'criteria_justified': True, 'verdict_value': self.verdict, 'explanation': 'Reviewed execution.'}
        return self.response(name, args)

    @staticmethod
    def response(name, args):
        return {'choices': [{'finish_reason': 'tool_calls', 'message': {'role': 'assistant',
            'content': None, 'tool_calls': [{'id': 'call_1', 'type': 'function',
            'function': {'name': name, 'arguments': json.dumps(args)}}]}}]}

    def test_full_chain_and_real_evidence(self):
        p = Pipeline(self.args, self.transport)
        report = p.run()
        self.assertEqual(self.called, ['create_workspace', 'submit_to_test_engineer',
                         'run_calculator', 'verification_engineer', 'submit_verdict'])
        evidence = json.loads((p.pool/'evidence/REQ-001-T01.json').read_text())
        self.assertEqual(evidence['stdout'], '8\n')
        self.assertEqual(evidence['input'], '5 + 3')
        v = report['requirements'][0]['verdicts'][0]
        self.assertEqual(v['execution_id'], evidence['execution_id'])
        self.assertEqual(v['test_case_id'], evidence['test_case_id'])
        self.assertEqual(report['workflow_status'], 'COMPLETED')
        self.assertEqual(json.loads(p.report_path.read_text()), report)

    def test_rejected_requirement_never_executes(self):
        self.decision = 'REJECT'
        report = Pipeline(self.args, self.transport).run()
        self.assertNotIn('run_calculator', self.called)
        self.assertEqual(report['requirements'][0]['workflow_status'], 'NOT_TESTED')

    def test_needs_clarification_never_executes(self):
        self.decision = 'NEEDS_CLARIFICATION'
        Pipeline(self.args, self.transport).run()
        self.assertNotIn('run_calculator', self.called)

    def test_crash_recorded_as_real_software_error(self):
        self.calc.write_text('raise ZeroDivisionError("division by zero")\n')
        self.verdict = 'FAIL'
        p = Pipeline(self.args, self.transport)
        p.run()
        e = json.loads((p.pool/'evidence/REQ-001-T01.json').read_text())
        self.assertEqual(e['execution_status'], 'ERROR')
        self.assertNotEqual(e['return_code'], 0)
        self.assertIn('ZeroDivisionError', e['stderr'])

    def test_timeout_is_recorded(self):
        self.calc.write_text('import time\ntime.sleep(10)\n')
        self.verdict = 'INCONCLUSIVE'
        p = Pipeline(self.args, self.transport)
        p.run()
        e = json.loads((p.pool/'evidence/REQ-001-T01.json').read_text())
        self.assertEqual(e['execution_status'], 'TIMEOUT')

    def test_unauthorized_tool_never_dispatches(self):
        self.args.max_turns = 2
        def bad(role, body):
            if body['tools'][0]['function']['name'] == 'create_workspace':
                return self.transport(role, body)
            return self.response('run_shell', {'command': 'fake'})
        p = Pipeline(self.args, bad)
        with self.assertRaises(RuntimeError):
            p.run()
        self.assertEqual(json.loads(p.report_path.read_text())['workflow_status'], 'ERROR')
        self.assertFalse((p.pool/'evidence').exists())

    def test_duplicate_requirement_ids_rejected(self):
        line = self.args.requirements.read_text()
        self.args.requirements.write_text(line + line)
        with self.assertRaises(ValueError):
            Pipeline(self.args, self.transport)

    def test_downstream_failure_preserves_evidence_and_incomplete_status(self):
        def broken(role, body):
            if role == 'verification-engineer':
                raise RuntimeError('simulated unavailable server')
            return self.transport(role, body)
        p = Pipeline(self.args, broken)
        with self.assertRaises(RuntimeError):
            p.run()
        report = json.loads(p.report_path.read_text())
        self.assertEqual(report['not_completed'], ['REQ-001'])
        self.assertTrue((p.pool/'evidence/REQ-001-T01.json').exists())

    def test_source_change_aborts_execution(self):
        p = Pipeline(self.args, self.transport)
        p.create_workspace()
        self.calc.write_text('print(9)\n')
        with self.assertRaises(RuntimeError):
            p.execute({'requirement_id':'REQ-001'}, {'test_case_id':'REQ-001-T01', 'input':'5 + 3'})


if __name__ == '__main__':
    unittest.main()
