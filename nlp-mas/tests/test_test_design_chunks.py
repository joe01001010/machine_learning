"""Regression checks for bounded, streamed test design after model truncation."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from experiment import BudgetExceeded, Config, Experiment


PROMPT = ('Build a C command-line rideshare application with scripted menu input. '
          'Preserve the specified console wording, whitespace, and session line endings.')
REQUIREMENT = {'requirement_id': 'R001',
               'text': 'The application shall support scripted menu sessions with the specified console output.',
               'source_quote': 'scripted menu input'}
STDIN_LINES = ['1', 'admin', 'a/b-1234567890', '', '  rider Zoë 🚗  ', '2', '4', '3.50', '0', 'quit']
STDOUT_LINES = [
    f'{i:02d} | passenger={i} | driver={i+1} | distance=3.50 | status=waiting | fare=12.75 | menu=1/2/3/4/0  '
    for i in range(1, 25)
]
STDOUT_LINES[5] = ''
STDOUT_LINES[14] = '  Saved passenger “Zoë” with a trailing space: '


def reply(name, arguments, *, truncated=False, unparseable=False):
    message = {'role': 'assistant', 'content': None, 'tool_calls': []}
    if unparseable:
        message['content'] = '{"tests":[{"stdin":"unfinished'
    else:
        message['tool_calls'] = [{'id': 'chunk-test-call', 'type': 'function', 'function': {
            'name': name, 'arguments': json.dumps(arguments)}}]
    return {'choices': [{'finish_reason': 'length' if truncated else 'tool_calls', 'message': message}]}


def metadata(args, code, justification, stdin_count, stdin_ending, stdin_final,
             stdout_count, stdout_ending, stdout_final):
    return {'args': args, 'expected_exit_code': code, 'justification': justification,
            'stdin_line_count': stdin_count, 'stdin_line_ending': stdin_ending,
            'stdin_final_newline': stdin_final, 'stdout_line_count': stdout_count,
            'stdout_line_ending': stdout_ending, 'stdout_final_newline': stdout_final}


def chunks(lines):
    return [('append_test_lines', {'lines': lines[offset:offset+8]}) for offset in range(0, len(lines), 8)]


def state_of(body):
    return json.loads(body['messages'][0]['content'])


class ScriptedDesigner:
    def __init__(self, actions, *, hook=None):
        self.actions = list(actions)
        self.hook = hook
        self.states = []
        self.reviewed = []
        self.experiment = None

    def __call__(self, role, body):
        state = state_of(body)
        tools = {tool['function']['name'] for tool in body['tools']}
        if state['mode'] == 'test_review':
            assert not self.actions, 'Independent review started before all test fields were assembled'
            assert not self.experiment.tests, 'Tests became final before independent review'
            assert not (self.experiment.root / 'test_plan.json').exists()
            self.reviewed.append(state)
            return reply('review_tests', {'decision': 'ACCEPT', 'feedback': 'The complete scripted session follows the request.'})
        assert state['mode'] == 'test_design', state
        assert self.actions, 'Designer was asked to repeat a completed test'
        index = len(self.states)
        self.states.append(copy.deepcopy(state))
        if self.hook:
            self.hook(index, state, tools)
        name, arguments, *options = self.actions.pop(0)
        assert name in tools, (name, tools, state)
        return reply(name, arguments, **(options[0] if options else {}))


class TestDesignChunkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.output = Path(self.temp.name) / 'results'

    def tearDown(self):
        self.temp.cleanup()

    def experiment(self, team, **options):
        exp = Experiment(Config(prompt=PROMPT, output=self.output, language='c', **options), transport=team)
        exp.requirements = [REQUIREMENT]
        team.experiment = exp
        return exp

    def test_streamed_sessions_preserve_exact_bytes_and_freeze_only_after_all_fields_and_review(self):
        poison = {'tests': [{'test_case_id': 'R001-T01', 'args': [], 'stdin': 'POISON',
                            'expected_stdout': 'POISON', 'expected_exit_code': 0,
                            'justification': 'Ignored truncated candidate.'}], 'rationale': 'Ignored.'}
        actions = [
            ('submit_tests', poison, {'truncated': True}),
            ('plan_tests', {'test_count': 2, 'rationale': 'Cover a successful and rejected menu session.'}),
            ('describe_test', metadata([], 0, 'Successful scripted menu session.', 10, 'CRLF', False, 24, 'LF', True)),
            *chunks(STDIN_LINES), *chunks(STDOUT_LINES),
            ('describe_test', metadata(['--quiet'], 1, 'Invalid menu path preserves exact formatting.', 1, 'NONE', False, 3, 'CRLF', False)),
            *chunks(['bad']), *chunks(['Error: bad menu', '  detail with spaces  ', 'Retry: ']),
        ]
        def hook(index, state, tools):
            if index > 0:
                self.assertNotIn('submit_tests', tools, 'Fallback must not request the oversized whole batch again')
                self.assertFalse((team.experiment.root / 'test_plan.json').exists())
            if index == 8:
                self.assertEqual(state['test_draft']['completed_test_ids'], ['R001-T01'])
                self.assertEqual(state['test_draft']['target_test_case_id'], 'R001-T02')
                self.assertNotIn('plan_tests', tools)
        team = ScriptedDesigner(actions, hook=hook)
        exp = self.experiment(team)
        exp.design_tests()
        self.assertFalse(team.actions)
        self.assertEqual(len(team.reviewed), 2)
        self.assertEqual([test['test_case_id'] for test in exp.tests], ['R001-T01', 'R001-T02'])
        expected = [('\r\n'.join(STDIN_LINES), '\n'.join(STDOUT_LINES) + '\n'),
                    ('bad', 'Error: bad menu\r\n  detail with spaces  \r\nRetry: ')]
        for test, (stdin, stdout) in zip(exp.tests, expected):
            self.assertEqual(test['stdin'].encode(), stdin.encode())
            self.assertEqual(test['expected_stdout'].encode(), stdout.encode())
            self.assertNotIn('POISON', json.dumps(test))
        self.assertEqual([test for review in team.reviewed for test in review['tests']], exp.tests)
        self.assertTrue(all(len(review['tests']) == 1 for review in team.reviewed))
        self.assertTrue(all([test['test_case_id'] for test in review['test_plan_overview']] == ['R001-T01', 'R001-T02']
                            for review in team.reviewed))
        plan = json.loads((exp.root / 'test_plan.json').read_text())
        self.assertEqual(plan['tests'], exp.tests)
        self.assertEqual([test['expected_exit_code'] for test in exp.tests], [0, 1])
        self.assertEqual(exp.tests[1]['args'], ['--quiet'])
        self.assertTrue(list((exp.root / 'test_drafts/R001').rglob('*.json')), 'Chunk drafting events must be archived')

    def test_truncated_chunk_does_not_advance_offsets_or_contaminate_the_final_test(self):
        before = []
        def hook(index, state, tools):
            if index == 3:
                before.append(copy.deepcopy(state['test_draft']))
            if index == 4:
                self.assertEqual(state['test_draft'], before[0])
                self.assertIn('no action was executed', state['previous_action']['error'])
        actions = [
            ('submit_tests', {}, {'truncated': True, 'unparseable': True}),
            ('plan_tests', {'test_count': 1, 'rationale': 'One bounded scripted session.'}),
            ('describe_test', metadata([], 0, 'Exact scripted session.', 2, 'LF', True, 1, 'NONE', False)),
            ('append_test_lines', {'lines': ['POISON', 'POISON']}, {'truncated': True}),
            ('append_test_lines', {'lines': ['real', 'payload']}),
            ('append_test_lines', {'lines': ['ready']}),
        ]
        team = ScriptedDesigner(actions, hook=hook)
        exp = self.experiment(team)
        exp.design_tests()
        self.assertEqual(len(exp.tests), 1)
        self.assertEqual(exp.tests[0]['stdin'], 'real\npayload\n')
        self.assertEqual(exp.tests[0]['expected_stdout'], 'ready')
        self.assertNotIn('POISON', (exp.root / 'test_plan.json').read_text())

    def test_plan_and_metadata_truncation_keep_eight_line_chunks_and_completed_tests_intact(self):
        first_input = [f'first-{number}' for number in range(8)]
        second_input = [f'second-{number}' for number in range(8)]
        saved_metadata_state = []
        def hook(index, state, tools):
            draft = state.get('test_draft')
            if index in (1, 2):
                self.assertEqual(draft['phase'], 'PLAN')
            if index in (4, 8):
                self.assertEqual(draft['phase'], 'LINES')
                self.assertEqual(draft['field'], 'stdin')
                self.assertEqual(draft['next_chunk_count'], 8,
                                 'A truncated plan/header must not shrink subsequent line chunks')
            if index == 6:
                self.assertEqual(draft['phase'], 'DESCRIBE')
                self.assertEqual(draft['completed_test_ids'], ['R001-T01'])
                saved_metadata_state.append(copy.deepcopy(draft))
            if index == 7:
                self.assertEqual(draft, saved_metadata_state[0])
                self.assertIn('no action was executed', state['previous_action']['error'])
            if index == 8:
                self.assertEqual(draft['completed_test_ids'], ['R001-T01'])
        actions = [
            ('submit_tests', {}, {'truncated': True, 'unparseable': True}),
            ('plan_tests', {'test_count': 2, 'rationale': 'This plan response is truncated.'}, {'truncated': True}),
            ('plan_tests', {'test_count': 2, 'rationale': 'Two complete input sessions.'}),
            ('describe_test', metadata([], 0, 'First eight-line session.', 8, 'LF', True, 1, 'NONE', False)),
            ('append_test_lines', {'lines': first_input}),
            ('append_test_lines', {'lines': ['first complete']}),
            ('describe_test', metadata([], 0, 'This header response is truncated.', 8, 'CRLF', False, 1, 'NONE', False), {'truncated': True}),
            ('describe_test', metadata([], 0, 'Second eight-line session.', 8, 'CRLF', False, 1, 'NONE', False)),
            ('append_test_lines', {'lines': second_input}),
            ('append_test_lines', {'lines': ['second complete']}),
        ]
        team = ScriptedDesigner(actions, hook=hook)
        exp = self.experiment(team)
        exp.design_tests()
        self.assertEqual([test['stdin'] for test in exp.tests],
                         ['\n'.join(first_input) + '\n', '\r\n'.join(second_input)])
        self.assertEqual([test['expected_stdout'] for test in exp.tests], ['first complete', 'second complete'])
        self.assertEqual(len(team.states), 10)
        self.assertEqual(len(team.reviewed), 2)

    def test_partial_draft_respects_both_budgets_and_never_freezes_or_reaches_review(self):
        for limits in ({'max_model_calls': 4, 'max_turns': 16}, {'max_model_calls': 30, 'max_turns': 4}):
            with self.subTest(limits=limits):
                actions = [
                    ('submit_tests', {}, {'truncated': True, 'unparseable': True}),
                    ('plan_tests', {'test_count': 1, 'rationale': 'A session with both input and output.'}),
                    ('describe_test', metadata([], 0, 'Both fields must be completed.', 1, 'LF', True, 1, 'LF', True)),
                    ('append_test_lines', {'lines': ['input']}),
                    ('append_test_lines', {'lines': ['output']}),
                ]
                team = ScriptedDesigner(actions)
                exp = self.experiment(team, **limits)
                with self.assertRaises(BudgetExceeded):
                    exp.design_tests()
                self.assertEqual(exp.model_calls, 4)
                self.assertEqual(len(team.states), 4)
                self.assertEqual(team.reviewed, [])
                self.assertFalse(exp.tests)
                self.assertFalse((exp.root / 'test_plan.json').exists())
                self.assertTrue(list((exp.root / 'test_drafts/R001').rglob('*.json')))

    def test_rejected_assembled_candidate_is_archived_and_redrafted_before_review(self):
        actions = [
            ('submit_tests', {}, {'truncated': True, 'unparseable': True}),
            ('plan_tests', {'test_count': 1, 'rationale': 'Check integer addition.'}),
            ('describe_test', metadata(['2+3'], 0, 'Add the two operands.', 0, 'NONE', False, 1, 'NONE', False)),
            ('append_test_lines', {'lines': ['999']}),
            ('plan_tests', {'test_count': 1, 'rationale': 'Correct the rejected independent arithmetic expectation.'}),
            ('describe_test', metadata(['2+3'], 0, 'Add the two operands.', 0, 'NONE', False, 1, 'NONE', False)),
            ('append_test_lines', {'lines': ['5']}),
        ]
        def hook(index, state, tools):
            if index == 4:
                self.assertTrue(state['previous_action']['result'].get('error'))
                self.assertFalse(team.experiment.tests)
        team = ScriptedDesigner(actions, hook=hook)
        exp = Experiment(Config(prompt='Build a Python calculator that adds two integers.', output=self.output,
                                calculator_reference=True), transport=team)
        exp.requirements = [{**REQUIREMENT, 'text': 'The calculator shall add two integers.',
                             'source_quote': 'adds two integers'}]
        team.experiment = exp
        exp.design_tests()
        self.assertEqual(exp.tests[0]['expected_stdout'], '5')
        self.assertEqual(len(team.reviewed), 1)
        self.assertNotIn('999', (exp.root / 'test_plan.json').read_text())
        drafts = list((exp.root / 'test_drafts/R001').rglob('*.json'))
        self.assertTrue(any('999' in path.read_text() for path in drafts), 'Rejected expectations must remain in the audit trail')


if __name__ == '__main__':
    unittest.main()
