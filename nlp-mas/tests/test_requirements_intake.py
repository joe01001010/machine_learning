"""Requirements intake advances grounded source blocks without model servers."""
import json
from pathlib import Path
import re
import tempfile
import unittest

from experiment import BudgetExceeded, Config, Experiment


FEATURES = [
    'Register passengers with unique user names and a contact telephone number.',
    'Register drivers with a unique identifier, a vehicle plate, and a current location.',
    'Let a passenger request a ride by supplying pickup and destination coordinates.',
    'Assign each requested ride to the nearest available driver.',
    'Allow the passenger to cancel a ride before its driver marks it completed.',
    'Let the assigned driver mark a ride completed after reaching its destination.',
    'Calculate the fare from a fixed base charge plus a charge per unit distance.',
    'Keep driver availability consistent when rides are assigned, cancelled, or completed.',
    'Display the current status and assigned driver for each active ride.',
    'Maintain a history of completed rides and the total fares paid by each passenger.',
    'Reject an invalid menu choice with an error and keep the menu available.',
    'Quit the application cleanly when the user selects the exit menu choice.',
]
BACKGROUND = [
    f'Background note {index:02d}: This paragraph describes classroom presentation context for the rideshare experiment and introduces no additional application behavior beyond the capabilities above.'
    for index in range(1, 24)
]
LONG_SOURCES = FEATURES + BACKGROUND
LONG_PROMPT = '\n'.join(LONG_SOURCES)
TEXTS = {feature: 'The application shall ' + feature[0].lower() + feature[1:] for feature in FEATURES}


def completion(requirements, complete=True, *, finish_reason='tool_calls', include_complete=True):
    arguments = {'requirements': requirements}
    if include_complete:
        arguments['complete'] = complete
    return {'choices': [{'finish_reason': finish_reason, 'message': {
        'role': 'assistant', 'content': None,
        'tool_calls': [{'id': 'intake-test', 'type': 'function', 'function': {
            'name': 'submit_requirements', 'arguments': json.dumps(arguments)}}],
    }}]}


def review_completion(decision, feedback):
    response = completion([])
    call = response['choices'][0]['message']['tool_calls'][0]['function']
    call.update(name='review_requirements', arguments=json.dumps({'decision': decision, 'feedback': feedback}))
    return response


def state_of(body):
    return json.loads(body['messages'][0]['content'])


def candidate(identifier, text, source_id):
    return {'requirement_id': identifier, 'text': text, 'source_id': source_id}


def accepted_ids(state):
    return [requirement['requirement_id'] for requirement in state['accepted_requirements']]


def archived_batches(exp):
    return [json.loads(path.read_text()) for path in sorted((exp.root / 'requirements_batches').glob('*.json'))]


def extract_current_features(state):
    number = int(state['next_requirement_id'][1:])
    requirements = []
    for source in state['source_block']['sources']:
        if source['text'] in TEXTS:
            requirements.append(candidate(f'R{number:03d}', TEXTS[source['text']], source['source_id']))
            number += 1
    return requirements


class RequirementsIntakeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.output = Path(self.temp.name) / 'results'

    def tearDown(self):
        self.temp.cleanup()

    def make_experiment(self, transport, *, prompt=LONG_PROMPT, **options):
        return Experiment(Config(prompt=prompt, output=self.output, **options), transport=transport)

    def assert_provenance(self, exp, prompt):
        sources = json.loads((exp.root / 'requirements_sources.json').read_text())
        for requirement in exp.requirements:
            self.assertEqual(requirement['source_quote'], sources[requirement['source_id']])
            self.assertIn(requirement['source_quote'], prompt)

    def test_requirements_only_revises_before_committing_and_accepts_covered_block(self):
        prompt = '\n'.join(FEATURES[:3] + ['The above behavior defines the initial scope.'])
        modes = []

        def team(role, body):
            state = state_of(body)
            mode = state['mode']
            modes.append(mode)
            self.assertNotIn('interface', state)
            self.assertFalse((exp.root / 'requirements.json').exists())
            if mode == 'requirements':
                self.assertEqual(role, 'requirements-engineer')
                if state['source_ids'] == ['S004']:
                    self.assertEqual(accepted_ids(state), ['R001', 'R002', 'R003'])
                    return completion([])
                requirements = extract_current_features(state)
                if 'revision_feedback' not in state:
                    return completion(requirements[:-1])
                self.assertEqual(state['revision_feedback'], 'S003 is missing.')
                self.assertEqual(len(state['previous_proposal']), 2)
                self.assertEqual(state['next_requirement_id'], 'R001')
                self.assertFalse(archived_batches(exp))
                return completion(requirements)
            self.assertEqual(role, 'verification-engineer')
            proposed = state['proposed_requirements']
            if state['source_ids'] == ['S004']:
                self.assertEqual(proposed, [])
                return review_completion('ACCEPT', 'This block adds no new product behavior.')
            if len(proposed) == 2:
                return review_completion('REVISE', 'S003 is missing.')
            self.assertEqual(proposed[2]['source_quote'], FEATURES[2])
            return review_completion('ACCEPT', 'All three features are represented.')

        exp = self.make_experiment(team, prompt=prompt, requirements_only=True)
        report = exp.run()
        self.assertEqual(report['status'], 'PASS')
        self.assertEqual(report['rounds'], [])
        self.assertEqual(modes, ['requirements', 'requirements_review'] * 3)
        review_dir = exp.root / 'requirements_review' / 'block-001'
        self.assertEqual(json.loads((review_dir / 'review-0.json').read_text())['decision'], 'REVISE')
        self.assertEqual(json.loads((review_dir / 'review-1.json').read_text())['decision'], 'ACCEPT')
        self.assertEqual(len(archived_batches(exp)), 2)
        self.assertEqual(report['requirements_source_excerpts_completed'], 4)
        self.assertEqual(len(exp.requirements), 3)

    def test_review_limit_cannot_freeze_unaccepted_requirements(self):
        def team(role, body):
            state = state_of(body)
            if role == 'requirements-engineer':
                return completion(extract_current_features(state))
            return review_completion('REVISE', 'Clarify the expected validation behavior.')

        exp = self.make_experiment(team, prompt='\n'.join(FEATURES[:3]), requirements_only=True)
        report = exp.run()
        self.assertEqual(report['status'], 'INCOMPLETE')
        self.assertIn('review limit', report['reason'])
        self.assertEqual(exp.model_calls, 6)
        self.assertEqual(exp.requirements, [])
        self.assertFalse(archived_batches(exp))
        self.assertFalse((exp.root / 'requirements.json').exists())
        self.assertEqual(len(list((exp.root / 'requirements_review' / 'block-001').glob('review-*.json'))), 3)

    def test_35_fragments_advance_in_12_calls_and_global_commit_waits_for_last_source_block(self):
        states = []
        exp = None
        def team(role, body):
            state = state_of(body)
            states.append(state)
            self.assertEqual(role, 'requirements-engineer')
            self.assertNotIn('interface', state)
            self.assertFalse(exp.requirements)
            self.assertFalse((exp.root / 'requirements.json').exists())
            current = dict(re.findall(r'^\[(S\d+)\] (.*)$', state['customer_prompt'], re.M))
            self.assertEqual(list(current), state['source_ids'])
            self.assertEqual(list(current), [source['source_id'] for source in state['source_block']['sources']])
            self.assertEqual(len(re.findall(r'^\[S\d+\]', state['full_prompt_context'], re.M)), 35)
            self.assertEqual(state['batch_limit'], 3)
            return completion(extract_current_features(state))
        exp = self.make_experiment(team)
        self.assertGreater(len(LONG_PROMPT), 4000)
        self.assertTrue(all(len(source) <= 200 for source in LONG_SOURCES))
        exp.intake()
        self.assertEqual(len(states), 12)
        self.assertEqual(len(exp.requirements), len(FEATURES))
        self.assertEqual([r['text'] for r in exp.requirements], list(TEXTS.values()))
        self.assertEqual([r['requirement_id'] for r in exp.requirements], [f'R{i:03d}' for i in range(1, 13)])
        batches = archived_batches(exp)
        self.assertEqual([sid for batch in batches for sid in batch['source_ids']], [f'S{i:03d}' for i in range(1, 36)])
        self.assertEqual([batch['intake_complete'] for batch in batches], [False] * 11 + [True])
        self.assertTrue(any(not batch['submitted_requirements'] for batch in batches), 'Heading/background blocks should be explicitly skipped')
        self.assertEqual(json.loads((exp.root / 'requirements.json').read_text()), exp.requirements)
        self.assert_provenance(exp, LONG_PROMPT)

    def test_truncated_payload_is_not_accepted_and_shrinking_source_blocks_loses_no_fragments(self):
        states = []
        exp = None
        def team(role, body):
            state = state_of(body)
            states.append(state)
            self.assertFalse(exp.requirements)
            if len(states) == 1:
                return completion(extract_current_features(state), finish_reason='length')
            self.assertEqual(state['batch_limit'], 1)
            self.assertEqual(len(state['source_ids']), 1)
            if len(states) == 2:
                self.assertEqual(accepted_ids(state), [])
                self.assertEqual(state['source_ids'], ['S001'])
                self.assertFalse(archived_batches(exp))
                retry = json.dumps(state['previous_action']).lower()
                self.assertIn('requirement', retry)
                self.assertNotIn('source chunk', retry)
                self.assertNotIn('append', retry)
            return completion(extract_current_features(state))
        exp = self.make_experiment(team, max_turns=40, max_model_calls=50)
        exp.intake()
        self.assertEqual(len(states), 36)
        self.assertEqual(len(exp.requirements), 12)
        batches = archived_batches(exp)
        self.assertEqual([sid for batch in batches for sid in batch['source_ids']], [f'S{i:03d}' for i in range(1, 36)])
        self.assertEqual([r['text'] for r in exp.requirements], list(TEXTS.values()))
        self.assert_provenance(exp, LONG_PROMPT)

    def test_duplicate_raw_ids_reject_atomically_without_advancing_the_current_block(self):
        for duplicate in ('existing', 'within_batch'):
            with self.subTest(duplicate=duplicate):
                states = []
                def team(role, body):
                    state = state_of(body)
                    states.append(state)
                    if len(states) == 1:
                        return completion([candidate('R001', TEXTS[FEATURES[0]], 'S001')])
                    self.assertEqual(state['source_ids'], ['S004'])
                    if len(states) == 2:
                        repeated_id = 'R001' if duplicate == 'existing' else 'R002'
                        return completion([candidate('R002', 'The app shall assign a driver.', 'S004'),
                                           candidate(repeated_id, 'The app shall record the assignment.', 'S004')])
                    self.assertEqual(accepted_ids(state), ['R001'])
                    self.assertEqual(state['next_requirement_id'], 'R002')
                    self.assertTrue(state['previous_action']['result'].get('error'))
                    return completion([candidate('R002', 'The app shall assign a driver.', 'S004')])
                exp = self.make_experiment(team, prompt='\n'.join(FEATURES[:4]))
                exp.intake()
                self.assertEqual(len(states), 3)
                self.assertEqual([r['requirement_id'] for r in exp.requirements], ['R001', 'R002'])
                self.assertEqual(len(archived_batches(exp)), 2)
                self.assert_provenance(exp, exp.config.prompt)

    def test_unknown_source_id_is_atomic_and_host_preserves_original_unicode_quotation(self):
        prompt = 'Use „a” as the passenger menu label, retaining its exact spelling.\n' + FEATURES[1]
        states = []
        good = [candidate('R001', 'The app shall use the requested passenger label.', 'S001'),
                candidate('R002', TEXTS[FEATURES[1]], 'S002')]
        def team(role, body):
            state = state_of(body)
            states.append(state)
            if len(states) == 1:
                return completion([good[0], {**good[1], 'source_id': 'S999'}])
            self.assertEqual(accepted_ids(state), [])
            self.assertEqual(state['next_requirement_id'], 'R001')
            self.assertTrue(state['previous_action']['result'].get('error'))
            return completion(good)
        exp = self.make_experiment(team, prompt=prompt)
        exp.intake()
        self.assertEqual(len(archived_batches(exp)), 1)
        self.assertEqual(exp.requirements[0]['source_quote'], prompt.splitlines()[0])
        self.assertIn('„a”', exp.requirements[0]['source_quote'])
        self.assert_provenance(exp, prompt)

    def test_partial_global_intake_respects_model_and_turn_limits_without_premature_commit(self):
        for limits in ({'max_model_calls': 2, 'max_turns': 16}, {'max_model_calls': 30, 'max_turns': 2}):
            with self.subTest(limits=limits):
                calls = []
                def team(role, body):
                    state = state_of(body)
                    calls.append(state)
                    return completion(extract_current_features(state))
                exp = self.make_experiment(team, prompt='\n'.join(FEATURES[:10]), **limits)
                with self.assertRaises(BudgetExceeded):
                    exp.intake()
                self.assertEqual(len(calls), 2)
                self.assertEqual(exp.model_calls, 2)
                self.assertFalse(exp.requirements)
                self.assertFalse((exp.root / 'requirements.json').exists())
                batches = archived_batches(exp)
                self.assertEqual(len(batches), 2)
                self.assertTrue(all(batch['block_complete'] and not batch['intake_complete'] for batch in batches))
                self.assertEqual([sid for batch in batches for sid in batch['source_ids']], [f'S{i:03d}' for i in range(1, 7)])

    def test_consecutive_truncation_stops_after_three_calls_with_stage_specific_guidance(self):
        states = []
        def team(role, body):
            state = state_of(body)
            states.append(state)
            return completion(extract_current_features(state), finish_reason='length')
        exp = self.make_experiment(team, max_turns=16, max_model_calls=30)
        with self.assertRaises(BudgetExceeded):
            exp.intake()
        self.assertEqual(len(states), 3)
        self.assertEqual([state['batch_limit'] for state in states], [3, 1, 1])
        for state in states[1:]:
            guidance = json.dumps(state['previous_action']).lower()
            self.assertIn('requirement', guidance)
            self.assertNotIn('source chunk', guidance)
            self.assertNotIn('append', guidance)
        self.assertFalse(exp.requirements)
        self.assertFalse((exp.root / 'requirements.json').exists())
        self.assertFalse(archived_batches(exp))

    def test_false_or_missing_complete_flag_cannot_commit_or_advance_a_block(self):
        for flag in ('missing', 'false'):
            with self.subTest(flag=flag):
                states = []
                def team(role, body):
                    state = state_of(body)
                    states.append(state)
                    proposal = [candidate('R001', TEXTS[FEATURES[0]], 'S001')]
                    if len(states) == 1:
                        return completion(proposal, complete=False, include_complete=flag != 'missing')
                    self.assertEqual(accepted_ids(state), [])
                    self.assertEqual(state['source_ids'], ['S001'])
                    self.assertTrue(state['previous_action']['result'].get('error'))
                    return completion(proposal)
                exp = self.make_experiment(team, prompt=FEATURES[0])
                exp.intake()
                self.assertEqual(len(states), 2)
                self.assertEqual(len(exp.requirements), 1)
                self.assertEqual(len(archived_batches(exp)), 1)

    def test_duplicate_fact_reuses_canonical_id_while_new_facts_get_contiguous_ids_and_case_is_preserved(self):
        prompt = '\n'.join(['Keep rider names unique.', 'Recognize rider A.', 'Passenger section heading.',
                            'Keep rider names unique.', 'Charge a base fare.', 'Recognize rider a.'])
        submissions = []
        def team(role, body):
            state = state_of(body)
            if state['source_block']['index'] == 1:
                proposal = [candidate('R001', 'The app shall keep rider names unique.', 'S001'),
                            candidate('R002', 'The app shall recognize rider A.', 'S002')]
            else:
                self.assertEqual(accepted_ids(state), ['R001', 'R002'])
                proposal = [candidate('R003', 'The   app shall keep   rider names unique.', 'S004'),
                            candidate('R004', 'The app shall charge a base fare.', 'S005'),
                            candidate('R005', 'The app shall recognize rider a.', 'S006')]
            submissions.append(proposal)
            return completion(proposal)
        exp = self.make_experiment(team, prompt=prompt)
        exp.intake()
        self.assertEqual([r['requirement_id'] for r in exp.requirements], ['R001', 'R002', 'R003', 'R004'])
        self.assertEqual([r['text'] for r in exp.requirements], ['The app shall keep rider names unique.',
                         'The app shall recognize rider A.', 'The app shall charge a base fare.',
                         'The app shall recognize rider a.'])
        second = archived_batches(exp)[1]
        self.assertEqual(second['submitted_requirements'], submissions[1])
        self.assertEqual(second['duplicates'], [{'submitted_requirement_id': 'R003',
            'canonical_requirement_id': 'R001', 'source_id': 'S004', 'source_quote': 'Keep rider names unique.'}])
        self.assertEqual(second['id_assignments'], [{'submitted_requirement_id': 'R004', 'canonical_requirement_id': 'R003'},
                                                   {'submitted_requirement_id': 'R005', 'canonical_requirement_id': 'R004'}])
        self.assertEqual([r['requirement_id'] for r in second['requirements']], ['R003', 'R004'])
        self.assert_provenance(exp, prompt)

    def test_all_duplicate_block_advances_and_does_not_renumber_later_new_facts(self):
        lines = ['Keep rider names unique.', 'Passenger heading.', 'Driver heading.',
                 'Keep rider names unique.', 'Keep rider names unique.', 'Repeated section heading.',
                 'Charge a base fare.', 'Fare heading.', 'End section heading.']
        states = []
        def team(role, body):
            state = state_of(body)
            states.append(state)
            block = state['source_block']['index']
            if block == 1:
                proposal = [candidate('R001', 'The app shall keep rider names unique.', 'S001')]
            elif block == 2:
                proposal = [candidate('R002', 'The app shall keep rider names unique.', 'S004'),
                            candidate('R003', 'The  app shall keep rider names unique.', 'S005')]
            else:
                self.assertEqual(state['next_requirement_id'], 'R002')
                self.assertEqual(accepted_ids(state), ['R001'])
                proposal = [candidate('R002', 'The app shall charge a base fare.', 'S007')]
            return completion(proposal)
        exp = self.make_experiment(team, prompt='\n'.join(lines))
        exp.intake()
        self.assertEqual(len(states), 3)
        self.assertEqual([r['requirement_id'] for r in exp.requirements], ['R001', 'R002'])
        batches = archived_batches(exp)
        self.assertEqual(len(batches[1]['duplicates']), 2)
        self.assertEqual(batches[1]['requirements'], [])
        self.assertEqual(batches[1]['accepted_ids'], ['R001'])
        self.assertEqual([sid for batch in batches for sid in batch['source_ids']], [f'S{i:03d}' for i in range(1, 10)])


if __name__ == '__main__':
    unittest.main()
