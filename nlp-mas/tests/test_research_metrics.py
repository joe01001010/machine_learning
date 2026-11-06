import json
from pathlib import Path
import tempfile
import unittest

from experiment import Config, Experiment
from research_metrics import collect, generate


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


class ResearchMetricsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)/'results'/'run-1'
        self.root.mkdir(parents=True)

    def fixture(self):
        rows = []
        for i, (match, verdict) in enumerate([(False, 'PASS'), (True, 'FAIL'), (False, 'FAIL'), (False, 'INCONCLUSIVE')]):
            rows.append({'test_case_id': f'R001-T{i}', 'execution': {'execution_id': str(i), 'host_match': match}, 'verdict': {'verdict': verdict}})
        write(self.root/'report.json', {'status': 'FAIL', 'started_at': '2026-01-01T00:00:00+00:00', 'finished_at': '2026-01-01T00:00:10+00:00',
            'requirements': [{'requirement_id': 'R001'}, {'requirement_id': 'R002'}], 'rounds': [{'build': {'success': True}, 'tests': rows}]})
        write(self.root/'test_plan.json', {'tests': [{'test_case_id': 'R001-T0'}, {'test_case_id': 'R001-T1'}]})

    def test_quality_metrics_use_independent_truth_and_preserve_abstentions(self):
        self.fixture()
        labels = {'reference_requirements': ['G1', 'G2'], 'requirement_matches': {'R001': 'G1', 'R002': None},
            'test_validity': {'R001-T0': True, 'R001-T1': False},
            'execution_correctness': {'0': False, '1': True, '2': False, '3': False},
            'task_success': False, 'mutations': [{'id': 'm1', 'eligible': True, 'killed': True}, {'id': 'm2', 'eligible': True, 'killed': False}, {'id': 'm3', 'eligible': False, 'killed': False}]}
        m = collect(self.root, labels)
        req = m['agents']['requirements-engineer']['quality']
        self.assertEqual(req['f1']['value'], .5)
        test = m['agents']['test-engineer']['quality']
        self.assertEqual(test['mutation_score']['value'], .5)
        self.assertEqual(test['valid_test_rate']['value'], .5)
        verification = m['agents']['verification-engineer']['quality']
        self.assertEqual(verification['confusion_matrix'], {'tp': 1, 'fp': 1, 'fn': 1, 'tn': 0})
        self.assertEqual(verification['false_pass_rate']['value'], 1/3)
        self.assertEqual(verification['abstention_rate']['value'], .25)
        self.assertFalse(m['system']['independent_task_success'])
        self.assertEqual(m['elapsed_seconds'], 10)

    def test_missing_labels_are_unavailable_and_failed_runs_stay_in_denominator(self):
        self.fixture()
        other = self.root.parent/'run-2'
        write(other/'report.json', {'status': 'INCOMPLETE'})
        generate(self.root)
        summary = json.loads((self.root/'research_summary.json').read_text())
        self.assertEqual(summary['runs'], 2)
        self.assertEqual(summary['pipeline_success_rate']['denominator'], 2)
        self.assertEqual(summary['independent_task_success_rate']['value'], None)
        m = json.loads((self.root/'metrics.json').read_text())
        self.assertIsNone(m['agents']['requirements-engineer']['quality']['precision']['value'])
        self.assertIn('N/A', (self.root/'metrics.html').read_text())

    def test_feedback_progress_counts_group_reviews_and_eventual_acceptance(self):
        write(self.root/'report.json', {'status': 'INCOMPLETE', 'requirements_source_excerpts_completed': 2})
        write(self.root/'requirements_sources.json', {'S001': 'one', 'S002': 'two', 'S003': 'three'})
        write(self.root/'requirements_review/block-001/review-0.json', {'decision': 'REVISE'})
        write(self.root/'requirements_review/block-001/review-1.json', {'decision': 'ACCEPT'})
        write(self.root/'requirements_review/block-002/review-0.json', {'decision': 'REVISE'})
        write(self.root/'requirements_review/block-002/review-1.json', {'decision': 'REVISE'})
        write(self.root/'test_design/R001/review-0.json', {'decision': 'ACCEPT'})
        write(self.root/'test_design/R001/review-0-case-1.json', {'decision': 'ACCEPT'})
        feedback = generate(self.root)['feedback']
        self.assertEqual(feedback['prompt_excerpts_processed']['value'], 2/3)
        requirements = feedback['requirements_review']
        self.assertEqual(requirements['reviews'], 4)
        self.assertEqual(requirements['revise_decisions'], 3)
        self.assertEqual(requirements['revision_acceptance_rate']['value'], .5)
        tests = feedback['test_plan_review']
        self.assertEqual(tests['reviews'], 1)
        self.assertIsNone(tests['revision_acceptance_rate']['value'])
        self.assertIn('Accepted after revision', (self.root/'metrics.html').read_text())

    def test_requirements_only_outcomes_are_separate_from_full_pipeline(self):
        self.fixture()
        other = self.root.parent/'requirements-run'
        write(other/'report.json', {'status': 'PASS'})
        manifest = {'language': 'python', 'entrypoint': 'main.py'}
        write(self.root/'manifest.json', manifest)
        write(other/'manifest.json', {**manifest, 'requirements_only': True})
        write(other/'evaluation_labels.json', {'task_success': True})
        metrics = generate(other)
        self.assertIsNone(metrics['system']['pipeline_success'])
        self.assertTrue(metrics['system']['requirements_only_success'])
        summary = json.loads((other/'research_summary.json').read_text())
        self.assertEqual(summary['runs'], 2)
        self.assertEqual(summary['pipeline_success_rate']['denominator'], 1)
        self.assertEqual(summary['pipeline_success_rate']['value'], 0)
        self.assertEqual(summary['requirements_only_success_rate']['value'], 1)
        self.assertIsNone(summary['independent_task_success_rate']['value'])
        self.assertEqual(summary['independent_requirements_success_rate']['value'], 1)
        self.assertEqual(summary['by_application']['python:main.py']['runs'], 1)
        self.assertEqual(summary['by_requirements_application']['python:main.py']['runs'], 1)

    def test_injected_responses_do_not_count_as_live_outcomes(self):
        self.fixture()
        write(self.root/'server_models.json', [])
        other = self.root.parent/'demo'
        write(other/'report.json', {'status': 'PASS'})
        write(other/'manifest.json', {'agent_transport': 'injected'})
        self.assertEqual(generate(self.root)['agent_transport'], 'http')
        live = json.loads((self.root/'research_summary.json').read_text())
        self.assertEqual(live['status_counts'], {'FAIL': 1})
        self.assertEqual(generate(other)['agent_transport'], 'injected')
        demo = json.loads((other/'research_summary.json').read_text())
        self.assertEqual(demo['status_counts'], {'PASS': 1})
        self.assertIn('Injected role responses', (other/'metrics.html').read_text())

    def test_usage_latency_errors_and_partial_trace_are_counted(self):
        write(self.root/'report.json', {'status': 'ERROR'})
        events = [{'role': 'coder', 'response': {'usage': {'prompt_tokens': 10, 'completion_tokens': 5}, 'choices': [{'finish_reason': 'length'}]}, 'duration_seconds': 2},
                  {'role': 'coder', 'transport_error': 'RuntimeError', 'duration_seconds': 4},
                  {'role': 'coder', 'tool': 'write_file', 'result': {'error': 'invalid'}}]
        (self.root/'trace.jsonl').write_text('\n'.join(map(json.dumps, events))+'\n{"partial":')
        m = collect(self.root)
        a = m['agents']['coder']
        self.assertEqual((a['calls'], a['transport_errors'], a['truncations'], a['rejected_tool_calls']), (2,1,1,1))
        self.assertEqual(a['completion_tokens'], 5)
        self.assertEqual(a['usage_reported_calls'], 1)
        self.assertEqual(a['mean_call_seconds'], 3)
        self.assertEqual(m['data_quality']['malformed_trace_lines'], 1)

    def test_duplicate_or_foreign_labels_are_rejected(self):
        self.fixture()
        for labels in [{'reference_requirements': ['G1'], 'requirement_matches': {'R001': 'G1', 'R002': 'G1'}}, {'test_validity': {'fake': True}}]:
            with self.assertRaises(ValueError):
                collect(self.root, labels)

    def test_pipeline_error_interrupt_and_invalid_annotations_still_emit_reports(self):
        for error in [RuntimeError('model offline'), KeyboardInterrupt()]:
            with self.subTest(error=type(error).__name__):
                def fail(role, body):
                    raise error
                exp = Experiment(Config(prompt='Build a calculator', output=self.root.parent), transport=fail)
                report = exp.run()
                self.assertIn(report['status'], ('ERROR', 'INCOMPLETE'))
                self.assertTrue((exp.root/'metrics.html').is_file())
                self.assertTrue((exp.root/'metrics.json').is_file())
        labels = self.root.parent/'bad.json'
        write(labels, {'unsupported': 1})
        exp = Experiment(Config(prompt='Build a calculator', output=self.root.parent, evaluation_labels=labels), transport=lambda r,b: (_ for _ in ()).throw(RuntimeError('offline')))
        report = exp.run()
        self.assertEqual(report['status'], 'ERROR')
        self.assertTrue((exp.root/'metrics_error.json').is_file())
        self.assertTrue((exp.root/'metrics.html').is_file())

    def test_regenerating_report_preserves_completed_label_template(self):
        self.fixture()
        template = self.root/'evaluation_labels.template.json'
        labels = {'reference_requirements': ['G1'], 'requirement_matches': {'R001': 'G1', 'R002': None}}
        write(template, labels)
        generate(self.root, template)
        self.assertEqual(json.loads(template.read_text()), labels)
        self.assertEqual(json.loads((self.root/'evaluation_labels.json').read_text()), labels)
        self.assertEqual(json.loads((self.root/'metrics.json').read_text())['agents']['requirements-engineer']['quality']['precision']['value'], .5)
