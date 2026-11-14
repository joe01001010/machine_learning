"""Offline research reports; no model calls. Unknown quality is never scored as zero."""
import argparse
import csv
import hashlib
from collections import Counter
from datetime import datetime
from html import escape
import json
from pathlib import Path
import statistics

ROLES = ('requirements-engineer', 'test-engineer', 'coder', 'verification-engineer')


def read(path, default=None):
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else default


def rate(n, d, source, note=''):
    return {'value': n / d if d else None, 'numerator': n, 'denominator': d,
            'source': source, 'note': note}


def unknown(note):
    return {'value': None, 'source': 'independent labels required', 'note': note}


def prf(tp, fp, fn, source):
    p, r = rate(tp, tp+fp, source), rate(tp, tp+fn, source)
    return {'precision': p, 'recall': r, 'f1': {
        'value': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None, 'source': source}}


def validate_labels(labels, requirements, tests, execution_ids):
    """Reject mismatched labels instead of silently dropping them."""
    allowed = {'reference_requirements', 'requirement_matches', 'test_validity',
               'execution_correctness', 'task_success', 'mutations'}
    if not isinstance(labels, dict) or set(labels)-allowed:
        raise ValueError('Unknown evaluation-label fields')
    refs = labels.get('reference_requirements', [])
    if not isinstance(refs, list) or any(not isinstance(x, str) for x in refs) or len(set(refs)) != len(refs):
        raise ValueError('reference_requirements must contain unique string IDs')
    matches = labels.get('requirement_matches', {})
    if not isinstance(matches, dict) or set(matches)-requirements:
        raise ValueError('requirement_matches contains unknown extracted requirement IDs')
    targets = [x for x in matches.values() if x is not None]
    if any(not isinstance(x, str) or x not in refs for x in targets) or len(targets) != len(set(targets)):
        raise ValueError('Matches must be one-to-one reference IDs; use null for unsupported/duplicate extractions')
    for field, ids in [('test_validity', tests), ('execution_correctness', execution_ids)]:
        values = labels.get(field, {})
        if not isinstance(values, dict) or set(values)-ids or any(type(x) is not bool for x in values.values()):
            raise ValueError(f'{field} requires known IDs and boolean labels')
    if 'task_success' in labels and type(labels['task_success']) is not bool:
        raise ValueError('task_success must be a boolean independent assessment')
    mutations = labels.get('mutations', [])
    if not isinstance(mutations, list):
        raise ValueError('mutations must be an array')
    names = set()
    for row in mutations:
        if (not isinstance(row, dict) or set(row) != {'id', 'eligible', 'killed'}
                or not isinstance(row['id'], str) or row['id'] in names
                or type(row['eligible']) is not bool or type(row['killed']) is not bool):
            raise ValueError('Each mutation needs a unique id and boolean eligible/killed values')
        names.add(row['id'])


def review_progress(root, folder, unit):
    reviews = []
    for path in (root/folder).glob('*/review-*.json'):
        attempt = path.stem.removeprefix('review-')
        if attempt.isdigit():
            decision = read(path).get('decision')
            if decision in ('ACCEPT', 'REVISE'):
                reviews.append((path.parent.name, int(attempt), decision))
    revised, accepted_after_revision = set(), set()
    for name, _, decision in sorted(reviews):
        if decision == 'REVISE':
            revised.add(name)
        elif name in revised:
            accepted_after_revision.add(name)
    counts = Counter(decision for _, _, decision in reviews)
    return {'reviews': len(reviews), 'revise_decisions': counts['REVISE'],
            'revision_acceptance_rate': rate(len(accepted_after_revision), len(revised),
                'archived agent reviews', f'{unit} accepted after revision / {unit.lower()} marked for revision.')}


def transport_kind(root, manifest):
    return manifest.get('agent_transport', 'http' if (root/'server_models.json').is_file() else 'unknown')


def collect(root, labels=None):
    root = Path(root)
    report = read(root/'report.json', {})
    plan = read(root/'test_plan.json', {}).get('tests', [])
    requirements = report.get('requirements', read(root/'requirements.json', []))
    rounds = report.get('rounds', [])
    rows = [t for r in rounds for t in r.get('tests', [])]
    # An execution can exist on disk even if interruption prevented its report row.
    evidence = {t['execution']['execution_id']: t['execution'] for t in rows if t.get('execution')}
    for path in root.glob('rounds/*/evidence/*.json'):
        item = read(path)
        evidence[item['execution_id']] = item
    labels = labels or {}
    req_ids = {r['requirement_id'] for r in requirements}
    test_ids = {t['test_case_id'] for t in plan}
    validate_labels(labels, req_ids, test_ids, set(evidence))
    agents = {r: {'calls': 0, 'transport_errors': 0, 'truncations': 0,
        'rejected_tool_calls': 0, 'tool_actions': 0, 'usage_reported_calls': 0,
        'prompt_tokens': 0, 'completion_tokens': 0, '_latencies': [], 'quality': {}} for r in ROLES}
    malformed = 0
    trace = root/'trace.jsonl'
    for line in trace.read_text(encoding='utf-8').splitlines() if trace.exists() else []:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            malformed += 1
            continue
        if event.get('role') not in agents:
            continue
        a = agents[event['role']]
        if 'response' in event or 'transport_error' in event:
            a['calls'] += 1
            a['transport_errors'] += int('transport_error' in event)
            duration = event.get('duration_seconds')
            if isinstance(duration, (float, int)):
                a['_latencies'].append(duration)
            response = event.get('response', {})
            choices = response.get('choices', [])
            a['truncations'] += int(bool(choices) and choices[0].get('finish_reason') == 'length')
            usage = response.get('usage') or {}
            if all(type(usage.get(k)) is int for k in ('prompt_tokens', 'completion_tokens')):
                a['usage_reported_calls'] += 1
                for key in ('prompt_tokens', 'completion_tokens'):
                    a[key] += usage[key]
        elif 'tool' in event:
            a['tool_actions'] += 1
            a['rejected_tool_calls'] += int(isinstance(event.get('result'), dict) and 'error' in event['result'])
    for a in agents.values():
        times = a.pop('_latencies')
        a['latency_reported_calls'] = len(times)
        a['mean_call_seconds'] = statistics.mean(times) if times else None
        a['model_seconds'] = sum(times) if times else None
        a['truncation_rate'] = rate(a['truncations'], a['calls'], 'controller trace')
        a['tool_rejection_rate'] = rate(a['rejected_tool_calls'], a['tool_actions'], 'controller trace')
        if not a['usage_reported_calls']:
            a['prompt_tokens'] = a['completion_tokens'] = None
    rq = agents[ROLES[0]]['quality']
    refs, matches = labels.get('reference_requirements', []), labels.get('requirement_matches', {})
    if refs and set(matches) == req_ids:
        tp = sum(v is not None for v in matches.values())
        rq.update(prf(tp, len(req_ids)-tp, len(refs)-tp, 'independent one-to-one requirement labels'))
    else:
        rq.update({k: unknown('Supply reference requirements and label every extracted requirement.') for k in ('precision', 'recall', 'f1')})
    tq = agents[ROLES[1]]['quality']
    valid = labels.get('test_validity', {})
    tq['valid_test_rate'] = rate(sum(valid.values()), len(valid), 'independent test labels', 'Denominator is labeled frozen tests only.')
    tq['test_label_coverage'] = rate(len(valid), len(test_ids), 'annotation coverage')
    mutations = [m for m in labels.get('mutations', []) if m['eligible']]
    tq['mutation_score'] = rate(sum(m['killed'] for m in mutations), len(mutations), 'independent mutation results', 'Exclude equivalent and invalid mutants; uncovered eligible mutants count as survivors. Mutation testing is not run automatically.')
    cq = agents[ROLES[2]]['quality']
    cq['build_success_rate'] = rate(sum(bool(r['build']['success']) for r in rounds), len(rounds), 'host build records')
    truth = labels.get('execution_correctness', {})
    final = rounds[-1].get('tests', []) if rounds else []
    final_ids = {t['execution']['execution_id'] for t in final if t.get('execution')}
    known = [truth[x] for x in final_ids if x in truth]
    cq['independent_test_pass_rate'] = rate(sum(known), len(known), 'independent final-execution labels')
    cq['execution_label_coverage'] = rate(len(known), len(final_ids), 'annotation coverage')
    cq['frozen_test_match_rate'] = rate(sum(evidence[x]['host_match'] for x in final_ids), len(final_ids), 'host comparison against agent-generated expectations', 'Not an independent correctness measure.')
    vq = agents[ROLES[3]]['quality']
    tp = fp = fn = tn = abstentions = labeled = false_passes = wrong_executions = 0
    for row in rows:
        eid = row.get('execution', {}).get('execution_id')
        if eid not in truth or not row.get('verdict'):
            continue
        labeled += 1
        verdict = row['verdict']['verdict']
        bad = not truth[eid]
        wrong_executions += bad
        false_passes += bad and verdict == 'PASS'
        if verdict not in ('PASS', 'FAIL'):
            abstentions += 1
            continue
        tp += bad and verdict == 'FAIL'
        fp += not bad and verdict == 'FAIL'
        fn += bad and verdict == 'PASS'
        tn += not bad and verdict == 'PASS'
    vq.update(prf(tp, fp, fn, 'independent execution labels; FAIL is positive; abstentions excluded'))
    vq['confusion_matrix'] = dict(tp=tp, fp=fp, fn=fn, tn=tn)
    vq['false_pass_rate'] = rate(false_passes, wrong_executions, 'independent labeled executions', 'Incorrect executions labeled PASS / incorrect executions with a recorded verdict.')
    vq['abstention_rate'] = rate(abstentions, labeled, 'independent labeled verdicts')
    seconds = None
    if report.get('started_at') and report.get('finished_at'):
        seconds = (datetime.fromisoformat(report['finished_at'])-datetime.fromisoformat(report['started_at'])).total_seconds()
    manifest = read(root/'manifest.json', {})
    requirements_only = manifest.get('requirements_only', False)
    passed = report.get('status') == 'PASS'
    excerpts = len(read(root/'requirements_sources.json', {}))
    feedback = {
        'prompt_excerpts_processed': rate(report.get('requirements_source_excerpts_completed', 0),
            excerpts, 'controller intake records', 'Processing every excerpt does not establish that every requirement was captured.'),
        'requirements_review': review_progress(root, 'requirements_review', 'Source blocks'),
        'test_plan_review': review_progress(root, 'test_design', 'Test plans'),
    }
    return {'schema_version': 1, 'application': manifest.get('language', 'unknown') + ':' + manifest.get('entrypoint', 'unknown'),
        'requirements_only': requirements_only, 'agent_transport': transport_kind(root, manifest),
        'run_id': root.name, 'status': report.get('status', 'UNFINISHED'),
        'reason': report.get('reason'), 'elapsed_seconds': seconds, 'agents': agents, 'feedback': feedback,
        'system': {'pipeline_success': None if requirements_only else passed,
            'requirements_only_success': passed if requirements_only else None,
            'independent_task_success': labels.get('task_success'),
            'repair_attempts': max(0, len(rounds)-1),
            'initial_pipeline_failure': bool(rounds) and (not rounds[0]['build']['success'] or any(not t.get('execution', {}).get('host_match') for t in rounds[0].get('tests', []))),
            'requirements_count': len(req_ids), 'frozen_tests_count': len(test_ids),
            'executions_count': len(evidence)},
        'data_quality': {'malformed_trace_lines': malformed,
            'note': 'Pipeline PASS and frozen-test agreement do not establish independent correctness. Missing labels/usage/timing remain unavailable.'}}


def display(value):
    if isinstance(value, dict):
        v = value.get('value')
        return 'N/A' if v is None else f'{v:.1%}' + (f' ({value["numerator"]}/{value["denominator"]})' if 'denominator' in value else '')
    return 'N/A' if value is None else str(value)


def dashboard(metrics, summary):
    sections = []
    scope = 'Requirements only' if metrics['requirements_only'] else 'End to end'
    transport = {'http': 'Live model responses', 'injected': 'Injected role responses'}.get(
        metrics['agent_transport'], 'Response source unrecorded')
    feedback = metrics['feedback']
    review_rows = ''
    for name, key in [('Requirements review', 'requirements_review'), ('Test plan review', 'test_plan_review')]:
        review = feedback[key]
        review_rows += (f'<tr><td>{name}</td><td>{review["reviews"]}</td>'
                        f'<td>{review["revise_decisions"]}</td>'
                        f'<td>{display(review["revision_acceptance_rate"])}</td></tr>')
    feedback_section = f'''<h2>Prompt processing and feedback</h2>
<p>Prompt excerpts processed: {display(feedback['prompt_excerpts_processed'])}. Excerpt processing records intake progress. Use independent requirement labels to measure content coverage.</p>
<table><tr><th>Stage</th><th>Review decisions</th><th>Revise decisions</th><th>Accepted after revision</th></tr>{review_rows}</table>
<p>Accepted after revision is the share of source blocks or test plans marked for revision that were later accepted by the reviewer.</p>'''
    for role, a in metrics['agents'].items():
        cells = ''.join(f'<tr><td>{escape(k)}</td><td>{display(v)} ' + (f'<meter min="0" max="1" value="{v["value"]}"></meter>' if isinstance(v, dict) and v.get('value') is not None else '') + f'</td><td>{escape(v.get("source", "")) if isinstance(v, dict) else ""}</td></tr>'
                        for k,v in a['quality'].items() if k != 'confusion_matrix')
        sections.append(f'<h2>{escape(role)}</h2><p>Calls: {a["calls"]} · Truncations: {a["truncations"]} · Rejected actions: {a["rejected_tool_calls"]} · Input tokens: {display(a["prompt_tokens"])} · Output tokens: {display(a["completion_tokens"])} · Mean seconds/call: {display(a["mean_call_seconds"])}</p><table><tr><th>Metric</th><th>Value</th><th>Evidence</th></tr>{cells}</table>')
    bars = ''.join(f'<div>{escape(status)}: {count}<div class="bar" style="width:{100*count/max(1,summary["runs"]):.2f}%"></div></div>' for status,count in summary['status_counts'].items())
    groups = ''
    for label, key, outcome_key in [
        ('End to end', 'by_application', 'pipeline_success_rate'),
        ('Requirements only', 'by_requirements_application', 'requirements_only_success_rate'),
    ]:
        groups += ''.join(f'<tr><td>{escape(k)}</td><td>{label}</td><td>{v["runs"]}</td><td>{display(v[outcome_key])}</td><td>{display(v["independent_task_success_rate"])}</td></tr>'
                          for k, v in summary[key].items())
    runs = ''.join(f'<tr><td>{escape(r["run_id"])}</td><td>{escape(r["status"])}</td><td>{display(r["elapsed_seconds"])}</td><td>{display(r["completion_tokens"])}</td><td>{display(r["independent_task_success"])}</td></tr>' for r in summary['run_costs'])
    return f'''<!doctype html><meta charset="utf-8"><title>Agent research metrics</title><style>body{{font:16px system-ui;max-width:1100px;margin:32px auto;padding:0 20px;color:#172238}}table{{border-collapse:collapse;width:100%;margin-bottom:24px}}td,th{{padding:8px;border-bottom:1px solid #ccd4df;text-align:left}}.bar{{background:#3478bc;height:18px;margin:5px 0 14px}}h2{{margin-top:32px}}</style><h1>Research report: {escape(metrics['run_id'])}</h1><p>Status: <b>{escape(metrics['status'])}</b> · {scope} · {transport}. {escape(metrics.get('reason') or '')}</p><p>N/A means unavailable, not zero. Agent agreement is not independent correctness. Labels are supplied by the researcher, not generated by the agents.</p>{feedback_section}{''.join(sections)}<h2>Recorded outcomes with the same response source</h2>{bars}<p>End-to-end pipeline success: {display(summary['pipeline_success_rate'])}. End-to-end independent task success: {display(summary['independent_task_success_rate'])}. Pipeline repair success: {display(summary['repair_success_rate'])}.</p><p>Requirements-only success: {display(summary['requirements_only_success_rate'])}. Requirements-only independent task success: {display(summary['independent_requirements_success_rate'])}.</p><h2>Results by application</h2><table><tr><th>Application</th><th>Scope</th><th>Runs</th><th>Controller success</th><th>Independent success</th></tr>{groups}</table><h2>Quality and cost by run</h2><table><tr><th>Run</th><th>Outcome</th><th>Elapsed seconds</th><th>Reported output tokens</th><th>Independent success</th></tr>{runs}</table><p>Controller success includes every recorded outcome within each scope. Independent success uses only labeled runs. See research_summary.json for label coverage. Token totals can be partial: inspect usage_reported_calls in metrics.json.</p>'''


def generate(root, labels_path=None, ignore_labels=False):
    root = Path(root)
    labels = {} if ignore_labels else (read(Path(labels_path), {}) if labels_path else read(root/'evaluation_labels.json', {}))
    if labels_path and not Path(labels_path).is_file():
        raise ValueError('Evaluation labels file does not exist')
    metrics = collect(root, labels)
    if labels_path:
        (root/'evaluation_labels.json').write_text(json.dumps(labels, indent=2), encoding='utf-8')
    source = Path(__file__).read_bytes()
    metrics['reporter_sha256'] = hashlib.sha256(source).hexdigest()
    (root/'research_metrics_source.py').write_bytes(source)
    (root/'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    template = {'reference_requirements': [],
        'requirement_matches': {r['requirement_id']: None for r in read(root/'report.json', {}).get('requirements', [])},
        'test_validity': {t['test_case_id']: None for t in read(root/'test_plan.json', {}).get('tests', [])},
        'execution_correctness': {read(p)['execution_id']: None for p in root.glob('rounds/*/evidence/*.json')}, 'mutations': []}
    if not (root/'evaluation_labels.template.json').exists():
        (root/'evaluation_labels.template.json').write_text(json.dumps(template, indent=2), encoding='utf-8')
    with (root/'metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['agent', 'metric', 'value', 'numerator', 'denominator', 'source', 'note'])
        for role, agent in metrics['agents'].items():
            for name, item in agent.items():
                if name == 'quality':
                    items = item.items()
                else:
                    items = [(name, item)]
                for key, value in items:
                    if key == 'confusion_matrix':
                        continue
                    row = value if isinstance(value, dict) else {'value': value, 'source': 'controller trace'}
                    writer.writerow([role, key, row.get('value'), row.get('numerator'), row.get('denominator'), row.get('source'), row.get('note')])
    all_metrics = []
    for run in sorted(root.parent.iterdir()):
        if run.is_dir() and (run/'report.json').exists():
            try:
                all_metrics.append(collect(run, read(run/'evaluation_labels.json', {})))
            except (ValueError, KeyError, TypeError):
                # A corrupt historical run cannot suppress the current report.
                try:
                    manifest = read(run/'manifest.json', {})
                    requirements_only = manifest.get('requirements_only', False)
                    agent_transport = transport_kind(run, manifest)
                except (ValueError, AttributeError):
                    requirements_only = False
                    agent_transport = 'unknown'
                all_metrics.append({'run_id': run.name, 'status': 'UNREADABLE', 'application': 'unknown',
                    'requirements_only': requirements_only, 'agent_transport': agent_transport,
                    'system': {}, 'agents': {}, 'elapsed_seconds': None})
    all_metrics = [m for m in all_metrics if m['agent_transport'] == metrics['agent_transport']]
    pipeline_runs = [m for m in all_metrics if not m['requirements_only']]
    requirements_runs = [m for m in all_metrics if m['requirements_only']]
    pipeline_labeled = [m['system']['independent_task_success'] for m in pipeline_runs
                        if m['system'].get('independent_task_success') is not None]
    requirements_labeled = [m['system']['independent_task_success'] for m in requirements_runs
                            if m['system'].get('independent_task_success') is not None]
    repaired = [m for m in pipeline_runs if m['system'].get('initial_pipeline_failure')]
    summary = {'agent_transport': metrics['agent_transport'], 'runs': len(all_metrics), 'status_counts': dict(Counter(m['status'] for m in all_metrics)),
        'pipeline_success_rate': rate(sum(m['status']=='PASS' for m in pipeline_runs), len(pipeline_runs), 'all end-to-end runs'),
        'independent_task_success_rate': rate(sum(pipeline_labeled), len(pipeline_labeled), 'independently labeled end-to-end runs'),
        'task_label_coverage': rate(len(pipeline_labeled), len(pipeline_runs), 'annotation coverage'),
        'requirements_only_success_rate': rate(sum(m['status']=='PASS' for m in requirements_runs), len(requirements_runs), 'all requirements-only runs'),
        'independent_requirements_success_rate': rate(sum(requirements_labeled), len(requirements_labeled), 'independently labeled requirements-only runs'),
        'requirements_label_coverage': rate(len(requirements_labeled), len(requirements_runs), 'annotation coverage'),
        'repair_success_rate': rate(sum(m['status']=='PASS' for m in repaired), len(repaired), 'pipeline outcomes after initial failure; not independent correctness'),
        'run_costs': [{'run_id': m['run_id'], 'status': m['status'], 'elapsed_seconds': m['elapsed_seconds'],
                       'requirements_only': m['requirements_only'],
                       'completion_tokens': sum(a['completion_tokens'] or 0 for a in m['agents'].values()) if any(a['completion_tokens'] is not None for a in m['agents'].values()) else None,
                       'independent_task_success': m['system'].get('independent_task_success')} for m in all_metrics]}
    for key, runs, outcome_key in [
        ('by_application', pipeline_runs, 'pipeline_success_rate'),
        ('by_requirements_application', requirements_runs, 'requirements_only_success_rate'),
    ]:
        summary[key] = {}
        for application in sorted({m['application'] for m in runs}):
            group = [m for m in runs if m['application'] == application]
            known = [m['system']['independent_task_success'] for m in group if m['system'].get('independent_task_success') is not None]
            summary[key][application] = {
                'runs': len(group), 'status_counts': dict(Counter(m['status'] for m in group)),
                outcome_key: rate(sum(m['status']=='PASS' for m in group), len(group), 'all application runs in this scope'),
                'independent_task_success_rate': rate(sum(known), len(known), 'independently labeled application runs in this scope')}
    (root/'research_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (root/'metrics.html').write_text(dashboard(metrics, summary), encoding='utf-8')
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory', type=Path)
    parser.add_argument('--labels', type=Path)
    args = parser.parse_args()
    generate(args.run_directory, args.labels)
    print(args.run_directory/'metrics.html')
