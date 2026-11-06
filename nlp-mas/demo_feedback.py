#!/usr/bin/env python3
"""Run a deterministic classroom demonstration through the experiment controller."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from textwrap import dedent

from experiment import BASE, Config, Experiment


@dataclass(frozen=True)
class Scenario:
    language: str
    entrypoint: str
    prompt: str
    draft_requirement: str
    requirement: str
    requirement_feedback: str
    original_test: dict
    revised_test: dict
    review_feedback: str
    source: str
    buggy_line: str
    fixed_line: str


def test(args, stdin, expected_stdout):
    return {'test_case_id': 'R001-T01', 'args': args, 'stdin': stdin,
            'expected_stdout': expected_stdout, 'expected_exit_code': 0}


SCENARIOS = {
    'calculator': Scenario(
        language='python',
        entrypoint='main.py',
        prompt='Accept one expression as a command-line argument. Apply multiplication before addition and print the integer result.',
        draft_requirement='The calculator shall evaluate the expression argument and print an integer result.',
        requirement='The calculator shall evaluate multiplication before addition in an expression argument and print the integer result.',
        requirement_feedback='The prompt explicitly requires multiplication before addition. Include that rule.',
        original_test=test(['2+3'], '', '5\n'),
        revised_test=test(['2+3*4'], '', '14\n'),
        review_feedback='Addition alone cannot distinguish the required precedence from left-to-right evaluation. Include 2+3*4.',
        source=dedent("""\
            from math import prod
            import sys

            expression = sys.argv[1]
            totals = [sum(map(int, group.split('+'))) for group in expression.split('*')]
            print(prod(totals))
            """),
        buggy_line=("totals = [sum(map(int, group.split('+'))) for group in expression.split('*')]\n"
                    "print(prod(totals))"),
        fixed_line=("totals = [prod(map(int, group.split('*'))) for group in expression.split('+')]\n"
                    "print(sum(totals))"),
    ),
    'rideshare': Scenario(
        language='c',
        entrypoint='rideshare.c',
        prompt=('Read base fare, price per minute, price per mile, miles, minutes, and minimum fare from stdin. '
                'Print max(minimum fare, base fare + minutes * minute price + miles * mile price) with two decimals.'),
        draft_requirement='The rideshare fare calculator shall print base fare plus time and distance charges, to two decimals.',
        requirement=('The rideshare fare calculator shall read fare and trip data from stdin and print the greater of the minimum fare '
                     'and base fare plus minutes times price per minute plus miles times price per mile, to two decimals.'),
        requirement_feedback='Specify the stdin interface and that the minimum fare overrides a smaller computed fare.',
        original_test=test([], '2.00\n0.50\n1.00\n10\n12\n10.00\n', '18.00\n'),
        revised_test=test([], '2.00\n0.50\n1.00\n10\n12\n20.00\n', '20.00\n'),
        review_feedback='The minimum is below the computed fare in this test. Set the minimum to 20.00 so the minimum-fare rule determines the result.',
        source=dedent("""\
            #include <stdio.h>

            int main(void) {
                double base, per_minute, per_mile, miles, minimum;
                int minutes;
                if (scanf("%lf %lf %lf %lf %d %lf",
                          &base, &per_minute, &per_mile, &miles, &minutes, &minimum) != 6) {
                    return 1;
                }
                double subtotal = base + per_minute * minutes + per_mile * miles;
                printf("%.2f\\n", subtotal);
                return 0;
            }
            """),
        buggy_line='printf("%.2f\\n", subtotal);',
        fixed_line='printf("%.2f\\n", subtotal < minimum ? minimum : subtotal);',
    ),
}


def completion(name, **arguments):
    return {'choices': [{'finish_reason': 'tool_calls', 'message': {
        'role': 'assistant',
        'tool_calls': [{'id': 'demo-call', 'type': 'function', 'function': {
            'name': name, 'arguments': json.dumps(arguments)}}],
    }}]}


def scripted_agents(scenario):
    def reply(role, request):
        state = json.loads(request['messages'][0]['content'])
        mode = state['mode']
        if mode == 'requirements':
            feedback = state.get('revision_feedback')
            if feedback and feedback != scenario.requirement_feedback:
                raise RuntimeError('Reviewer feedback did not reach the requirements engineer')
            text = scenario.requirement if feedback else scenario.draft_requirement
            print(f'  Requirements engineer {"revises" if feedback else "drafts"} — {text}')
            requirement = {'requirement_id': state['next_requirement_id'],
                           'source_id': state['source_ids'][0], 'text': text}
            return completion('submit_requirements', requirements=[requirement], complete=True)

        if mode == 'requirements_review':
            accepted = state['proposed_requirements'][0]['text'] == scenario.requirement
            decision = 'ACCEPT' if accepted else 'REVISE'
            feedback = 'The revised requirement includes the explicit rule.' if accepted else scenario.requirement_feedback
            return completion('review_requirements', decision=decision, feedback=feedback)

        if mode == 'test_design':
            revised = state['review_feedback'] is not None
            if revised and state['review_feedback'] != scenario.review_feedback:
                raise RuntimeError('Reviewer feedback did not reach the test designer')
            case = scenario.revised_test if revised else scenario.original_test
            print(f'  Test engineer {"revises" if revised else "proposes"} '
                  f'{case["test_case_id"]}, expecting {case["expected_stdout"].strip()}')
            rationale = 'Exercise the decisive condition identified by the reviewer.' if revised else 'Exercise a straightforward successful case.'
            return completion('submit_tests', tests=[{**case, 'justification': rationale}], rationale=rationale)

        if mode == 'test_review':
            proposed = state['tests'][0]
            accepted = all(proposed[key] == value for key, value in scenario.revised_test.items())
            decision = 'ACCEPT' if accepted else 'REVISE'
            feedback = 'The revised test exercises the required rule.' if accepted else scenario.review_feedback
            print(f'  Verification engineer {decision} — {feedback}')
            return completion('review_tests', decision=decision, feedback=feedback)

        if mode == 'coding':
            source = scenario.source
            if state['round']:
                feedback = state['feedback']
                if (len(feedback) != 1 or feedback[0].get('assessment', {}).get('verdict') != 'FAIL'
                        or any(feedback[0].get(key) != scenario.revised_test[key]
                               for key in ('test_case_id', 'args', 'stdin', 'expected_stdout'))):
                    raise RuntimeError('Frozen-test failure did not reach the coder')
                print(f'  Coder receives {feedback[0]["test_case_id"]} failure and fixes the rule.')
                source = source.replace(scenario.buggy_line, scenario.fixed_line)
            return completion('write_file', path=scenario.entrypoint, content=source.splitlines(),
                              mode='replace', complete=True)

        if mode == 'test_execution':
            return completion('run_test', test_case_id=state['test']['test_case_id'])

        if mode == 'verification':
            execution = state['execution']
            expected = state['test']['expected_stdout'].strip()
            actual = execution['stdout'].strip()
            verdict = 'PASS' if execution['host_match'] else 'FAIL'
            print(f'  Observed {actual!r}, frozen expectation {expected!r} — {verdict}')
            return completion('submit_verdict', verdict=verdict, criteria_justified=True,
                              explanation=f'Observed {actual!r} against the frozen expectation {expected!r}.')

        raise RuntimeError(f'Unexpected {role} task {mode}')

    return reply


def run_scenario(name, output):
    scenario = SCENARIOS[name]
    print(f'\n=== {name.capitalize()} feedback demonstration ===', flush=True)
    experiment = Experiment(Config(
        prompt=scenario.prompt,
        output=output / name,
        language=scenario.language,
        entrypoint=scenario.entrypoint,
        review_requirements=True,
        max_repairs=1,
    ), transport=scripted_agents(scenario))
    report = experiment.run()
    if report['status'] != 'PASS':
        return False

    for folder in ('requirements_review/block-001', 'test_design/R001'):
        decisions = [json.loads((experiment.root / folder / f'review-{i}.json').read_text(encoding='utf-8'))['decision']
                     for i in range(2)]
        if decisions != ['REVISE', 'ACCEPT']:
            raise RuntimeError(f'Expected one revision and one acceptance in {folder}, got {decisions}')

    verdicts = [row['tests'][0]['verdict']['verdict'] for row in report['rounds']]
    if verdicts != ['FAIL', 'PASS']:
        raise RuntimeError(f'Expected one failed round and one passing repair, got {verdicts}')
    print(f'  Verified FAIL → PASS on {experiment.tests[0]["test_case_id"]} using one frozen plan.')
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scenario', choices=['both', *SCENARIOS], nargs='?', default='both')
    parser.add_argument('--output', type=Path, default=BASE / 'results-demo')
    args = parser.parse_args()
    names = SCENARIOS if args.scenario == 'both' else [args.scenario]

    print('Scripted role responses, real controller and program execution. Each run covers one narrow requirement.')
    results = [run_scenario(name, args.output) for name in names]
    return 0 if all(results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
