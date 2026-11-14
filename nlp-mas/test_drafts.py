"""Bounded, host-directed drafting of exact test inputs and outputs.

Draft pieces are immutable audit events. They do not become runnable tests
until the complete batch passes the caller's submission/review callback.
"""
from copy import deepcopy
import re

from jsonschema import ValidationError, validate

MAX_TESTS = 3
MAX_LINES = 64
MAX_LINE_LENGTH = 160
LINES_PER_CHUNK = 8
_LINE_ENDINGS = {'LF': '\n', 'CRLF': '\r\n', 'CR': '\r', 'NONE': ''}


def _object(properties):
    return {'type': 'object', 'properties': properties,
            'required': list(properties), 'additionalProperties': False}


def _text(**limits):
    return {'type': 'string', **limits}


def _tool(name, description, schema, handler):
    return ({'type': 'function', 'function': {
        'name': name, 'description': description, 'parameters': schema, 'strict': True}}, handler)


class TestDraft:
    """A deterministic PLAN -> DESCRIBE -> LINES test-drafting protocol."""
    def __init__(self, rid, submit, save, prefix):
        if not isinstance(rid, str) or not re.fullmatch(r'R[0-9]{3}', rid):
            raise ValueError('Test drafts require an Rnnn requirement ID')
        self.rid = rid
        self.submit = submit
        self.save = save
        self.prefix = str(prefix).rstrip('/')
        self.event_count = 0
        self.lines_per_chunk = LINES_PER_CHUNK
        self._state = self._empty_state()

    @staticmethod
    def _empty_state():
        return {'phase': 'PLAN', 'test_count': None, 'rationale': None,
                'completed': [], 'header': None,
                'buffers': {'stdin': [], 'expected_stdout': []}, 'field': None}

    @property
    def phase(self):
        return self._state['phase']

    @property
    def done(self):
        return self.phase == 'DONE'

    def _target_id(self, state=None):
        state = self._state if state is None else state
        if state['phase'] in ('PLAN', 'DONE', 'SUBMIT'):
            return None
        return f'{self.rid}-T{len(state["completed"]) + 1:02d}'

    def _require_phase(self, phase):
        if self.phase != phase:
            raise ValueError(f'This action is unavailable in phase {self.phase}; use the currently offered tool')

    @staticmethod
    def _plan_schema():
        return _object({'test_count': {'type': 'integer', 'minimum': 1, 'maximum': MAX_TESTS},
                        'rationale': _text(minLength=1, maxLength=160)})

    @staticmethod
    def _header_schema():
        properties = {
            'args': {'type': 'array', 'maxItems': 12,
                     'description': 'Actual process argv tokens only, excluding the program name. Use [] unless a documented command-line flag or positional argument is needed. Never put scenario descriptions or stdin answers here.',
                     'items': _text(maxLength=160, description='One literal command-line argument token, such as --test-minutes or 12; not narrative prose.')},
            'expected_exit_code': {'type': 'integer', 'minimum': 0, 'maximum': 255},
            'justification': _text(minLength=1, maxLength=240,
                description='Describe the actual single fresh-process scenario represented by this test and how it exercises the assigned requirement; do not combine separate sessions.'),
        }
        for field in ('stdin', 'stdout'):
            properties[f'{field}_line_count'] = {
                'type': 'integer', 'minimum': 0, 'maximum': MAX_LINES,
                'description': ('Exact number of input lines supplied to ONE fresh process, including blank lines; not a count of scenarios or attempts.' if field == 'stdin' else
                                'Exact number of expected output lines from that SAME fresh-process session, including blank lines; not a count of scenarios.')}
            properties[f'{field}_line_ending'] = _text(enum=list(_LINE_ENDINGS))
            properties[f'{field}_final_newline'] = {'type': 'boolean'}
        return _object(properties)

    def _remaining(self):
        field = self._state['field']
        if self.phase != 'LINES' or field is None:
            return 0
        name = 'stdin' if field == 'stdin' else 'stdout'
        return self._state['header'][f'{name}_line_count'] - len(self._state['buffers'][field])

    def _line_schema(self):
        count = min(self.lines_per_chunk, self._remaining())
        return _object({'lines': {'type': 'array', 'minItems': count, 'maxItems': count,
            'items': _text(maxLength=MAX_LINE_LENGTH, pattern=r'^[^\r\n]*$')}})

    def _commit(self, event, data, state, header=None):
        sequence = self.event_count + 1
        record = {'event': event, 'sequence': sequence, 'requirement_id': self.rid,
                  'data': deepcopy(data), 'phase': state['phase'],
                  'header': deepcopy(state['header'] if header is None else header),
                  'completed_tests': deepcopy(state['completed'])}
        self.save(f'{self.prefix}/event{sequence:03d}.json', record)
        self.event_count = sequence
        self._state = state

    @staticmethod
    def _assemble(lines, ending, final_newline):
        if not lines:
            return ''
        separator = _LINE_ENDINGS[ending]
        return separator.join(lines) + (separator if final_newline else '')

    def _advance(self, state):
        header = state['header']
        if header is None:
            return
        if len(state['buffers']['stdin']) < header['stdin_line_count']:
            state.update(phase='LINES', field='stdin')
            return
        if len(state['buffers']['expected_stdout']) < header['stdout_line_count']:
            state.update(phase='LINES', field='expected_stdout')
            return
        state['completed'].append({
            'test_case_id': header['test_case_id'], 'args': list(header['args']),
            'stdin': self._assemble(state['buffers']['stdin'], header['stdin_line_ending'],
                                    header['stdin_final_newline']),
            'expected_stdout': self._assemble(state['buffers']['expected_stdout'],
                header['stdout_line_ending'], header['stdout_final_newline']),
            'expected_exit_code': header['expected_exit_code'],
            'justification': header['justification'],
        })
        state.update(header=None, field=None, buffers={'stdin': [], 'expected_stdout': []})
        state['phase'] = 'SUBMIT' if len(state['completed']) == state['test_count'] else 'DESCRIBE'

    def _submit_if_ready(self):
        if self.phase != 'SUBMIT':
            return
        tests, rationale = deepcopy(self._state['completed']), self._state['rationale']
        try:
            self.submit(tests, rationale)
        except (ValueError, ValidationError) as exc:
            self._commit('submission_rejected', {'error': str(exc),
                'tests': tests, 'rationale': rationale}, self._empty_state())
            raise
        state = deepcopy(self._state)
        state['phase'] = 'DONE'
        self._commit('submission_accepted', {'test_case_ids': [t['test_case_id'] for t in tests]}, state)

    def plan_tests(self, test_count, rationale):
        self._require_phase('PLAN')
        data = {'test_count': test_count, 'rationale': rationale}
        validate(data, self._plan_schema())
        state = self._empty_state()
        state.update(phase='DESCRIBE', test_count=test_count, rationale=rationale)
        self._commit('plan_tests', data, state)
        return self.context()

    def describe_test(self, args, expected_exit_code, justification,
                      stdin_line_count, stdin_line_ending, stdin_final_newline,
                      stdout_line_count, stdout_line_ending, stdout_final_newline):
        self._require_phase('DESCRIBE')
        data = dict(args=args, expected_exit_code=expected_exit_code, justification=justification,
            stdin_line_count=stdin_line_count, stdin_line_ending=stdin_line_ending,
            stdin_final_newline=stdin_final_newline, stdout_line_count=stdout_line_count,
            stdout_line_ending=stdout_line_ending, stdout_final_newline=stdout_final_newline)
        validate(data, self._header_schema())
        header = {'test_case_id': self._target_id(), **deepcopy(data)}
        state = deepcopy(self._state)
        state['header'] = header
        self._advance(state)
        self._commit('describe_test', data, state, header=header)
        self._submit_if_ready()
        return self.context()

    def append_test_lines(self, lines):
        self._require_phase('LINES')
        validate({'lines': lines}, self._line_schema())
        if any('\n' in line or '\r' in line for line in lines):
            raise ValueError('Each item must be exactly one line, without embedded CR or LF; blank lines and tabs are preserved')
        state = deepcopy(self._state)
        field = state['field']
        data = {'test_case_id': state['header']['test_case_id'], 'field': field,
                'start_line': len(state['buffers'][field]) + 1, 'lines': list(lines)}
        header = deepcopy(state['header'])
        state['buffers'][field].extend(lines)
        self._advance(state)
        self._commit('append_test_lines', data, state, header=header)
        self._submit_if_ready()
        return self.context()

    def tools(self):
        if self.phase == 'PLAN':
            return [_tool('plan_tests',
                'Choose 1-3 tests. Prefer the minimum number of adequate scenarios so exact input/output can be drafted within the call budget.',
                self._plan_schema(), self.plan_tests)]
        if self.phase == 'DESCRIBE':
            return [_tool('describe_test',
                f'Describe only metadata for host-assigned test {self._target_id()}. Declare exact line counts and separators; the host will request input/output lines next. Do not put line content in this call.',
                self._header_schema(), self.describe_test)]
        if self.phase == 'LINES':
            count = min(self.lines_per_chunk, self._remaining())
            return [_tool('append_test_lines',
                f'Supply exactly the next {count} lines for {self._target_id()} field {self._state["field"]}. Each item is one line, without CR/LF characters. Preserve blank lines and tabs; the host controls order and advances automatically.',
                self._line_schema(), self.append_test_lines)]
        return []

    def shrink_chunk(self):
        """Reduce a truncated line chunk without discarding accepted draft data."""
        if self.phase != 'LINES':
            return self.context()
        previous = self.lines_per_chunk
        following = max(1, previous // 2)
        if following != previous:
            self._commit('shrink_chunk', {'previous_chunk_limit': previous,
                'next_chunk_limit': following}, deepcopy(self._state))
            self.lines_per_chunk = following
        return self.context()

    def context(self):
        field = self._state['field']
        lines = self._state['buffers'][field] if self.phase == 'LINES' and field else []
        name = 'stdin' if field == 'stdin' else 'stdout'
        total = self._state['header'][f'{name}_line_count'] if self.phase == 'LINES' else None
        state = {'phase': self.phase, 'target_test_case_id': self._target_id(),
                 'field': field, 'next_line': len(lines) + 1 if self.phase == 'LINES' else None,
                 'total_lines': total, 'next_chunk_count': min(self.lines_per_chunk, self._remaining()),
                 'header': deepcopy(self._state['header']),
                 'completed_test_ids': [t['test_case_id'] for t in self._state['completed']],
                 'completed_tests': [{'test_case_id': t['test_case_id'],
                                      'args': list(t['args']), 'justification': t['justification']}
                                     for t in self._state['completed']],
                 'last_lines': list(lines[-2:])}
        if self.phase == 'LINES' and field == 'expected_stdout':
            # Expected output must be grounded in the input the model actually
            # drafted; source state is compacted between individual tool calls.
            state['stdin_lines'] = list(self._state['buffers']['stdin'])
        return state
