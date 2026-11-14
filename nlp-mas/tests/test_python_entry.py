"""Behavioral checks for explicitly configured Python function entrypoints."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from run_python_app import launch

BOOTSTRAP = Path(__file__).resolve().parents[1] / 'run_python_app.py'


class PythonEntryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.app = Path(self.directory.name) / 'main.py'

    def tearDown(self):
        self.directory.cleanup()

    def invoke(self, source, *args, function='main'):
        self.app.write_text(source, encoding='utf-8')
        return subprocess.run([sys.executable, '-B', str(BOOTSTRAP), '--function', function,
                               str(self.app), *args], capture_output=True, text=True, timeout=5)

    def test_guarded_main_is_called_exactly_once(self):
        result = self.invoke("def main():\n    print('once')\n\nif __name__ == '__main__':\n    main()\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, 'once\n')
        self.assertEqual(result.stderr, '')

    def test_none_return_is_success(self):
        result = self.invoke("def main():\n    print('ready')\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, 'ready\n')

    def test_integer_return_is_the_exit_status(self):
        result = self.invoke("def main():\n    print('failure')\n    return 7\n")
        self.assertEqual(result.returncode, 7)
        self.assertEqual(result.stdout, 'failure\n')

    def test_system_exit_propagates(self):
        result = self.invoke("def main():\n    raise SystemExit(9)\n")
        self.assertEqual(result.returncode, 9)
        self.assertEqual(result.stdout, '')
        self.assertEqual(result.stderr, '')

    def test_application_exception_is_not_hidden(self):
        result = self.invoke("def main():\n    raise ValueError('application failed')\n")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('ValueError: application failed', result.stderr)

    def test_missing_or_noncallable_main_fails(self):
        for source in ('value = 3\n', 'main = 3\n'):
            with self.subTest(source=source):
                result = self.invoke(source)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('missing or not callable', result.stderr)

    def test_argument_list_is_preserved_including_negative_and_option_like_values(self):
        result = self.invoke("import json, sys\ndef main():\n    print(json.dumps(sys.argv))\n",
                             '-5+3', '--function', 'not-an-option-now')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), [str(self.app), '-5+3', '--function', 'not-an-option-now'])

    def test_custom_function_and_sibling_import_work(self):
        (self.app.parent / 'helper.py').write_text("MESSAGE = 'sibling works'\n", encoding='utf-8')
        result = self.invoke("from helper import MESSAGE\ndef start():\n    print(MESSAGE)\n", function='start')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, 'sibling works\n')

    def test_direct_launch_restores_caller_state_on_system_exit(self):
        self.app.write_text("def main():\n    raise SystemExit(4)\n", encoding='utf-8')
        previous_argv, previous_path = sys.argv, sys.path[:]
        with self.assertRaises(SystemExit) as raised:
            launch(self.app, ['input'])
        self.assertEqual(raised.exception.code, 4)
        self.assertIs(sys.argv, previous_argv)
        self.assertEqual(sys.path, previous_path)

    def test_top_level_call_remains_visible_as_an_application_defect(self):
        result = self.invoke("def main():\n    print('called')\nmain()\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, 'called\ncalled\n')

    def test_noninteger_return_fails_instead_of_claiming_success(self):
        result = self.invoke("def main():\n    return 'not an exit status'\n")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('must return an integer or None', result.stderr)

    def test_invalid_function_name_is_rejected(self):
        result = self.invoke("def main():\n    pass\n", function='main()')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('must be a Python identifier', result.stderr)


if __name__ == '__main__':
    unittest.main()
