"""Invoke an explicitly configured Python application entry function once.

This is an invocation adapter, not application logic. Loading with a distinct
run name prevents an ordinary __main__ guard from calling the function twice.
"""
import argparse
import keyword
from pathlib import Path
import runpy
import sys


def _function_name(value):
    if not value.isidentifier() or keyword.iskeyword(value):
        raise argparse.ArgumentTypeError('Entry function must be a Python identifier')
    return value


def launch(path, args, entry_function='main'):
    """Load ``path``, call its configured function, and return an exit status.

    The application's stdout, stderr, exceptions, and SystemExit propagate
    naturally. None means success; an integer is used as the exit status.
    Caller argv and import paths are restored even when application code fails.
    """
    if not isinstance(entry_function, str) or not entry_function.isidentifier() or keyword.iskeyword(entry_function):
        raise ValueError('Entry function must be a Python identifier')
    application = Path(path).resolve(strict=True)
    if not application.is_file():
        raise ValueError('Application entrypoint must be a file')
    previous_argv, previous_path = sys.argv, sys.path[:]
    try:
        sys.argv = [str(application), *args]
        # Match normal script execution for sibling application imports.
        sys.path.insert(0, str(application.parent))
        namespace = runpy.run_path(str(application), run_name='__application__')
        entry = namespace.get(entry_function)
        if not callable(entry):
            raise TypeError(f'Application entry function {entry_function!r} is missing or not callable')
        status = entry()
        if status is None:
            return 0
        if not isinstance(status, int):
            raise TypeError('Application entry function must return an integer or None')
        return status
    finally:
        sys.argv = previous_argv
        sys.path[:] = previous_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--function', default='main', type=_function_name,
                        help='Application function to call once (default: main)')
    parser.add_argument('file', type=Path, help='Python application entrypoint')
    parser.add_argument('args', nargs=argparse.REMAINDER,
                        help='Arguments passed unchanged to the application')
    options = parser.parse_args(argv)
    return launch(options.file, options.args, options.function)


if __name__ == '__main__':
    raise SystemExit(main())
