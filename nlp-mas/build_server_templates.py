#!/usr/bin/env python
"""Prepend server-owned role instructions to each model's original chat template."""
import argparse
import json
from pathlib import Path
from transformers import AutoTokenizer


ROLES = ('requirements-engineer', 'test-engineer', 'verification-engineer')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('models', nargs=3)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    args.output.mkdir(parents=True, exist_ok=True)

    for role, model in zip(ROLES, args.models):
        tokenizer = AutoTokenizer.from_pretrained(model)
        original = tokenizer.get_chat_template()
        prompt = (base / 'prompts' / f'{role}.txt').read_text(encoding='utf-8')
        prefix = (
            '{% set messages = [{"role": "system", "content": '
            + json.dumps(prompt, ensure_ascii=True)
            + '}] + (messages[1:] if messages and messages[0]["role"] == "system" else messages) %}\n'
        )
        template = prefix + original

        # Exercise the actual tokenizer/template, including tools, before launching servers.
        rendered = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': 'Template smoke test'}],
            chat_template=template, tokenize=False, add_generation_prompt=True,
            tools=[{'type': 'function', 'function': {'name': 'check',
                   'description': 'Check setup', 'parameters': {'type': 'object', 'properties': {}}}}],
        )

        if prompt not in rendered or 'check' not in rendered:
            raise RuntimeError(f'Role prompt or tool missing from {role} template')

        (args.output / f'{role}.jinja').write_text(template, encoding='utf-8')
        # Snapshot the exact instructions used by this server run.
        (args.output / f'{role}.txt').write_text(prompt, encoding='utf-8')
        print(f'Prepared server-side prompt for {role}')


if __name__ == '__main__':
    main()
