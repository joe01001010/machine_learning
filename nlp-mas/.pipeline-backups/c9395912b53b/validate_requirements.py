#!/usr/bin/env python

import argparse
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REQUIREMENTS_ENGINEER_ENDPOINT = "http://127.0.0.1:8001/v1/chat/completions"
REQUIREMENTS_ENGINEER_NAME = "requirements-engineer"
TEST_ENGINEER_ENDPOINT = "http://127.0.0.1:8002/v1/chat/completions"
TEST_ENGINEER_NAME = "test-engineer"
VERIFICATION_ENGINEER_ENDPOINT = "http://127.0.0.1:8003/v1/chat/completions"
VERIFICATION_ENGINEER_NAME = "verification-engineer"

BASE = Path(__file__).resolve().parent


def main():
    pass


if __name__ == "__main__":
    main()
