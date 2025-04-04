#!/usr/bin/env python3
"""
SyncSub Entry Point Script

This script initializes the CLI handler and runs the subtitle generation process.
"""

import sys
from syncsub.cli import CLIHandler

if __name__ == "__main__":
    # Basic check for minimal Python version if necessary
    if sys.version_info < (3, 7):
        sys.stderr.write("SyncSub requires Python 3.7 or later.\n")
        sys.exit(1)

    cli = CLIHandler()
    cli.run()