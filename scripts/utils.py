#!/usr/bin/env python3
"""
Utility functions shared across scripts.
"""

import sys
import getpass
from pathlib import Path
from datetime import datetime


def log_command(logfile='logfile'):
    """
    Log the command execution to a file.

    Args:
        logfile: Name of the log file (default: 'logfile')
                 File is created in the repository root directory.

    Log format: YYYY-MM-DD HH:MM:SS | username | full command
    """
    try:
        username = getpass.getuser()
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if '-h' in sys.argv or '--help' in sys.argv:
            # Don't print anything if the user's looking for help
            return
        command = ' '.join(sys.argv)

        log_entry = f"{timestamp} | {username} | {command}\n"

        logfile_path = Path(__file__).parent.parent / logfile
        with open(logfile_path, 'a') as f:
            f.write(log_entry)
    except Exception as e:
        print(f"Warning: Could not write to logfile: {e}", file=sys.stderr)
