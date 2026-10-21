#!/usr/bin/env python3
"""Compatibility entry point for the current, verified revision-3 manuscript."""
import sys
from build_revision_paper import main

if __name__ == '__main__':
    sys.argv = ['--outputs' if arg == '--output' else arg for arg in sys.argv]
    main()
