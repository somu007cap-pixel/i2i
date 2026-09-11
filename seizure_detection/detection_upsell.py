"""
Detection Upsell Entrypoint
-------------------------
This module provides a thin wrapper to invoke the main detection pipeline
via ``phase4_validation.main`` when executed as a script. It exists for
convenient command‑line access without importing the full pipeline.
"""

from phase4_validation import main

if __name__ == "__main__":
    main()
