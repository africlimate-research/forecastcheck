# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
"""Run the suite as NO country, whatever this machine's ``.deployment`` says.

A user's machine is set up as one country (``echo senegal > .deployment``); a test must
mean the same on every machine, so under pytest the selection is pinned to ``none``. A
test that needs one country's facts sets ``VERIFICATION_COUNTRY`` (or
``VERIFICATION_DEPLOYMENT``) itself.
"""
import os

os.environ["VERIFICATION_DEPLOYMENT"] = "none"
