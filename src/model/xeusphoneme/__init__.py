"""ESPnet compatibility: expose ``espnet2.legacy`` (or ``espnet``) as ``espnet_import``.

Only the ESPnet-derived XEUS / powsm modules need ESPnet; keeping the shim here
lets the SPAM, WavLM-free MFA and Koel paths import without ESPnet installed.
"""
import sys

try:
    import espnet2.legacy as _espnet
except ImportError:
    import espnet as _espnet

sys.modules.setdefault("espnet_import", _espnet)
