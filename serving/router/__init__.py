"""Prefix-aware router across N replicas.

AUTHOR-WRITTEN (routing policy). Phase 5.
Organizing principle: THE ROUTER HOLDS HINTS, THE REPLICA HOLDS TRUTH.
A stale hint costs cache locality, never correctness -- which is why there is
no consensus, no shared cache, no distributed transaction.

Developable against MOCK CPU REPLICAS: the policy is pure logic, so PACE queue
time never blocks router work (docs/RISK_REGISTER.md R21).
"""
