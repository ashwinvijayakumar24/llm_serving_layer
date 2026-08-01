"""
uvicorn --factory entry for a router. Configured by environment because Slurm
launches it, not Python.

    ROUTER_POLICY    prefix_aware | least_outstanding | round_robin | consistent_hash
    ROUTER_REPLICAS  comma-separated replica base URLs

The policy name is REQUIRED and unknown names raise rather than defaulting: a
run whose artifact says `prefix_aware` while B5 was actually serving would look
like a null result rather than like a bug, and is the worst-shaped error this
project can make.
"""

import os

from serving.router.app import build_default_router


def app():
    policy = os.environ["ROUTER_POLICY"]
    replicas = [u for u in os.environ["ROUTER_REPLICAS"].split(",") if u]
    if not replicas:
        raise SystemExit("ROUTER_REPLICAS is empty")
    print(f"router: policy={policy} replicas={len(replicas)}", flush=True)
    return build_default_router(replicas, policy_name=policy)
