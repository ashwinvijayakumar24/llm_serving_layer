"""
uvicorn --factory entry for a router. Configured by environment because Slurm
launches it, not Python.

    ROUTER_POLICY    prefix_aware | least_outstanding | round_robin | consistent_hash
    ROUTER_REPLICAS  comma-separated replica base URLs
    ROUTER_BLEND     optional, prefix_aware only. Affinity/load weight in
                     score = blend*affinity - (1-blend)*load_norm. Defaults to
                     build_default_router's 0.7. blend=0 is asserted by test to
                     be exactly least_outstanding, so a sweep that includes 0
                     carries its own baseline.

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

    # Absent means "use the library default", not "use 0.0". A blend silently
    # defaulting to zero would make every arm of a blend sweep identical to B5
    # while the artifacts said prefix_aware -- a null result that looks like a
    # finding, which is the worst-shaped error this project can make.
    kwargs = {}
    raw = os.environ.get("ROUTER_BLEND")
    if raw is not None and raw != "":
        blend = float(raw)
        if not 0.0 <= blend <= 1.0:
            raise SystemExit(f"ROUTER_BLEND must be in [0,1], got {blend}")
        kwargs["blend"] = blend

    shown = kwargs.get("blend", "default(0.7)")
    print(f"router: policy={policy} blend={shown} replicas={len(replicas)}", flush=True)
    return build_default_router(replicas, policy_name=policy, **kwargs)
