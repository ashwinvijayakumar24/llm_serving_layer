"""Replica HTTP surface: OpenAI-compatible ingress, SSE, cancellation.

The scheduler runs as a cooperative asyncio task; a step must stay short for
the event loop to remain responsive, which is what chunked prefill bounds.
See docs/ARCHITECTURE.md §7.

NOTE: the phrase "OpenAI-compatible server" is spent on the ENGINE's resume
line. This one is "serving layer" / "routing layer". SERVING_INTERFACE.md:157.
"""
