"""The village dashboard: FastAPI backend plus the canvas frontend.

Importing this package does not pull in the pipeline — ``web.app`` imports
``web.jobs`` lazily, so serving a page never pays for onnxruntime.
"""

from web.app import app, serve

__all__ = ["app", "serve"]
