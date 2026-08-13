"""Modular RAG components, written out explicitly rather than via a framework.

Each module owns one stage of the pipeline and depends only on the shared
dataclasses in types.py, never on another stage's internals. That is what makes
the stages swappable one at a time in an experiment.
"""
