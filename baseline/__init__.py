"""Phase 1 baseline-testing toolkit for CheXagent-2-3b on chest X-ray VQA.

This package is split into GPU-free parts (dataset preparation, question
categorization, answer scoring) that can be developed and unit-tested on any
machine, and a GPU-dependent part (`model.py`) that only needs to run where a
GPU is actually available (e.g. a free Google Colab T4 runtime).
"""
