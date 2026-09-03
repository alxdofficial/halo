"""ALAMEDA and COPS tooling, kept outside the Task-2 minimal set.

These are free-living state sources with no bounded executions, so they answer a
different question than Task 2 and were demoted to ``longitudinal_extension`` in
the adapter registry. The audit and encoder probe that target them live here so
that ``task2/`` contains only what the Task-2 protocol actually uses.
"""
