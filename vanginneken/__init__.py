"""
Van Ginneken buffer placement algorithm implementation.

This package implements the Van Ginneken algorithm for optimal buffer placement
in VLSI routing trees. The algorithm places buffers along the routing tree to
maximize the Required Arrival Time (RAT) at the driver.

Modules:
    graph: Graph representation of the routing tree
    solver: Implementation of the Van Ginneken algorithm
    timings: Delay and capacitance models
    benchmark: Benchmarking utilities
"""

from vanginneken.graph import Graph, Node, Edge
from vanginneken.solver import place_buffers
from vanginneken.timings import Model
from vanginneken.benchmark import run_linear_benchmark

__all__ = [
    'Graph', 'Node', 'Edge',
    'place_buffers',
    'Model',
    'run_linear_benchmark'
]