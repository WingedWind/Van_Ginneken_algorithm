import vanginneken.graph as gr
import vanginneken.timings as tm
import copy

class BufferSolution:
    """
    Represents a solution for buffer placement with capacitance, RAT, and buffer positions.
    Each solution represents a candidate buffer insertion strategy for a given point in the network.
    """
    def __init__(self, capacitance: float, rat: float) -> None:
        self.capacitance = capacitance
        self.rat = rat
        self.buffer_positions = []
        self.distance_to_prev_buffer = 0
        self.rat_at_prev_buffer = rat
        self.load_capacitance = capacitance
  
    def is_dominated_by(self, other_solution) -> bool:
        """
        Checks if this solution is dominated by another solution.
        A solution is dominated if it has worse (higher) capacitance AND worse (lower) RAT.
        """
        return (other_solution.capacitance <= self.capacitance and 
                other_solution.rat >= self.rat)
  
    def is_better_than(self, other_solution) -> bool:
        """
        Checks if this solution has a better (higher) RAT than another solution.
        """
        return self.rat > other_solution.rat
  
    def get_distance_to_prev_buffer(self) -> int:
        """Returns the distance to the previous buffer."""
        return self.distance_to_prev_buffer
  
    def get_rat_at_prev_buffer(self) -> float:
        """Returns the RAT at the previous buffer."""
        return self.rat_at_prev_buffer
  
    def get_rat(self) -> float:
        """Returns the RAT at the current point."""
        return self.rat
  
    def get_load_capacitance(self) -> float:
        """Returns the load capacitance."""
        return self.load_capacitance
  
    def update(self, buffer_pos: list, capacitance: float, rat: float, distance: int) -> None:
        """
        Updates the solution with new values.
        If buffer_pos is provided, adds a new buffer to the solution.
        """
        self.rat = rat
        self.capacitance = capacitance
        self.distance_to_prev_buffer = distance
        if len(buffer_pos) == 2:  # If valid buffer position coordinates are provided
            self.load_capacitance = capacitance
            self.buffer_positions.append(buffer_pos)
            self.rat_at_prev_buffer = rat
  
    @staticmethod
    def combine(solution1, solution2):
        """
        Combines two solutions from different branches.
        The resulting solution has the sum of capacitances and minimum RAT.
        """
        new_solution = BufferSolution(
            capacitance=solution1.capacitance + solution2.capacitance,
            rat=min(solution1.rat, solution2.rat)
        )
        new_solution.buffer_positions = solution1.buffer_positions + solution2.buffer_positions
        new_solution.distance_to_prev_buffer = min(
            solution1.distance_to_prev_buffer, 
            solution2.distance_to_prev_buffer
        )
        return new_solution
  
# Helper functions for traversing the graph
def get_traversal_order(graph: gr.Graph) -> list:
    """
    Determines the order in which nodes should be traversed.
    Returns a list of node IDs in postorder traversal (from sinks to source).
    """
    start_id = graph.get_start_node_id()
    traversal_order = []
    visited = {}

    stack = [graph.get_node(start_id).get_id()]

    while stack:
        node_id = stack.pop()
        if node_id in visited:
            continue

        # Insert at beginning for postorder
        traversal_order.insert(0, node_id)
        neighbors = graph.get_node_neighbors(node_id)
        for neighbor_id in neighbors:
            stack.append(graph.nodes[neighbor_id].get_id())  
        visited[node_id] = True
  
    return traversal_order

def evaluate_buffer_options(prev_solution: BufferSolution, current_pos: list, timings: tm.Model) -> list:
    """
    Evaluates two options at a given position:
    1. Continue without placing a buffer
    2. Place a buffer at the current position
    Returns a list containing both solutions.
    """
    distance = prev_solution.get_distance_to_prev_buffer()
    load_capacitance = prev_solution.get_load_capacitance()
    
    # Calculate wire capacitance for current segment
    wire_cap = timings.wire_capacitance(distance)
    
    # Option 1: No buffer at this position
    solution_without_buffer = copy.deepcopy(prev_solution)
    new_capacitance = load_capacitance + timings.wire_capacitance(distance + 1)
    # Calculate delay without buffer
    delay_without_buffer = timings.wire_delay(L=distance+1, C=load_capacitance)
    rat_without_buffer = prev_solution.get_rat_at_prev_buffer() - delay_without_buffer
    solution_without_buffer.update([], new_capacitance, rat_without_buffer, distance+1)

    # Option 2: Insert buffer at this position
    solution_with_buffer = copy.deepcopy(prev_solution)
    # Calculate load seen by the buffer (wire + downstream capacitance)
    buffer_load = load_capacitance + wire_cap
    # Calculate delay with buffer
    wire_delay = timings.wire_delay(L=distance+1, C=load_capacitance)
    buffer_delay = timings.buffer_delay(capacitance=buffer_load)
    rat_with_buffer = prev_solution.get_rat_at_prev_buffer() - wire_delay - buffer_delay
    # New capacitance is buffer input capacitance + wire capacitance for 1 unit
    new_cap_with_buffer = timings.get_buffer_capacitance() + timings.wire_capacitance(1)
    solution_with_buffer.update(current_pos, new_cap_with_buffer, rat_with_buffer, 1)

    return [solution_without_buffer, solution_with_buffer]

def compute_solutions_along_wire(wire: gr.Edge, end_solutions: list, timings: tm.Model) -> list:
    """
    Computes possible solutions for each possible buffer position along a wire.
    Starts from the sink (end of wire) and works toward the source.
    """
    wire_length = wire.get_len()
    current_solutions = end_solutions
    
    if wire_length == 0:
        return end_solutions.copy()

    # Process each possible buffer position along the wire
    for offset_from_end in range(wire_length):
        new_solutions = []
        for solution in current_solutions:
            # Get coordinates at current position
            pos = wire.get_position_by_offset(offset_from_end)
            # Evaluate options at this position
            new_solutions.extend(evaluate_buffer_options(solution, pos, timings))
        # Prune dominated solutions to keep solution set manageable
        current_solutions = prune_dominated_solutions(new_solutions)
    
    return current_solutions

def prune_dominated_solutions(all_solutions: list) -> list:
    """
    Removes dominated solutions from the solution set.
    A solution is dominated if there exists another solution with 
    better (or equal) capacitance AND better (or equal) RAT.
    """
    indices_to_remove = set()
    
    # Compare each pair of solutions
    for i in range(len(all_solutions)):
        for j in range(i + 1, len(all_solutions)):
            sol_i = all_solutions[i]
            sol_j = all_solutions[j]
            
            if sol_i.is_dominated_by(sol_j):
                indices_to_remove.add(i)
                continue
            if sol_j.is_dominated_by(sol_i):
                indices_to_remove.add(j)
    
    # Build the pruned list excluding dominated solutions
    pruned_solutions = []
    for i in range(len(all_solutions)):
        if i not in indices_to_remove:
            pruned_solutions.append(all_solutions[i])
            
    return pruned_solutions

def combine_branch_solutions(current_solutions: list, branch_solutions: list) -> list:
    """
    Combines solutions from the current node with solutions from a branch.
    If current_solutions is empty, simply returns branch_solutions.
    """
    if not current_solutions:
        return branch_solutions
        
    combined_solutions = []
    for current_sol in current_solutions:
        for branch_sol in branch_solutions:
            combined_solutions.append(BufferSolution.combine(current_sol, branch_sol))
            
    return prune_dominated_solutions(combined_solutions)

def combine_all_branch_solutions(all_branch_solutions: list) -> list:
    """
    Combines solutions from all branches connected to a node.
    """
    combined_solutions = []
    for branch_solutions in all_branch_solutions:
        combined_solutions = combine_branch_solutions(combined_solutions, branch_solutions)
    return combined_solutions

def compute_node_solutions(graph: gr.Graph, node_id: int, 
                         visited_nodes: dict, timings: tm.Model) -> list:
    """
    Computes all possible solutions for a given node.
    """
    # If we've already processed this node, return cached results
    if node_id in visited_nodes:
        return visited_nodes[node_id]
        
    neighbors = graph.get_node_neighbors(node_id)
    # Solutions from all branches connected to this node
    all_branch_solutions = []
    
    for neighbor_id in neighbors:
        # Only consider neighbors we've already visited
        if neighbor_id in visited_nodes:
            edge = graph.get_edge_between_nodes(node_id, neighbor_id)
            # Compute solutions along the wire connecting to this neighbor
            solutions_for_wire = compute_solutions_along_wire(
                edge, visited_nodes[neighbor_id], timings
            )
            all_branch_solutions.append(solutions_for_wire)
            
    return prune_dominated_solutions(combine_all_branch_solutions(all_branch_solutions))

def add_buffer_at_driver(solutions: list, timings: tm.Model, show_rat: bool) -> BufferSolution:
    """
    Adds a buffer at the driver location and selects the best solution.
    """
    # Проверка на пустой список решений
    if not solutions:
        # Создаем базовое решение с дефолтными значениями
        default_solution = BufferSolution(
            capacitance=timings.get_buffer_capacitance(),
            rat=0.0  # Нулевой RAT для обозначения невозможности найти решение
        )
        if show_rat:
            print("Warning: No solutions found. Using default solution.")
            print(f"RAT at driver: {default_solution.rat}")
        return default_solution
        
    for solution in solutions:
        L = solution.get_distance_to_prev_buffer()
        C_load_wire = solution.get_load_capacitance()
        C_load_buf = solution.get_load_capacitance() + timings.wire_capacitance(L)

        rat_at_prev_buf = solution.get_rat_at_prev_buffer()
        # Calculate delay from placing buffer at driver
        wire_delay = timings.wire_delay(L=L, C=C_load_wire) 
        buffer_delay = timings.buffer_delay(capacitance=C_load_buf)
        
        # Update solution with driver buffer
        rat_with_buf = rat_at_prev_buf - wire_delay - buffer_delay
        C_with_buf = timings.get_buffer_capacitance()
        solution.update([], C_with_buf, rat_with_buf, 0)

    # Find solution with best RAT
    best_solution = solutions[0]
    for solution in solutions:
        if solution.is_better_than(best_solution):
            best_solution = solution
            
    if show_rat:
        print(f"RAT at driver: {best_solution.rat}")
        
    return best_solution

def create_final_graph(graph: gr.Graph, best_solution: BufferSolution, show_buffers: bool) -> gr.Graph:
    """
    Creates a new graph with buffers inserted according to the best solution.
    """
    # Create a deep copy of the original graph
    new_graph = copy.deepcopy(graph)
    
    buffer_count = 0
    if show_buffers:
        print("Buffers inserted at:")
        
    # Add all buffers from the solution
    for buf_pos in best_solution.buffer_positions:
        new_graph.add_buffer(buf_pos)
        if show_buffers:
            print(f"\t{buf_pos}")
        buffer_count += 1
        
    if show_buffers:
        print(f"Total buffers: {buffer_count}")
        
    return new_graph

def initialize_sink_solutions(graph: gr.Graph, traversal_order: list) -> dict:
    """
    Initializes solutions at sink nodes with their capacitance and RAT values.
    """
    initial_solutions = {}
    for node_id in traversal_order:
        node = graph.get_node(node_id)
        params = node.get_param()
        if params is not None:  # Node is a sink
            initial_solutions[node_id] = [BufferSolution(params.get_capacitance(), params.get_rat())]
    return initial_solutions

def place_buffers(graph: gr.Graph, timings: tm.Model, 
                show_buffers: bool = False, show_rat: bool = False) -> gr.Graph:
    """
    Main function to place buffers in the graph using Van Ginneken's algorithm.
    """
    # Get traversal order (postorder: from sinks to source)
    traversal_order = get_traversal_order(graph)
    
    # Initialize solutions at sink nodes
    visited_nodes = initialize_sink_solutions(graph, traversal_order)
    
    # Process nodes in postorder
    for node_id in traversal_order:
        solutions = compute_node_solutions(graph, node_id, visited_nodes, timings)
        visited_nodes[node_id] = solutions
        
    # Get solutions at the driver (root) node
    solutions_at_driver = visited_nodes[graph.get_start_node_id()]
    
    # Find the best solution and create final graph
    best_solution = add_buffer_at_driver(solutions_at_driver, timings, show_rat)
    return create_final_graph(graph, best_solution, show_buffers)

# Export only the public API
__all__ = ['place_buffers']