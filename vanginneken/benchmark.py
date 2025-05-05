import vanginneken.graph as gr
import vanginneken.solver as solve
import vanginneken.timings as tm

import matplotlib.pyplot as plt
import numpy as np
import time
import csv
import os
from datetime import datetime

def run_linear_benchmark(bench_size: int, timings: tm.Model, 
                         num_of_runs=20, output_dir="benchmark_results") -> None:
    """
    Run a benchmark on linear wires of increasing length.
    """
    # Calculate step size for evenly spaced wire lengths
    num_of_iterations = min(bench_size, num_of_runs)
    step = max(1, int(bench_size / num_of_iterations))
    
    # Lists to store results
    wire_lengths = []
    execution_times = []
    rat_values = []
    buffer_counts = []
    
    print(f"Running benchmark with max wire length {bench_size}, {num_of_iterations} iterations")
    
    # Run benchmark for each wire length
    for length in range(step, bench_size + 1, step):
        print(f"Testing wire length {length}/{bench_size}:")
        
        # Create wire of specified length
        graph = gr.Graph.create_wire(length)
        
        # Measure execution time
        start_time = time.time()
        try:
            final_graph = solve.place_buffers(graph, timings, show_rat=False)
            
            # Calculate RAT at driver
            # We'll use a more robust approach here
            start_node_id = graph.get_start_node_id()
            sink_node_id = graph.get_node_neighbors(start_node_id)[0]  # Первый сосед - сток
            
            # Инициализируем решения для узлов-стоков
            solutions = {}
            sink_node = graph.get_node(sink_node_id)
            params = sink_node.get_param()
            solutions[sink_node_id] = [solve.BufferSolution(params.capacitance, params.rat)]
            
            # Вычисляем решения
            driver_solutions = solve.compute_node_solutions(final_graph, start_node_id, solutions, timings)
            best_solution = solve.add_buffer_at_driver(driver_solutions, timings, False)
            rat_at_driver = best_solution.rat
            
            # Count buffers added
            buffer_count = len(final_graph.nodes) - len(graph.nodes)
            
            # Record results
            execution_time = end_time = time.time() - start_time
            wire_lengths.append(length)
            execution_times.append(execution_time)
            rat_values.append(rat_at_driver)
            buffer_counts.append(buffer_count)
            
            print(f"  Time: {execution_time:.6f} sec")
            print(f"  RAT: {rat_at_driver:.2f}")
            print(f"  Buffers: {buffer_count}")
            
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"  Error at length {length}: {str(e)}")
            print(f"  Time: {execution_time:.6f} sec")
            # Continue to next length

    # Skip plotting if we have no data
    if not wire_lengths:
        print("No data points collected. Skipping plots.")
        return
        
    # Create directory for results if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save results to CSV
    csv_filename = f"{output_dir}/benchmark_results_{timestamp}.csv"
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Wire Length', 'Execution Time (s)', 'RAT', 'Buffer Count'])
        for i in range(len(wire_lengths)):
            writer.writerow([wire_lengths[i], execution_times[i], rat_values[i], buffer_counts[i]])
    
    # Plot execution time vs wire length
    plt.figure(figsize=(12, 8))
    
    # Set up 2x2 subplot grid
    plt.subplot(2, 2, 1)
    plt.plot(wire_lengths, execution_times, marker='o', linestyle='-', color='b')
    plt.title('Execution Time vs Wire Length')
    plt.xlabel('Wire Length')
    plt.ylabel('Execution Time (s)')
    plt.grid(True)
    
    # Add polynomial fit curve
    if len(wire_lengths) > 2:
        z = np.polyfit(wire_lengths, execution_times, 2)
        p = np.poly1d(z)
        x_smooth = np.linspace(min(wire_lengths), max(wire_lengths), 100)
        plt.plot(x_smooth, p(x_smooth), 'r--', linewidth=1)
        plt.legend(['Measured', f'Fit: {z[0]:.2e}x² + {z[1]:.2e}x + {z[2]:.2e}'])
    
    # Plot RAT vs wire length
    plt.subplot(2, 2, 2)
    plt.plot(wire_lengths, rat_values, marker='s', linestyle='-', color='g')
    plt.title('RAT vs Wire Length')
    plt.xlabel('Wire Length')
    plt.ylabel('RAT')
    plt.grid(True)
    
    # Plot buffer count vs wire length
    plt.subplot(2, 2, 3)
    plt.plot(wire_lengths, buffer_counts, marker='d', linestyle='-', color='r')
    plt.title('Buffer Count vs Wire Length')
    plt.xlabel('Wire Length')
    plt.ylabel('Number of Buffers')
    plt.grid(True)
    
    # Add linear fit for buffer count
    if len(wire_lengths) > 1:
        z = np.polyfit(wire_lengths, buffer_counts, 1)
        p = np.poly1d(z)
        x_smooth = np.linspace(min(wire_lengths), max(wire_lengths), 100)
        plt.plot(x_smooth, p(x_smooth), 'b--', linewidth=1)
        plt.legend(['Measured', f'Fit: {z[0]:.4f}x + {z[1]:.2f}'])
    
    # Plot delay per unit length
    plt.subplot(2, 2, 4)
    delays = [100 - rat for rat in rat_values]  # Assuming initial RAT was 100
    delays_per_unit = [delays[i]/wire_lengths[i] for i in range(len(delays))]
    plt.plot(wire_lengths, delays_per_unit, marker='^', linestyle='-', color='m')
    plt.title('Delay per Unit Length vs Wire Length')
    plt.xlabel('Wire Length')
    plt.ylabel('Delay per Unit Length')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/benchmark_plot_{timestamp}.png", dpi=300)
    plt.show()
    
    print(f"Benchmark results saved to {csv_filename}")
    print(f"Plots saved to {output_dir}/benchmark_plot_{timestamp}.png")