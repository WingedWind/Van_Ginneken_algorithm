import argparse
import os

import vanginneken

def main():
    """
    Main entry point for the Van Ginneken buffer placement algorithm.
    Parses command line arguments and executes the appropriate functions.
    """
    parser = argparse.ArgumentParser(description="Van Ginneken buffers placement algorithm")
    parser.add_argument("input_files", nargs='+', help="Technology file name; Input file name")
    parser.add_argument("--file-to-dump", dest='file_to_dump', type=str, 
                        help="Base name for output DOT files")
    parser.add_argument("--linear-bench-size", dest='bench_size', type=int, 
                        help='Max length of wire in linear benchmark', default=0)
    parser.add_argument("--show-inserted-buffers", dest="show_bufs", action="store_true",
                        help="Show locations of inserted buffers")
    parser.add_argument("--show-final-rat", dest="show_rat", action="store_true",
                        help="Show final RAT value at driver")
    parser.add_argument("--report-rat", dest="report_rat", action="store_true",
                        help="Report RAT for all configurations")
    args = parser.parse_args()

    bench_size = args.bench_size
    
    if len(args.input_files) == 0:
        raise ValueError("No input files have been specified")

    # Load timing model from the first input file
    timing_model = vanginneken.Model()
    timing_model.load_from_json(args.input_files[0])

    # Run linear benchmark if bench_size is specified
    if bench_size != 0:
        vanginneken.run_linear_benchmark(bench_size, timing_model)
        return

    if len(args.input_files) != 2:
        raise ValueError("Invalid number of files. Expected: [technology_file] [input_file]")

    input_file = args.input_files[1]

    # Load input graph
    graph = vanginneken.Graph()
    graph.load_from_json(input_file)
    
    # Output initial graph in DOT format if requested
    if args.file_to_dump:
        graph.dump_to_dot(args.file_to_dump + "_in.dot")

    # Run buffer placement algorithm
    out_file = os.path.splitext(os.path.basename(input_file))[0]
    show_rat = args.show_rat or args.report_rat  # Enable RAT display if reporting is requested
    
    final_graph = vanginneken.place_buffers(graph, timing_model, args.show_bufs, show_rat)
    final_graph.store_to_json(out_file + ".json")

    # Output final graph in DOT format if requested
    if args.file_to_dump:
        final_graph.dump_to_dot(args.file_to_dump + "_out.dot")
    
    # Report RAT values in a format suitable for analysis if requested
    if args.report_rat:
        print(f"Configuration: {out_file}")
        print(f"Input nodes: {len(graph.nodes)}")
        print(f"Output nodes: {len(final_graph.nodes)}")
        print(f"Buffers added: {len(final_graph.nodes) - len(graph.nodes)}")
        print(f"------------")

if __name__ == "__main__":
    main()