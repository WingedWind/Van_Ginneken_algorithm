import json

class Node:
    """
    Represents a node in the routing tree.
    
    Types of nodes:
    - "t": Terminal node (sink)
    - "s": Steiner point (routing junction)
    - "b": Buffer node (driver or inserted buffer)
    """
    # Node type constants
    TERMINAL_TYPE = "t"
    STEINER_TYPE = "s"
    BUFFER_TYPE = "b"

    class Parameters:
        """
        Holds electrical parameters for terminal nodes.
        """
        def __init__(self, capacitance: float, rat: float) -> None:
            """
            Initialize node parameters.
            """
            self.capacitance = capacitance
            self.rat = rat

        def get_capacitance(self) -> float:
            """Returns the capacitance value."""
            return self.capacitance
        
        def get_rat(self) -> float:
            """Returns the Required Arrival Time (RAT) value."""
            return self.rat
        
        def get_as_map(self) -> dict:
            """Returns parameters as a dictionary for JSON serialization."""
            return {
                "capacitance": self.capacitance,
                "rat": self.rat
            }

    def __init__(self, id: int, x: int, y: int, type: str, name: str) -> None:
        """
        Initialize a node in the routing tree.
        """
        self.id = id
        self.x = x
        self.y = y
        self.type = type
        self.name = name
        self.parameters = None
        # List of connected edge IDs
        self.edges = []

    def get_as_map(self) -> dict:
        """Returns node as a dictionary for JSON serialization."""
        res = {
            "id": self.id,
            "x": self.x,
            "y": self.y,
            "type": self.type,
            "name": self.name
        }
        if self.parameters is not None:
            res.update(self.parameters.get_as_map())
        return res

    def is_buffer(self) -> bool:
        """Checks if this node is a buffer."""
        return self.type == Node.BUFFER_TYPE

    def get_position(self) -> list:
        """Returns the [x, y] coordinates of the node."""
        return [self.x, self.y]
    
    def get_id(self) -> int:
        """Returns the node ID."""
        return self.id

    def get_param(self) -> Parameters:
        """Returns the electrical parameters of the node."""
        return self.parameters
    
    def get_edges(self) -> list[int]:
        """Returns the list of edge IDs connected to this node."""
        return self.edges
   
    def add_parameters(self, capacitance: float, rat: float) -> None:
        """
        Adds electrical parameters to a terminal node.
        """
        if self.type != Node.TERMINAL_TYPE:
            raise ValueError(f"only terminal nodes can have parameters: node [{self.id}]")
        self.parameters = Node.Parameters(capacitance=capacitance, rat=rat)

    def add_edge(self, edge_id: int) -> None:
        """
        Add an edge connection to this node.
        """
        self.edges.append(edge_id)
    
    def get_shape(self) -> str:
        """
        Returns the visualization shape for DOT file representation.
        
        Returns:
            Shape string for graphviz/DOT format
        """
        types_to_shapes = {
            "s": "circle", 
            "t": "rect",
            "b": "triangle"
        }
        return types_to_shapes[self.type]
  
class Edge:
    """
    Represents an edge (wire) in the routing tree.
    Each edge connects two nodes and consists of one or more orthogonal segments.
    """
    def __init__(self, id: int, start_vert_id: int, end_vert_id: int) -> None:
        """
        Initialize an edge in the routing tree.
        """
        self.id = id
        self.vertices = {
            "start": start_vert_id, 
            "end": end_vert_id
        }
        self.segments = []
        # Maps to track position-to-offset and offset-to-position
        self._points_to_offsets = {}
        self._offsets_to_points = {}
        self._len = 0
  
    @staticmethod
    def _calc_len(lhs: list, rhs: list) -> int:
        """
        Calculate Manhattan distance between two points.
        """
        if len(lhs) != 2 or len(rhs) != 2:
            raise ValueError("Invalid point coordinates - must have exactly 2 dimensions")
        return int(abs(lhs[0] - rhs[0]) + abs(lhs[1] - rhs[1]))

    def _get_wire_len(self) -> int:
        """
        Calculate the total Manhattan length of all segments in the edge.
        """
        length = 0
        cur_point = self.segments[0]
        for segment_end in self.segments[1:]:
            # Segments are orthogonal (Manhattan distance)
            length += Edge._calc_len(cur_point, segment_end)
            cur_point = segment_end
        return length

    @staticmethod
    def _get_pos_by_offset(start_pos: list, end_pos: list, offset: int) -> list:
        """
        Get the coordinates at a specific offset from the start position.
        """
        length = Edge._calc_len(start_pos, end_pos)
        if length == 0:
            return start_pos.copy()
            
        # For orthogonal segments, either x or y will change, not both
        if start_pos[0] == end_pos[0]:
            # Vertical segment
            y_direction = 1 if end_pos[1] > start_pos[1] else -1
            return [start_pos[0], start_pos[1] + y_direction * offset]
        else:
            # Horizontal segment
            x_direction = 1 if end_pos[0] > start_pos[0] else -1
            return [start_pos[0] + x_direction * offset, start_pos[1]]

    @staticmethod
    def _is_same_pos(lhs: list, rhs: list) -> bool:
        """
        Check if two positions are the same.
        """
        return lhs[0] == rhs[0] and lhs[1] == rhs[1]

    def _update_positions(self) -> None:
        """
        Update the position-to-offset and offset-to-position mappings.
        Called whenever the edge geometry changes.
        """
        self._points_to_offsets.clear()
        self._offsets_to_points.clear()
        
        if len(self.segments) < 2:
            return
            
        self._len = self._get_wire_len()
        
        # Start from the last segment and work backwards
        segment_start_idx = len(self.segments) - 1
        segment_end_idx = len(self.segments) - 2
        cumulative_length = 0
        
        # Map each possible position on the wire to its offset from the end
        for offset in range(self._len + 1):
            start = self.segments[segment_start_idx]
            end = self.segments[segment_end_idx]
            
            # Get position at current offset from segment start
            segment_offset = offset - cumulative_length
            cur_pos = Edge._get_pos_by_offset(start, end, segment_offset)
            
            # Store mappings
            self._offsets_to_points[offset] = cur_pos
            self._points_to_offsets[str(cur_pos)] = offset
            
            # If we've reached the end of the current segment, move to previous segment
            if Edge._is_same_pos(cur_pos, end):
                segment_start_idx = segment_end_idx
                segment_end_idx -= 1
                cumulative_length = offset

    @staticmethod
    def _pos_is_on_segment(segment_start: list, segment_end: list, pos: list) -> bool:
        """
        Check if a position lies on a segment.
        """
        # Check if point is on a horizontal segment
        on_horizontal = (segment_start[1] == segment_end[1] == pos[1]) and \
                        (min(segment_start[0], segment_end[0]) <= pos[0] <= max(segment_start[0], segment_end[0]))
        
        # Check if point is on a vertical segment
        on_vertical = (segment_start[0] == segment_end[0] == pos[0]) and \
                      (min(segment_start[1], segment_end[1]) <= pos[1] <= max(segment_start[1], segment_end[1]))
        
        return on_horizontal or on_vertical

    @staticmethod
    def _get_nodes_position(id: int, nodes: dict) -> list:
        """
        Get the position of a node by ID.
        """
        if id not in nodes:
            raise ValueError(f"Invalid node ID: {id}")
        return nodes[id].get_position()

    def get_as_map(self) -> dict:
        """
        Returns edge data as a dictionary for JSON serialization.
        """
        return {
            "id": self.id,
            "vertices": [self.vertices["start"], self.vertices["end"]],
            "segments": self.segments
        }

    def add_segment_point(self, x: int, y: int) -> None:
        """
        Add a point to the edge geometry.
        """
        self.segments.append([x, y])
        if len(self.segments) >= 2:
            self._update_positions()
  
    def get_distance(self, lhs_pos: list, rhs_pos: list) -> int:
        """
        Get the distance along the wire between two positions.
        """
        lhs_offset = self._points_to_offsets[str(lhs_pos)]
        rhs_offset = self._points_to_offsets[str(rhs_pos)]
        return int(abs(lhs_offset - rhs_offset))

    def get_id(self) -> int:
        """Returns the edge ID."""
        return self.id
  
    def get_nodes(self) -> tuple[int, int]:
        """
        Returns the IDs of the start and end nodes.
        """
        return self.vertices['start'], self.vertices['end']
  
    def get_start_pos(self) -> list:
        """
        Returns the starting position of the edge.
        """
        return self.segments[0]
  
    def get_len(self) -> int:
        """
        Returns the total length of the edge.
        """
        return self._len
  
    def get_offset_by_position(self, pos: list) -> int:
        """
        Get the offset from the end point for a given position.
        """
        return self._points_to_offsets[str(pos)]
  
    def has_position(self, pos: list) -> bool:
        """
        Check if a position is on this edge.
        """
        return str(pos) in self._points_to_offsets
  
    def get_position_by_offset(self, offset_from_end: int) -> list:
        """
        Get the position at a specific offset from the end point.
        """
        return self._offsets_to_points[offset_from_end]

    def get_segment_by_position(self, pos: list) -> list[list]:
        """
        Get the segment that contains a given position.
        """
        for segment_beg_id in range(len(self.segments) - 1):
            segment = [self.segments[segment_beg_id], self.segments[segment_beg_id + 1]]
            if Edge._pos_is_on_segment(segment_start=segment[0], segment_end=segment[1], pos=pos):
                return segment
        return None
  
    def get_coordinates_before_segment(self, segment: list[list]) -> list[list]:
        """
        Get all coordinates from the start of the edge up to a given segment.
        """
        segments_before = []
        for segment_start in self.segments:
            segments_before.append(segment_start)
            if str(segment_start) == str(segment[0]):
                break
        return segments_before

    def get_coordinates_after_segment(self, segment: list[list]) -> list[list]:
        """
        Get all coordinates from a given segment to the end of the edge.
        """
        for segment_pos_id in range(len(self.segments)):
            if str(self.segments[segment_pos_id]) == str(segment[1]):
                return self.segments[segment_pos_id:].copy()
        return []

    def validate(self, nodes: dict) -> bool:
        """
        Validate the edge geometry against node positions.
        """
        # Check if edge start matches the start node's position
        if Edge._get_nodes_position(self.vertices["start"], nodes) != self.segments[0]:
            raise ValueError(f"Invalid first segment point for edge {self.id}")
        
        # Check if edge end matches the end node's position
        if Edge._get_nodes_position(self.vertices["end"], nodes) != self.segments[-1]:
            raise ValueError(f"Invalid last segment point for edge {self.id}")
        
        # Verify all segments are orthogonal (Manhattan distance)
        for i in range(len(self.segments) - 1):
            first_pos = self.segments[i]
            second_pos = self.segments[i + 1]
            if first_pos[0] != second_pos[0] and first_pos[1] != second_pos[1]:
                return False
        return True

class Graph:
    """
    Represents the routing tree as a graph of nodes and edges.
    """
    def __init__(self) -> None:
        """Initialize an empty graph."""
        self.nodes = {}
        self.edges = {}
        # Root (driver) node ID
        self.start_id = -1

    @staticmethod
    def _get_field(record: dict, key: str):
        """
        Extract a field from a record, with error checking.
        """
        if key not in record:
            raise ValueError(f"Invalid input format: `{key}` field is missing")
        return record[key]
 
    @staticmethod
    def _get_node_parameters(record: dict, node: Node) -> None:
        """
        Extract and set node parameters from a record.
        """
        if "capacitance" not in record or "rat" not in record:
            return
        node.add_parameters(
            capacitance=float(record["capacitance"]),
            rat=float(record["rat"])
        )

    def _parse_nodes(self, nodes_list: list) -> None:
        """
        Parse node data from JSON and create Node objects.
        """
        for node_record in nodes_list:
            id = Graph._get_field(node_record, "id")
            node = Node(
                id=id,
                x=Graph._get_field(node_record, "x"),
                y=Graph._get_field(node_record, "y"),
                type=Graph._get_field(node_record, "type"),
                name=Graph._get_field(node_record, "name")
            )
            Graph._get_node_parameters(node_record, node)
            self.nodes[id] = node

    def _parse_edges(self, edges_list: list) -> None:
        """
        Parse edge data from JSON and create Edge objects.
        """
        for edge_record in edges_list:
            vertices = Graph._get_field(edge_record, "vertices")
            id = Graph._get_field(edge_record, "id")
            edge = Edge(
                id=id,
                start_vert_id=vertices[0],
                end_vert_id=vertices[1]
            )
            segments = Graph._get_field(edge_record, "segments")
            for segment_point in segments:
                edge.add_segment_point(segment_point[0], segment_point[1])
            if not edge.validate(self.nodes):
                raise ValueError(f"Invalid segments in edge {id}")
            self.edges[id] = edge
  
    def _bind_nodes_with_edges(self) -> None:
        """
        Connect nodes with their edges.
        For each edge, add its ID to the connected nodes' edge lists.
        """
        for edge in self.edges.values():
            start_node_id, end_node_id = edge.get_nodes()
            self.nodes[start_node_id].add_edge(edge.get_id())
            self.nodes[end_node_id].add_edge(edge.get_id())

    def _find_start(self) -> None:
        """
        Find the driver (start) node in the graph.
        """
        for id, node in self.nodes.items():
            if node.is_buffer():
                if self.start_id != -1:
                    raise ValueError(f"Multiple drivers in input: {self.start_id} and {node.get_id()}")
                self.start_id = node.get_id()

    def _get_nodes_for_json(self) -> list:
        """
        Prepare node data for JSON serialization.
        """
        nodes_for_json = []
        for node in self.nodes.values():
            nodes_for_json.append(node.get_as_map())
        return nodes_for_json
  
    def _get_edges_for_json(self) -> list:
        """
        Prepare edge data for JSON serialization.
        """
        edges_for_json = []
        for edge in self.edges.values():
            edges_for_json.append(edge.get_as_map())
        return edges_for_json

    def _find_edge_id_by_pos(self, pos: list) -> int:
        """
        Find the edge ID that contains a given position.
        """
        for edge_id, edge in self.edges.items():
            if edge.has_position(pos):
                return edge_id
        return None

    def _split_edge_by_node(self, node_id: int, edge_id: int) -> None:
        """
        Split an edge into two edges at a node position.
        Used for buffer insertion.
        """
        edge = self.edges[edge_id]
        node = self.nodes[node_id]
        
        # Get the original edge's endpoints
        lhs_node_id, rhs_node_id = edge.get_nodes()
        
        # Create new edge IDs
        lhs_edge_id = len(self.edges)
        rhs_edge_id = edge.get_id()
        
        # Create new edges
        lhs_edge = Edge(
            id=lhs_edge_id, 
            start_vert_id=lhs_node_id, 
            end_vert_id=node_id
        )
        rhs_edge = Edge(
            id=rhs_edge_id, 
            start_vert_id=node_id, 
            end_vert_id=rhs_node_id
        )
        
        # Get node position and find the segment containing it
        node_pos = node.get_position()
        segment_with_node = edge.get_segment_by_position(node_pos)
        
        # Get coordinates for the left and right parts
        segment_points_for_lhs = edge.get_coordinates_before_segment(segment_with_node)
        segment_points_for_lhs.append(node_pos)

        segment_points_for_rhs = [node_pos] + edge.get_coordinates_after_segment(segment_with_node)

        # Add points to new edges
        for point in segment_points_for_lhs:
            lhs_edge.add_segment_point(point[0], point[1])
        for point in segment_points_for_rhs:
            rhs_edge.add_segment_point(point[0], point[1])

        # Replace the original edge with two new edges
        del self.edges[edge_id]
        self.edges[lhs_edge_id] = lhs_edge
        self.edges[rhs_edge_id] = rhs_edge

        # Validate new edges
        lhs_edge.validate(self.nodes)
        rhs_edge.validate(self.nodes)
        
        # Update node connections
        node.add_edge(lhs_edge_id)
        node.add_edge(rhs_edge_id)
        
        # Remove old edge from node connections and add new ones
        # First remove the old edge from the endpoints' lists
        if edge_id in self.nodes[lhs_node_id].edges:
            self.nodes[lhs_node_id].edges.remove(edge_id)
        if edge_id in self.nodes[rhs_node_id].edges:
            self.nodes[rhs_node_id].edges.remove(edge_id)
            
        # Then add the new edges to the endpoints
        self.nodes[lhs_node_id].add_edge(lhs_edge_id)
        self.nodes[rhs_node_id].add_edge(rhs_edge_id)

    def add_buffer(self, position: list) -> None:
        """
        Add a buffer node at the specified position.
        """
        # Create a new node ID
        new_node_id = len(self.nodes)
        
        # Create a new buffer node
        self.nodes[new_node_id] = Node(
            id=new_node_id, 
            x=position[0],
            y=position[1],
            type=Node.BUFFER_TYPE,
            name="buf"
        )
        
        # Find the edge that contains this position
        edge_id = self._find_edge_id_by_pos(pos=position)
        
        # Split the edge at the buffer position
        self._split_edge_by_node(new_node_id, edge_id)

    def get_start_node_id(self) -> int:
        return self.start_id
  
    def get_node(self, node_id: int) -> Node:
        return self.nodes[node_id]
  
    def get_edge_between_nodes(self, lhs_id: int, rhs_id: int) -> Edge:
        lhs_node = self.get_node(lhs_id)
        edge_ids = lhs_node.get_edges()
        
        for edge_id in edge_ids:
            edge = self.get_edge(edge_id)
            start_id, end_id = edge.get_nodes()
            if rhs_id == start_id or rhs_id == end_id:
                return edge
                
        return None

    def get_edge(self, edge_id: int) -> Edge:
        return self.edges[edge_id]
  
    def get_node_neighbors(self, node_id: int) -> list[int]:
        """
        Get all neighboring nodes of a given node.
        """
        neighbors = set()
        node = self.get_node(node_id)
        
        for edge_id in node.get_edges():
            edge = self.get_edge(edge_id)
            start_node_id, end_node_id = edge.get_nodes()
            neighbors.add(start_node_id)
            neighbors.add(end_node_id)
            
        # Remove self from neighbors
        neighbors.discard(node_id)
        return list(neighbors)

    def load_from_json(self, json_file: str) -> None:
        """
        Load graph data from a JSON file.
        """
        with open(json_file, 'r') as file:
            data = json.load(file)
            
        self._parse_nodes(Graph._get_field(data, "node"))
        self._parse_edges(Graph._get_field(data, "edge"))
        self._bind_nodes_with_edges()
        self._find_start()

    def store_to_json(self, json_file: str) -> None:
        """
        Store graph data to a JSON file.
        """
        json_representation = {
            "node": self._get_nodes_for_json(),
            "edge": self._get_edges_for_json()
        }
        
        with open(json_file, "w") as file:
            json.dump(json_representation, file, indent=4)

    def dump_to_dot(self, file_name: str) -> None:
        """
        Export the graph to a DOT file for visualization.
        """
        with open(file_name, 'w') as file:
            file.write("digraph nodes {\n")
            
            # Write node definitions
            for node in self.nodes.values():
                file.write(f'  "node_{node.id}"  [shape = {node.get_shape()} label = " {node.id}[{node.x}, {node.y}] "]\n')
            
            # Write edge definitions
            for edge in self.edges.values():
                start_node_id, end_node_id = edge.get_nodes()
                file.write(f'  "node_{start_node_id}" -> "node_{end_node_id}";\n')
                
            file.write("}\n")

    @staticmethod
    def create_wire(length: int):
        """
        Create a simple two-node wire graph for benchmarking.
        """
        graph = Graph()
        
        # Create start (driver) node
        start_node = Node(
            id=0,
            x=0,
            y=0,
            type=Node.BUFFER_TYPE,
            name='Start'
        )
        
        # Create end (sink) node
        end_node = Node(
            id=1,
            x=length,
            y=0,
            type=Node.TERMINAL_TYPE,
            name='End'
        )
        end_node.add_parameters(capacitance=1.5, rat=100)
        
        # Add nodes to graph
        graph.nodes[start_node.id] = start_node
        graph.nodes[end_node.id] = end_node
        graph.start_id = start_node.id

        # Create edge connecting start and end nodes
        edge = Edge(
            id=0,
            start_vert_id=start_node.id,
            end_vert_id=end_node.id
        )
        edge.add_segment_point(start_node.x, start_node.y)
        edge.add_segment_point(end_node.x, end_node.y)
        graph.edges[edge.id] = edge
        
        # Validate edge and connect to nodes
        edge.validate(graph.nodes)
        graph.nodes[start_node.id].add_edge(edge.id)
        graph.nodes[end_node.id].add_edge(edge.id)
        
        return graph