import json

class Model:
    """
    Represents timing and electrical models for buffer and wire delays.
    Used to calculate delays and capacitances for the Van Ginneken algorithm.
    """
    def __init__(self) -> None:
        """Initialize an empty timing model."""
        # Buffer parameters
        self.intrinsic_delay = 0.0  # Buffer intrinsic delay
        self.buffer_resistance = 0.0  # Buffer output resistance
        self.buffer_capacitance = 0.0  # Buffer input capacitance
        
        # Wire parameters
        self.unit_resistance = 0.0  # Resistance per unit length
        self.unit_capacitance = 0.0  # Capacitance per unit length

    @staticmethod
    def _get_field(record: dict, key: str):
        """
        Extract a field from a record, with error checking.
        """
        if key not in record:
            raise ValueError(f"Invalid input format: `{key}` field is missing")
        return record[key]
  
    def get_buffer_capacitance(self) -> float:
        """
        Get the buffer input capacitance.
        """
        return self.buffer_capacitance
  
    def wire_capacitance(self, L: int) -> float:
        """
        Calculate the capacitance of a wire segment.
        """
        return self.unit_capacitance * L

    def buffer_delay(self, capacitance: float) -> float:
        """
        Calculate the delay of a buffer driving a specific load capacitance.
        """
        return self.intrinsic_delay + self.buffer_resistance * capacitance
  
    def wire_delay(self, L: float, C: float) -> float:
        """
        Calculate the delay of a wire segment.
        Uses the Elmore delay model with distributed RC network.
        """
        # First term accounts for distributed RC of the wire itself
        # Second term accounts for the delay driving the load
        return (self.unit_resistance * self.unit_capacitance * L * L) / 2 + \
               (self.unit_resistance * L * C)
  
    def load_from_json(self, json_file: str) -> None:
        """
        Load timing model parameters from a JSON file.
        """
        with open(json_file, 'r') as file:
            data = json.load(file)
            
        # Extract buffer parameters
        module = Model._get_field(data, "module")
        input_data = Model._get_field(module[0], "input")
        self.intrinsic_delay = Model._get_field(input_data[0], "intrinsic_delay")
        self.buffer_resistance = Model._get_field(input_data[0], "R")
        self.buffer_capacitance = Model._get_field(input_data[0], "C")
        
        # Extract technology parameters
        technology = Model._get_field(data, "technology")
        self.unit_resistance = Model._get_field(technology, "unit_wire_resistance")
        self.unit_capacitance = Model._get_field(technology, "unit_wire_capacitance")
        
    def print_parameters(self) -> None:
        """Print the timing model parameters."""
        print("Timing Model Parameters:")
        print(f"  Buffer:")
        print(f"    Intrinsic Delay: {self.intrinsic_delay}")
        print(f"    Resistance: {self.buffer_resistance}")
        print(f"    Capacitance: {self.buffer_capacitance}")
        print(f"  Wire (per unit length):")
        print(f"    Resistance: {self.unit_resistance}")
        print(f"    Capacitance: {self.unit_capacitance}")