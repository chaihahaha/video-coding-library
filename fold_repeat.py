import re
import sympy as sp
import random
from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.sympy_parser import T
import warnings
from sympy.utilities.exceptions import SymPyDeprecationWarning

warnings.filterwarnings(
    # replace "ignore" with "error" to make the warning raise an exception.
    # This useful if you want to test you aren't using deprecated code.
    "ignore",

    # message may be omitted to filter all SymPyDeprecationWarnings
    message=r"""(?s).*Using non-Expr arguments in Mul is deprecated \(in this case, one of\nthe arguments has type 'Tuple'\).""",

    category=SymPyDeprecationWarning,
    module=r"."
)
def simplify_numeric_constants(expr):
    if isinstance(expr, sp.Tuple)  or isinstance(expr, tuple):
        # If it's a tuple, apply simplification recursively to each element
        simplified_tuple = tuple(simplify_numeric_constants(el) for el in expr)
        return simplified_tuple
    elif isinstance(expr, sp.Mul):
        # Separate numeric and symbolic parts in a multiplication
        numeric_part = 1
        non_numeric_parts = []
        
        for arg in expr.args:
            if arg.is_number:
                numeric_part *= arg  # Multiply numeric constants
            else:
                non_numeric_parts.append(simplify_numeric_constants(arg))  # Recursively simplify non-numeric parts

        # Return simplified multiplication without evaluation (to preserve structure)
        return sp.Mul(numeric_part, *non_numeric_parts, evaluate=False)
    else:
        # Return the expression as-is if it's neither a tuple nor a multiplication
        return expr

def find_longest_repeat(s):
    n = len(s)
    for length in range(n // 2, 0, -1):
        pattern = s[:length]
        # Create a regex pattern to find multiple occurrences of the substring
        occurrences = re.finditer(f'^({re.escape(pattern)})+', s)
        for match in occurrences:
            # Get the full matched substring and calculate repetitions
            repeated_str = match.group(0)
            repeat_count = len(repeated_str) // length
            if repeat_count > 1:
                return pattern, repeat_count, match.end()
    return s[0], 1, 1  # If no repeating pattern is found

def recursive_fold_repeat_binary_string(s):
    result = []
    i = 0
    while i < len(s):
        # Find the longest repeating pattern from the current position
        substring, repeat_count, end_idx = find_longest_repeat(s[i:])
        
        # Recursively compress the matched part for smaller patterns
        if repeat_count > 1:
            compressed_substring = recursive_fold_repeat_binary_string(substring)
            result.append(f"(({compressed_substring})*{repeat_count})")
        else:
            result.append(f"'{substring}'")
        
        i += end_idx  # Move to the next part of the string
    
    # Join the result into a single expression and return
    return '+'.join(result)

class FoldRepeatCoder:
    def __init__(self, zero, one):
        self.zero = zero
        self.one = one
        self.symbols = zero + one + '()+*0123456789'
    def encode(self, binary):
        binary_string = ''.join([str(i) for i in binary])
        compressed = recursive_fold_repeat_binary_string(binary_string)
        compressed = compressed.replace("'0'",self.zero).replace("'1'",self.one).replace("+",",")
        
        # Use SymPy to simplify the result
        non_comm_expr = parse_expr(compressed, evaluate=False, transformations=T[:5]+T[6:])
        simplified_expr = simplify_numeric_constants(non_comm_expr)
        simplified = str(simplified_expr)
    
        simplified = simplified.replace(" ","").replace(",","+")
        return simplified
    
    def decode(self, compressed):
        compressed = compressed.replace(self.zero,"'0'").replace(self.one,"'1'")
        unfold = eval(compressed)
        binary = [int(i) for i in unfold]
        return binary

def generate_random_binary(min_length, max_length):
    # Generate a random length for the arr, starting from min_length + 1
    length = random.randint(min_length, max_length)  # for example, between 101 and 150
    # Generate a random binary arr of the specified length
    binary = [random.choice([0, 1]) for _ in range(length)]
    return binary 

if __name__=='__main__':
    # Test case
    coder = FoldRepeatCoder('o', 'i')
    for i in range(100):
        binary = generate_random_binary(10, 100)
        compressed = coder.encode(binary)
        print(compressed)
        print(len(compressed), len(binary))
        decompressed = coder.decode(compressed)
        if binary != decompressed:
            print(binary)
            print(decompressed)
            print(compressed)
        assert binary == decompressed
