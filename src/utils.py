class MethodNames:
    """Stores the names of each quadrature method from
    quadrature.py

    Attributes
    ----------
    Each quadrature method, represented as an integer
    for easy indexing
    """
    LEFT_POINT_RIEMMAN = 0
    RIGHT_POINT_RIEMMAN = 1
    MID_POINT_RIEMMAN = 2
    TRAPEZOIDAL = 3
    SIMPSONS = 4
    GAUSSIAN = 5
    MONTE_CARLO = 6


def parse_function(func_str):
    """Parses a string to turn it into a valid
    python expression

    Parameters
    ----------
    func_str: the string to parse

    Returns
    -------
    func_str transformed into a valid python expression
    """
    # Remove all spaces for easier parsing
    # (possibly not necessary, but less characters to
    # parse should be faster right?)
    func_str = func_str.replace(' ', '')

    # Replace ^ power operators with python-compliant
    # ** operators
    func_str = func_str.replace('^', '**')

    return func_str
