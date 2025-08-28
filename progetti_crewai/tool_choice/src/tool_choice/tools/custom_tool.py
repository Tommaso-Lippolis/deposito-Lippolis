from crewai.tools import tool

@tool("Mathematical Sum of Two Numbers")
def add_numbers(a: float, b: float) -> float:
    """
    This tool receives two numbers as strings (extracted from the user's input by the agent),
    converts them to float, computes their sum, and returns the result as a string.
    The agent is responsible for extracting the two numbers from the user's request and passing them as arguments.
    Example: if the user says 'Add 5.2 and 7', the agent should call add_numbers('5.2', '7').
    """
    try:
        num1 = float(a)
        num2 = float(b)
        result = num1 + num2
        return result
    except Exception as e:
        return f"Error: could not compute the sum. Details: {e}"
