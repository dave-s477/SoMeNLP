import datetime

def get_time_marker(format='full'):
    """Get current time as a formatted string.

    Args:
        format (string): full (with time) vs. short (only date)

    Returns:
        string: current time
    """
    if format=='full':
        out_s = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    else:
        out_s = datetime.datetime.now().strftime("%d-%m-%Y")
    return out_s