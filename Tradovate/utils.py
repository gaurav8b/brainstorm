import math
import os
from datetime import date, datetime,timezone,timedelta
tz_utc = timezone.utc

def round_down(number:float, decimals:int=2):
    """
    Returns a value rounded down to a specific number of decimal places.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more")
    elif decimals == 0:
        return math.floor(number)

    factor = 10 ** decimals
    return math.floor(number * factor) / factor

def round_to_min_size(size,min_size):
    dec = int(math.log10(1/min_size))
    return round_down(size,dec)

def create_dir(dirR):
    if not os.path.exists(os.path.dirname(dirR)):
                os.makedirs(os.path.dirname(dirR))
def timetz(*args):
    return datetime.now(tz_utc).timetuple()


# x  = round_to_min_size(10.123956,0.001)
# print(x)