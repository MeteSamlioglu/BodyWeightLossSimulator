import math

def custom_round(value):
    if value - int(value) >= 0.5:
        return math.ceil(value)
    else:
        return math.floor(value)