from unittest import result


import math

def speed(x):
    return 2000 / x

def speed_to_2000m_sec(speed):
    return 2000 / speed

def sec_to_2000m_time(sec):
    m = math.floor(sec / 60)
    s = round((sec % 60), 1)
    if s < 10:
        return str(m) + ':0' + str(s)
    return str(m) + ':' + str(s)

def sec_to_500m_lap_time(sec):
    sec = sec / 4
    m = math.floor(sec / 60)
    s = round((sec % 60), 1)
    if s < 10:
        return str(m) + ':0' + str(s)
    return str(m) + ':' + str(s)