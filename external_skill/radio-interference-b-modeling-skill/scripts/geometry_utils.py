"""Minimal geometry verification helpers for the radio-interference skill."""
from math import atan2, pi, hypot

def wrap_angle(x):
    return (x + pi) % (2*pi) - pi

def shortest_angle(a, b):
    return wrap_angle(a-b)

def polygon_diameter_bruteforce(points):
    best = (0.0, None)
    for i, a in enumerate(points):
        for j in range(i+1, len(points)):
            b = points[j]
            d = hypot(a[0]-b[0], a[1]-b[1])
            if d > best[0]: best = (d, (i,j))
    return best

def bearing(src, dst):
    return atan2(dst[1]-src[1], dst[0]-src[0])
