#!/usr/bin/env python3

import os
import pandas
import echonet
import sys
import sklearn
import math
import matplotlib.pyplot as plt
import collections
import numpy as np

if __name__ == "__main__":
    x = pandas.read_csv(sys.argv[1])
    filename = x["Filename"]
    phase = x["Phase"]
    inter = x["Intersection"]
    union = x["Union"]
    inter / (inter + union)
    dice = (2 * inter / (inter + union)).mean()

    for (phase, mask) in [
        ("overall", phase == phase),
        ("diastole", phase == "d"),
        ("systole", phase == "s")
    ]:
        print("Dice ({}):   {:.3f} ({:.3f} - {:.3f})".format(phase, *echonet.utils.bootstrap(inter[mask], union[mask], lambda i, u: (2 * i / (i + u + 1e-10)).mean())))
