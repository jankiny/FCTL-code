import json
import math
from pathlib import Path
from datetime import datetime
import os
import sys
import logging
import argparse
import json

import torch
import matplotlib
from sklearn import metrics
import matplotlib.pyplot as plt
import importlib
import pandas as pd
import numpy as np

def to_datetime(seconds):
    weeks = math.floor(seconds / (60 * 60 * 24 * 7))
    days = math.floor((seconds % (60 * 60 * 24 * 7)) / (60 * 60 * 24))
    hours = math.floor((seconds % (60 * 60 * 24)) / (60 * 60))
    minutes = math.floor((seconds % (60 * 60)) / 60)
    seconds = math.floor(seconds % 60)

    string = ""
    if weeks > 0:
        string += str(weeks) + "w "
    if days > 0:
        string += str(days) + "d "
    if hours > 0:
        string += str(hours) + "h "
    if minutes > 0:
        string += str(minutes) + "m "
    if seconds > 0:
        string += str(seconds) + "s"

    return string

def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {
        'black': '\033[30m',  # basic colors
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'bright_black': '\033[90m',  # bright colors
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
        'end': '\033[0m',  # misc
        'bold': '\033[1m',
        'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']