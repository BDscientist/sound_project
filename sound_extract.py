import argparse
import array
import math
import numpy as np
import random
import wave
import sys
import io
import pickle


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--C:\cleanfile', type=str, required=True)
    parser.add_argument('--C:\noisefile', type=str, required=True)
    parser.add_argument('--C:\ouput_clean', type=str, default='')
    parser.add_argument('--C:\output_noise', type=str, default='')
    parser.add_argument('--C:\output_noise', type=str, default='', required=True)
    parser.add_argument('--C:\sn', type=float, default='', required=True)
    args = parser.parse_args()
    return args



def cal_amp(wf):
    buffer = wf.readframes(wf.getnframes())
    amptitude = (np.frombuffer(buffer, dtype="int16")).astype(np.float64)
    return amptitude
