import os
import io
import sys
import sklearn
import numpy
import random

sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding = 'utf-8')

n_labels =11
batch_size=32
sequence_length =251
feature_dimension = 513

def prepare_data():

    global train_samples, train_labels, valid_samples, valid_labels, test_samples, test_labels, data_mean, data_std


    train_samples  = open('C:/project1/train_samples.txt').read().strip().split('\n')
    train_labels = [int(label) for label in open('C:/project1/train_labels.txt').read().strip().split('\n')]

    valid_samples = open('C:/project1/valid_samples.txt').read().strip().split('\n')
    valid_labels = [int(label)for label in open('C:/project1/valid_labels.txt').read().strip().split('\n')]

    test_samples = open('C:/project1/test_samples.txt').read().strip().split('\n')
    test_labels = [int(labels)for labels in open('C:/project1/test_labels.txt').read().strip().split('\n')]

    data_mean = numpy.load('C:/project1/train/spectrum/data_mean.npy')
    data_std = numpy.load('C:/project1/train/spectrum/data_std.npy')

def get_random_sample(part):

    global train_samples, train_labels, valid_samples, valid_labels, test_samples, test_labels, data_mean, data_std

    if part =='train':
        samples = train_samples
        labels = train_labels
    elif part == 'valid':
        samples = valid_samples
        labels = valid_labels
    elif part =='test':
        smaples = test_samples
        labels = test_labels
    else :
        print('please use train, valid, or test for the part name')

    i = random.randrange(len(samples))
    spectrum = numpy.load('C:/project1/'+part+'/spectrum/'+samples[i]+'.npy')
    spectrum  = (spectrum -data_mean) / (data_std+ 0.0001)
    return spectrum, labels[i]


def get_random_batch(part):
    X = numpy.zeros((batch_size, sequence_length, feature_dimension,1))
    Y = numpy.zeros((batch_size,))
    for b in range(batch_size):
        s, l =get_random_sample(part)    #spectrum , labels[i]
        X[b, :, :, 0] = s[:sequence_length, :feature_dimension]
        Y[b] =l
    return X,Y
