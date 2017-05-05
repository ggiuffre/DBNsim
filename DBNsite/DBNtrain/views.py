from django.http import HttpResponse
from django.shortcuts import render

import numpy as np
from DBNlogic.nets import DBN, RBM
from DBNlogic.sets import *
from DBNlogic.train import CDTrainer

def index(request):
    context = {
        # ...
    }
    return render(request, 'DBNtrain/index.html', context)

def train(request):
    L1  = RBM(14, 5)
    net = DBN([L1], name = 'mnist')

    trainset = DataSet.fromPickle('data/left_8.pkl')
    # trainset = DataSet.fromPickle('data/faces_all.pkl')
    # trainset = SmallerMNIST().data

    net.learn(trainset, max_epochs = 1, batch_sz = 2)
    result = net.evaluate(trainset[2]).reshape(8, 1)
    return HttpResponse(str(result))
