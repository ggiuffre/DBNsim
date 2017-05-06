from django.http import HttpResponse
from django.shortcuts import render

import numpy as np
from DBNlogic.nets import DBN, RBM
from DBNlogic.sets import *
from DBNlogic.util import Configuration
from DBNlogic.train import CDTrainer

def index(request):
    context = {
        'config': Configuration(),
    }
    return render(request, 'DBNtrain/index.html', context)

def train(request):
    L1  = RBM(8, 15)
    net = DBN([L1], name = 'left')

    trainset = DataSet.fromPickle('DBNlogic/data/left_8.pkl')
    # trainset = DataSet.fromPickle('DBNlogic/data/faces_all.pkl')
    # trainset = SmallerMNIST().data

    max_epochs = int(request.POST['epochs'])
    batch_size = int(request.POST['batch_size'])
    learn_rate = float(request.POST['learn_rate'])
    momentum   = float(request.POST['momentum'])

    net.learn(trainset, Configuration())
    result = net.evaluate(trainset[2]).reshape(8, 1)
    return HttpResponse(str(result))
