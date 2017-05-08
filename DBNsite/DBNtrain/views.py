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
        'datasets': DataSet.allSets()
    }
    return render(request, 'DBNtrain/index.html', context)

def train(request):
    dataset_name = request.POST['dataset']
    trainset = DataSet.fromWhatever(dataset_name)

    vis_size = len(trainset[0])
    hid_size = int(request.POST['hid_sz'])
    L1  = RBM(vis_size, hid_size)
    net = DBN([L1], name = dataset_name)

    config = { 
        'max_epochs' : int(request.POST['epochs']),
        'batch_size' : int(request.POST['batch_size']),
        'learn_rate' : float(request.POST['learn_rate']),
        'momentum'   : float(request.POST['momentum']),
    }
    net.learn(trainset, Configuration(**config))

    result = net.evaluate(trainset[2]).reshape(8, 1)
    return HttpResponse(str(result))
