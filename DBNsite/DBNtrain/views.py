from django.http import HttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

import numpy as np
import json
import string
import random
from DBNlogic.nets import DBN, RBM
from DBNlogic.sets import *
from DBNlogic.util import Configuration
from DBNlogic.train import CDTrainer



training_jobs = {}



def index(request):
    context = {
        'config': Configuration(),
        'datasets': DataSet.allSets()
    }
    return render(request, 'DBNtrain/index.html', context)

def train(request):
    trainset_name = request.POST['dataset']
    trainset = DataSet.fromWhatever(trainset_name)

    vis_size = len(trainset[0])
    try:
        hid_size = int(request.POST['hid_sz'])
    except ValueError:
        print('Bad argument for hidden size: defaulting to 1.')
        hid_size = 1

    L1  = RBM(vis_size, hid_size)
    net = DBN([L1], name = trainset_name)

    config = { 
        'max_epochs' : int(request.POST['epochs']),
        'batch_size' : int(request.POST['batch_size']),
        'learn_rate' : float(request.POST['learn_rate']),
        'momentum'   : float(request.POST['momentum'])
    }

    # train_generator = net.learn(trainset, Configuration(**config))
    random_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k = 16))
    training_jobs[random_id] = {
        'trainer': CDTrainer(L1, config = Configuration(**config)),
        'dataset': trainset,
        'errors': []
    }

    return HttpResponse(random_id)

@csrf_exempt
def getError(request):
    body = json.loads(request.body.decode())
    job = body['job_id']

    trainer = training_jobs[job]['trainer']
    dataset = training_jobs[job]['dataset']
    stop = False

    try:
        next_err = next(trainer.run(dataset))
        training_jobs[job]['errors'].append(next_err)
    except StopIteration:
        stop = True

    response = {
        'error': training_jobs[job]['errors'][-1],
        'stop': stop
    }
    json_response = json.dumps(response)
    return HttpResponse(json_response, content_type = 'application/json')
