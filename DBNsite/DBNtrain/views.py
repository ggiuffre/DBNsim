from django.http import HttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

import json
import string
import random
from DBNlogic.nets import DBN, RBM
from DBNlogic.sets import DataSet
from DBNlogic.util import Configuration



training_jobs = {}



def index(request):
    """Return the main page."""
    ordered_datasets = sorted(DataSet.allSets(), key = str.lower)
    context = {
        'config': Configuration(),   # default configuration
        'datasets': ordered_datasets # available datasets
    }
    return render(request, 'DBNtrain/index.html', context)

@csrf_exempt
def train(request):
    """Set up a network to be trained according to the
    parameters in the request."""
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

    random_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k = 16))
    training_jobs[random_id] = {
        'net': net,
        'config': config,
        'dataset': trainset
    }

    return HttpResponse(random_id)

@csrf_exempt
def getError(request):
    """Run one training iteration for a particular network
    and return the reconstruction error."""
    body = json.loads(request.body.decode())
    job = body['job_id']
    stop = False

    net = training_jobs[job]['net']
    config = training_jobs[job]['config']
    dataset = training_jobs[job]['dataset']

    try:
        next_err = next(net.learn(dataset, Configuration(**config)))
    except StopIteration:
        stop = True

    response = {
        'error': next_err,
        'stop': stop
    }
    json_response = json.dumps(response)
    return HttpResponse(json_response, content_type = 'application/json')
