from django.http import HttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

import json
import string
import random
from time import time
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

    try:
        num_layers = int(request.POST['num_layers'])
    except ValueError:
        num_layers = 1
        # return HttpResponse({'error': 'you haven\'t specified [...]'})

    net = DBN(name = trainset_name)
    vis_size = int(request.POST['vis_sz'])
    for layer in range(1, num_layers):
        hid_size = int(request.POST['hid_sz_' + str(layer)])
        print('creating a', vis_size, 'x', hid_size, 'RBM...')
        net.append(RBM(vis_size, hid_size))
        vis_size = hid_size # for the next RBM

    config = {
        'max_epochs' : int(request.POST['epochs']),
        'batch_size' : int(request.POST['batch_size']),
        'learn_rate' : float(request.POST['learn_rate']),
        'momentum'   : float(request.POST['momentum'])
    }

    random_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k = 10))
    print(random_id)
    training_jobs[random_id] = {
        'birthday': time(),
        'generator': net.learn(trainset, Configuration(**config))
    }

    # delete a random job older than one hour:
    random_old_job = random.choice(list(training_jobs.keys()))
    if time() - training_jobs[random_old_job]['birthday'] > 60 * 60:
        print('deleting old job n.', random_old_job)
        del training_jobs[random_old_job]

    return HttpResponse(random_id)

@csrf_exempt
def getError(request):
    """Run one training iteration for a particular network
    and return the reconstruction error."""
    body = json.loads(request.body.decode())
    job = body['job_id']
    train_gen = training_jobs[job]['generator']
    stop = False

    try:
        next_err = next(train_gen)
    except StopIteration:
        next_err = None
        stop = True

    response = {
        'error': next_err,
        'stop': stop
    }
    json_response = json.dumps(response)
    return HttpResponse(json_response, content_type = 'application/json')
