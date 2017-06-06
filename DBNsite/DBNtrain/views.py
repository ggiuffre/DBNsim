from django.http import HttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

import json
import string
import random
import math
from time import time
from DBNlogic.nets import DBN, RBM
from DBNlogic.sets import DataSet
from DBNlogic.util import Configuration



# pending training jobs on the server
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
    """Set up a network to be trained according to
    the parameters in the HTTP request."""
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
        vis_size = hid_size # for constructing the next RBM

    config = {
        'max_epochs' : int(request.POST['epochs']),
        'batch_size' : int(request.POST['batch_size']),
        'learn_rate' : float(request.POST['learn_rate']),
        'momentum'   : float(request.POST['momentum'])
    }

    random_id = ''.join(random.choice(string.ascii_uppercase + string.digits) for i in range(10))
    training_jobs[random_id] = {
        'birthday': time(),
        'network': net,
        'generator': net.learn(trainset, Configuration(**config))
    }

    # delete a random pending job older than one hour:
    random_old_job = random.choice(list(training_jobs.keys()))
    if time() - training_jobs[random_old_job]['birthday'] > 3600:
        print('deleting old job n.', random_old_job)
        del training_jobs[random_old_job] # risky...

    return HttpResponse(random_id)

@csrf_exempt
def getError(request):
    """Run one training iteration for a particular
    network and return the reconstruction error."""
    body = json.loads(request.body.decode())
    job = body['job_id']
    train_gen = training_jobs[job]['generator']

    curr_rbm = None
    next_err = None
    stop     = False

    try:
        train_info = next(train_gen)
        curr_rbm = train_info['rbm']
        next_err = train_info['err']
    except StopIteration:
        # training_jobs[job]['network'].save()
        del training_jobs[job]['generator']
        stop = True

    response = {
        'curr_rbm': curr_rbm,
        'error': next_err,
        'stop': stop
    }
    json_response = json.dumps(response)
    return HttpResponse(json_response, content_type = 'application/json')

def heatmap(array):
    """Return a Highcharts-formatted heatmap from
    a Python array."""
    dim = int(math.sqrt(len(array)))
    for row in range(dim):
        for col in range(dim):
            array[row * dim + col] = [col, dim - 1 - row, array[row * dim + col]] # from Python array to X,Y coordinates...
    return array

def getInput(request):
    """Return a specific input image of a specific dataset."""
    dataset = request.GET['dataset']
    index = int(request.GET['index'])

    image = DataSet.fromWhatever(dataset)[index].tolist()
    response = heatmap(image)

    json_response = json.dumps(response)
    return HttpResponse(json_response, content_type = 'application/json')

def getReceptiveField(request):
    """Return the receptive field of a specific
    neuron in a specific layer of a DBN."""
    job = request.GET['job_id']
    net = training_jobs[job]['network']
    layer = int(request.GET['layer'])
    neuron = int(request.GET['neuron'])

    rec_field = net.receptiveField(layer, neuron).tolist()
    response = heatmap(rec_field)

    json_response = json.dumps(response)
    return HttpResponse(json_response, content_type = 'application/json')

def getHistogram(request):
    """Return a histogram of the distribution of the weights
    of a specific RBM inside a specific DBN."""
    job = request.GET['job_id']
    net = training_jobs[job]['network']
    rbm = int(request.GET['rbm'])

    response = net.weightsHistogram(rbm)
    json_response = json.dumps(response)
    return HttpResponse(json_response, content_type = 'application/json')
