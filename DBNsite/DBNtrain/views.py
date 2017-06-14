from django.http import HttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

import json
import string
import random
from time import time
from DBNlogic.nets import DBN, RBM
from DBNlogic.sets import DataSet
from DBNlogic.util import Configuration, heatmap



# pending training jobs on the server:
training_jobs = {}

# available datasets on the server:
ordered_datasets = sorted(DataSet.allSets(), key = str.lower)
datasets_info = {d: DataSet.fromWhatever(d).shape[1] for d in ordered_datasets}



def index(request):
    """Return the main page."""
    context = {
        'config': Configuration(), # default configuration
        'datasets': datasets_info  # available datasets
    }
    return render(request, 'DBNtrain/index.html', context)

@csrf_exempt
def train(request):
    """Set up a network to be trained according to
    the parameters in the HTTP request."""
    trainset_name = request.POST['dataset']
    trainset = DataSet.fromWhatever(trainset_name)

    try:
        num_layers = 1 + int(request.POST['num_hid_layers'])
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

    # delete the old client job that is being replaced (if any):
    last_job = request.POST['last_job_id']
    if last_job in training_jobs:
        del training_jobs[last_job]

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
        next_err = round(train_info['err'], 3)
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

def getInput(request):
    """Return a specific input image of a specific dataset."""
    dataset_name = request.GET['dataset']
    dataset = DataSet.fromWhatever(dataset_name)
    index = int(request.GET['index'])
    if index < 0:
        index = random.randint(0, len(dataset) - 1)

    image = dataset[index].tolist()
    response = heatmap(image)

    json_response = json.dumps(response)
    return HttpResponse(json_response, content_type = 'application/json')

def getReceptiveField(request):
    """Return the receptive field of a specific
    neuron in a specific layer of a DBN."""
    if 'job_id' not in request.GET:
        return HttpResponse('', content_type = 'application/json')

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
    if 'job_id' not in request.GET:
        return HttpResponse('', content_type = 'application/json')

    job = request.GET['job_id']
    net = training_jobs[job]['network']
    rbm = int(request.GET['rbm'])

    response = net[rbm].weightsHistogram()
    json_response = json.dumps(response)
    return HttpResponse(json_response, content_type = 'application/json')
