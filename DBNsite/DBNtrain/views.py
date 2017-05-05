from django.http import HttpResponse
#from django.template import loader
from django.shortcuts import render

def index(request):
	# return HttpResponse('ok')
    return render(request, 'DBNtrain/index.html', {'range': range(10)})
