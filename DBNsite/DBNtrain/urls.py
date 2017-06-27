from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^$', views.index, name='index'),
    url(r'^train/$', views.train),
    url(r'^getError/$', views.getError),
    url(r'^getInput/$', views.getInput),
    url(r'^getReceptiveField/$', views.getReceptiveField),
    url(r'^getHistogram/$', views.getHistogram),
    url(r'^saveNet/$', views.saveNet),
    url(r'^getArchFromNet/$', views.getArchFromNet),
]
