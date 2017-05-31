from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^$', views.index, name='index'),
    url(r'^train/$', views.train, name='train'),
    url(r'^getError/$', views.getError, name='getError'),
    url(r'^getInput/$', views.getInput, name='getInput'),
]
