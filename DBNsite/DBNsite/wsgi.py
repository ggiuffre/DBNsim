"""
WSGI config for DBNsite project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/1.11/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "DBNsite.settings")

# Originally was...
# application = get_wsgi_application()

# ... then added this for easy Heroku support:
from dj_static import Cling
application = Cling(get_wsgi_application())
