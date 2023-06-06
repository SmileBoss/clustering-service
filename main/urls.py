from django.urls import path, include

from . import views, swagger
from .views import TextClusteringView, TextParsingView

urlpatterns = [
    path('', include(swagger)),
    path('parsing/', TextParsingView.as_view(), name='parsing'),
    path('clustering/', TextClusteringView.as_view(), name='clustering'),
]
