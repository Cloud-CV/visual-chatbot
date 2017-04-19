from django.conf.urls import url
from chat import views

urlpatterns = [
    url(r'^$', views.home, name='home'),
    url(r'^upload/', views.upload_image, name='upload'),
]
