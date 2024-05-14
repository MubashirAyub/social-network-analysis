from django.urls import path
from RS_Lab import views
from RS_Lab.dash_apps import SNA_Home,upload_test

urlpatterns = [
    path('', views.under_dev_view.as_view(), name="UnderConst"),
    path('Contact/', views.Contact.as_view(), name="Contact"),
    path('Project/', views.Project_About.as_view(), name="Project"),
    path('Documentation/', views.Documentation.as_view(), name="Documentation"),
    path('upload/', views.upload_test.as_view(), name="upload"),

]
