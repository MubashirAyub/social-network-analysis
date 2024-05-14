from django.views.generic import (TemplateView)
from django.shortcuts import render


# Create your views here.

class under_dev_view(TemplateView):
    template_name = "Under_Development.html"


class Contact(TemplateView):
    template_name = "Contact.html"

class Documentation(TemplateView):
    template_name = "Documentation.html"

class Project_About(TemplateView):
    template_name = "Project_About.html"

class upload_test(TemplateView):
    template_name = "upload_Test.html"
