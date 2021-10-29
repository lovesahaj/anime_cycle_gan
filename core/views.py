import pathlib
import uuid
from django.shortcuts import render
from .form import ImageUpload
import os


def homepage(request):
    template = 'base.html'

    context = {}
    return render(request, template_name=template, context=context)


def image_upload_handle(instance, filename):
    file_path = pathlib.Path(filename)
    new_file_path = str(uuid.uuid1())
    return f"{instance}/{new_file_path}{file_path.suffix}"


def save_file(f, instance):
    filename = image_upload_handle(instance, str(f))
    media = os.path.join(os.getcwd(), 'media')
    path = os.path.join(media, filename)

    with open(path, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)

    return path


# Create your views here.
def upload(request, category):
    if category == 'anime':
        form = ImageUpload(request.POST or None, request.FILES or None)
    else:
        form = ImageUpload(request.POST or None, request.FILES or None)

    if form.is_valid():
        img = request.FILES['image']
        img_path = save_file(img, category)

    template = 'form.html'
    context = {
        'form': form,
        'category': category,
    }

    return render(request, template_name=template, context=context)