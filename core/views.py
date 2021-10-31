import pathlib
import uuid
from django.shortcuts import redirect, render
from django.core.files.base import ContentFile

from core.models import History
from .form import ImageUpload
import os
from cyclegan.utils import get_predicted_image
from PIL import Image
import cv2


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


def results(request):
    this = request.session['object']
    this = History.objects.get(pk=this)

    img_path = this.input_img.path
    category = request.session['category']

    output_name = '/media/output.jpg'
    model_dir = os.path.join(os.getcwd(), './cyclegan/trained_models')

    output_img = get_predicted_image(
        img_path,
        output=category,
        output_name=output_name,
        model_dir=model_dir,
        does_return=True,
    )

    ret, buf = cv2.imencode('.jpg', output_img)

    output_img = ContentFile(buf.tobytes())
    this.output_img.save('output.jpg', output_img)
    this.save()

    context = {
        'object': this,
    }

    template = 'result.html'

    return render(request, template_name=template, context=context)


# Create your views here.
def upload(request, category):
    if category == 'anime':
        form = ImageUpload(request.POST or None, request.FILES or None)
    else:
        form = ImageUpload(request.POST or None, request.FILES or None)

    if form.is_valid():
        img = request.FILES['image']
        print(type(img))
        this = History()
        this.input_img.save('input.jpg', img)
        this.save()
        # img_path = save_file(img, category)

        request.session['object'] = this.pk
        request.session['category'] = category

        return redirect('result')

    template = 'form.html'
    context = {
        'form': form,
        'category': category,
    }

    return render(request, template_name=template, context=context)