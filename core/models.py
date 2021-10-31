from django.db import models
import pathlib
import uuid


def image_upload_handle(instance, filename):
    file_path = pathlib.Path(filename)
    new_file_path = str(uuid.uuid1())
    return f"media/{new_file_path}{file_path.suffix}"


# Create your models here.
class History(models.Model):
    input_img = models.ImageField(
        verbose_name='input_img',
        name='input_img',
        upload_to=image_upload_handle,
        blank=False,
        null=False,
    )

    output_img = models.ImageField(
        verbose_name='output_img',
        name='output_img',
        upload_to=image_upload_handle,
        blank=True,
        null=True,
    )