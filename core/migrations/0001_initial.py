# Generated by Django 3.2.8 on 2021-10-31 05:50

import core.models
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='History',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('input_img', models.ImageField(upload_to=core.models.image_upload_handle, verbose_name='input_img')),
                ('output_img', models.ImageField(blank=True, null=True, upload_to=core.models.image_upload_handle, verbose_name='output_img')),
            ],
        ),
    ]
