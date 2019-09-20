import os
import random
import urllib
import uuid

from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render

import chat.constants as constants
from .models import Job
from .sender import viscap


def landing(request, template_name="chat/landing.html"):
    return render(request, template_name)

def home(request, template_name="chat/index.html"):
    socketid = uuid.uuid4()
    intro_message = random.choice(constants.BOT_INTORDUCTION_MESSAGE)

    if request.method == "POST":
        try:
            socketid = request.POST.get("socketid")
            question = request.POST.get("question")
            img_path = request.POST.get("img_path")
            job_id = request.POST.get("job_id")
            history = request.POST.get("history", "")
            img_path = urllib.unquote(img_path)
            abs_image_path = str(img_path)
            viscap(str(abs_image_path), socketid, job_id, str(question), str(history))

            return JsonResponse({"success": True})
        except Exception:
            return JsonResponse({"success": False})

    elif request.method == "GET":
        return render(request, template_name, {
                                               "socketid": socketid,
                                               "bot_intro_message": intro_message})


# Create a Job for captioning
def upload_image(request):

    if request.method == "POST":
        image = request.FILES['file']
        socketid = request.POST.get('socketid')
        output_dir = os.path.join(settings.MEDIA_ROOT, 'svqa', socketid)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        img_path = os.path.join(output_dir, str(image))
        handle_uploaded_file(image, img_path)
        img_url = img_path.replace(settings.BASE_DIR, "")
        job = Job.objects.create(job_id=socketid, image=img_url)
        viscap(img_path, socketid, job.id)

        return JsonResponse({"file_path": img_path, "img_url": img_url, "job_id": job.id})
    else:
        raise TypeError("Only POST requests allowed, check request method!")


def handle_uploaded_file(f, path):
    with open(path, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)
