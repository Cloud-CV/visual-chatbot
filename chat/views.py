from nltk.tokenize import word_tokenize

from django.conf import settings
from django.shortcuts import render
from django.http import JsonResponse

from .sender import svqa, captioning
from .utils import log_to_terminal
from .models import Job, Dialog

import chat.constants as constants
import uuid
import os
import traceback
import random
import urllib2


def home(request, template_name="chat/index.html"):
    socketid = uuid.uuid4()
    intro_message = random.choice(constants.BOT_INTORDUCTION_MESSAGE)
    if request.method == "POST":
        try:
            socketid = request.POST.get("socketid")
            question = request.POST.get("question")
            question = question.replace("?", "").lower()
            img_path = request.POST.get("img_path")
            job_id = request.POST.get("job_id")
            history = request.POST.get("history", "")

            img_path = urllib2.unquote(img_path)
            abs_image_path = str(img_path)

            q_tokens = word_tokenize(str(question))
            if q_tokens[-1] != "?":
                question = "{0}{1}".format(question, "?")

            response = svqa(str(question), str(history),
                            str(abs_image_path), socketid, job_id)
            return JsonResponse({"success": True})
        except Exception, err:
            return JsonResponse({"success": False})

    elif request.method == "GET":
        return render(request, template_name, {
                                               "socketid": socketid,
                                               "bot_intro_message": intro_message})


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

        captioning(img_path, socketid, job.id)

        return JsonResponse({"file_path": img_path, "img_url": img_url, "job_id": socketid})


def handle_uploaded_file(f, path):
    with open(path, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)
