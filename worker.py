from __future__ import absolute_import

import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'visdial.settings')

import django
django.setup()

from django.conf import settings
from chat.utils import log_to_terminal
from chat.models import Job, Dialog
from models import VisDialDummyModel
import chat.constants as constants
import pika
import yaml
import json
import traceback

VisDialModel = VisDialDummyModel

VisDialATorchModel = VisDialModel(
    constants.VISDIAL_CONFIG['input_json'],
    constants.VISDIAL_CONFIG['load_path'],
    constants.VISDIAL_CONFIG['beamSize'],
    constants.VISDIAL_CONFIG['beamLen'],
    constants.VISDIAL_CONFIG['sampleWords'],
    constants.VISDIAL_CONFIG['temperature'],
    constants.VISDIAL_CONFIG['gpuid'],
    constants.VISDIAL_CONFIG['backend'],
    constants.VISDIAL_CONFIG['proto_file'],
    constants.VISDIAL_CONFIG['model_file'],
    constants.VISDIAL_CONFIG['maxThreads'],
    constants.VISDIAL_CONFIG['encoder'],
    constants.VISDIAL_CONFIG['decoder'],
)

connection = pika.BlockingConnection(pika.ConnectionParameters(
    host='localhost'))

channel = connection.channel()

channel.queue_declare(queue='visdial_task_queue', durable=True)

django.db.close_old_connections()


def callback(ch, method, properties, body):
    try:
        body = yaml.safe_load(body)
        body['history'] = body['history'].split("||||")

        result = VisDialATorchModel.predict(
            body['image_path'], body['history'], body['input_question'])
        result['input_image'] = str(
            result['input_image']).replace(settings.BASE_DIR, '')
        result['question'] = str(result['question'])
        result['answer'] = str(result['answer'])
        result['history'] = result['history']
        result['history'] = result['history'].replace("<START>", "")
        result['history'] = result['history'].replace("<END>", "")

        log_to_terminal(body['socketid'], {"result": json.dumps(result)})
        ch.basic_ack(delivery_tag=method.delivery_tag)

        try:
            job = Job.objects.get(id=int(body['job_id']))
            Dialog.objects.create(job=job, question=result['question'], answer=result['answer'].replace("<START>", "").replace("<END>", ""))
        except:
            print(str(traceback.print_exc()))

        django.db.close_old_connections()

    except Exception:
        print(str(traceback.print_exc()))

channel.basic_consume(callback,
                      queue='visdial_task_queue')

channel.start_consuming()
