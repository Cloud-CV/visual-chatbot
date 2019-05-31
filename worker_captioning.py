from __future__ import absolute_import

import os

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'visdial.settings')
import sys
print(sys.executable)

import django
django.setup()

from django.conf import settings
from chat.utils import log_to_terminal
from chat.models import Job

import chat.constants as constants

import pika
import yaml
import json
import traceback

from models import CaptioningTorchDummyModel

django.db.close_old_connections()

CaptioningModel = CaptioningTorchDummyModel

CaptioningTorchModel = CaptioningModel(
    constants.CAPTIONING_CONFIG['model_path'],
    constants.CAPTIONING_CONFIG['backend'],
    constants.CAPTIONING_CONFIG['input_sz'],
    constants.CAPTIONING_CONFIG['layer'],
    constants.CAPTIONING_CONFIG['seed'],
    constants.CAPTIONING_GPUID,
)

connection = pika.BlockingConnection(pika.ConnectionParameters(
    host='localhost'))

channel = connection.channel()

channel.queue_declare(queue='visdial_captioning_task_queue', durable=True)
print(' [*] Waiting for messages. To exit press CTRL+C')


def callback(ch, method, properties, body):
    try:
        body = yaml.safe_load(body)
        result = CaptioningTorchModel.predict(
                                              body['image_path'],
                                              constants.CAPTIONING_CONFIG['input_sz'],
                                              constants.CAPTIONING_CONFIG['input_sz'])
        result['input_image'] = str(result['input_image']).replace(settings.BASE_DIR, '')
        log_to_terminal(body['socketid'], {"result": json.dumps(result)})
        ch.basic_ack(delivery_tag=method.delivery_tag)
        print('succesfull callback')

        try:
            Job.objects.filter(id=int(body['job_id'])).update(caption=result['pred_caption'])
        except Exception as e:
            print str(traceback.print_exc())

        django.db.close_old_connections()

    except Exception, err:
        print str(traceback.print_exc())

channel.basic_consume(callback,
                      queue='visdial_captioning_task_queue')

channel.start_consuming()
