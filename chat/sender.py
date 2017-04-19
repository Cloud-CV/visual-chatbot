from django.conf import settings
from chat.utils import log_to_terminal

import os
import pika
import sys
import json


def visdial(input_question, history, image_path, socketid, job_id):
    connection = pika.BlockingConnection(pika.ConnectionParameters(
        host='localhost'))
    channel = connection.channel()

    channel.queue_declare(queue='visdial_task_queue', durable=True)
    message = {
        'image_path': image_path,
        'input_question': input_question,
        'history': history,
        'socketid': socketid,
        'job_id': job_id,
    }

    channel.basic_publish(exchange='',
                          routing_key='visdial_task_queue',
                          body=json.dumps(message),
                          properties=pika.BasicProperties(
                              delivery_mode=2,  # make message persistent
                          ))
    connection.close()


def captioning(image_path, socketid, job_id):
    connection = pika.BlockingConnection(pika.ConnectionParameters(
        host='localhost'))
    channel = connection.channel()

    channel.queue_declare(queue='visdial_captioning_task_queue', durable=True)
    message = {
        'image_path': image_path,
        'socketid': socketid,
        'job_id': job_id,
    }
    channel.basic_publish(exchange='',
                          routing_key='visdial_captioning_task_queue',
                          body=json.dumps(message),
                          properties=pika.BasicProperties(
                              delivery_mode=2,  # make message persistent
                          ))
    connection.close()
