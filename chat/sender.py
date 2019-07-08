from django.conf import settings
from chat.utils import log_to_terminal

import os
import pika
import sys
import json


def svqa(input_question, history, image_path, socketid, job_id):
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
    print(f"svqa-msg: {message}")
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
    print(f"cap-message: {message}")
    channel.basic_publish(exchange='',
                          routing_key='visdial_captioning_task_queue',
                          body=json.dumps(message),
                          properties=pika.BasicProperties(
                              delivery_mode=2,  # make message persistent
                          ))
    connection.close()


def viscap(image_path, socketid, job_id, input_question=None, history=None):
    connection = pika.BlockingConnection(pika.ConnectionParameters(
        host='localhost'))
    channel = connection.channel()
    routing_key = "visdial_caption_task_queue"

    if input_question is None and history is None:
        channel.queue_declare(queue=routing_key, durable=True)
        message = {
            'image_path': image_path,
            'socketid': socketid,
            'job_id': job_id,
            'type': "caption"
        }

    else:
        channel.queue_declare(queue=routing_key, durable=True)
        message = {
            'image_path': image_path,
            'input_question': input_question,
            'history': history,
            'socketid': socketid,
            'job_id': job_id,
            'type': "visdial"
        }

    channel.basic_publish(exchange='',
                          routing_key=routing_key,
                          body=json.dumps(message),
                          properties=pika.BasicProperties(
                              delivery_mode=2,  # make message persistent
                          ))
    connection.close()
