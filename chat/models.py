from __future__ import unicode_literals

from django.db import models
from django.utils.html import format_html


class Job(models.Model):
    job_id = models.CharField(max_length=1000, blank=True, null=True)
    image = models.CharField(max_length=1000, blank=True, null=True)
    created_at = models.DateTimeField("Time", null=True, auto_now_add=True)
    caption = models.CharField(max_length=1000, blank=True, null=True)

    def __unicode__(self):
        return self.job_id

    def img_url(self):
        return format_html("<img src='{}' height='150px'>", self.image)


class Dialog(models.Model):
    answer = models.CharField(max_length=1000, blank=True, null=True)
    job = models.ForeignKey(Job)
    question = models.CharField(max_length=10000, blank=True, null=True)
    created_at = models.DateTimeField("Time", null=True, auto_now_add=True)

    def __unicode__(self):
        return self.job.job_id

    def img_url(self):
        return format_html("<img src='{}' height='150px'>", self.job.image)
