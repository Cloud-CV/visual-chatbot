from django.contrib import admin

from .models import Job, Dialog


class JobAdmin(admin.ModelAdmin):
    list_display = ('job_id', 'img_url', 'caption', 'created_at')


class DialogAdmin(admin.ModelAdmin):
    list_display = ('job_id', 'img_url', 'question', 'answer', 'created_at')

admin.site.register(Job, JobAdmin)
admin.site.register(Dialog, DialogAdmin)
