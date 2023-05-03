from django.db import models
# Create your models here.


class users(models.Model):
	name=models.CharField(max_length=100);
	email=models.CharField(max_length=100);
	pwd=models.CharField(max_length=100);
	phone=models.CharField(max_length=100);

class accuracysc(models.Model):
    algo=models.CharField(max_length=100);
    accuracyv=models.FloatField(max_length=1000)

class accuracycnn(models.Model):
    accuracyv=models.FloatField(max_length=1000);
    accuracyloss=models.FloatField(max_length=1000)

