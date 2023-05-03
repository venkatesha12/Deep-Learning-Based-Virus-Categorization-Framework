"""WebC URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from . import views


urlpatterns = [ 

    path('', views.home, name="Welcome"), 
    path('alogin/', views.adminlogindef, name="adminlogindef"), 
    path('ulogin/', views.userlogindef, name="userlogindef"),    
    path('userreg/', views.signupdef, name="signupdef"),    
    path('usignupaction/', views.usignupactiondef, name="usignupactiondef"),
    path('userloginaction/', views.userloginactiondef, name="userloginactiondef"),
    path('userhome/', views.userhomedef, name="userhome"),
    path('userlogout/', views.userlogoutdef, name="userlogout"),
    path('adminloginaction/', views.adminloginactiondef, name="adminloginactiondef"),
    path('adminhome/', views.adminhomedef, name="adminhome"),
    path('adminlogout/', views.adminlogoutdef, name="adminlogout"),
    path('trainingpage/', views.trainingpage, name="trainingpage"),
    path('trainingpage2/', views.trainingpage2, name="trainingpage2"),
    path('svm/', views.svm, name="svm"),
    path('nb/', views.nbdef, name="nb"),
    path('knn/', views.knndef, name="knn"),
    path('nn/', views.nndef, name="nn"),
    path('rf/', views.rfdef, name="rf"),
    path('svmtesting/', views.svmtesting, name="svmtesting"),
    path('nbtesting/', views.nbtesting, name="nbtesting"),
    path('knntesting/', views.knntesting, name="knntesting"),
    path('nntesting/', views.nntesting, name="nntesting"),
    path('rftesting/', views.rftesting, name="rftesting"),
    path('viewaccuracy/', views.viewaccuracy, name="viewaccuracy"),
    path('viewgraph/', views.viewgraph, name="viewgraph"),
    path('cnntrain/', views.cnntrain, name="cnntrain"),
    path('cnnacc/', views.cnnacc, name="cnnacc"),

    path('ulogin/', views.userlogindef, name="userlogindef"),    
    path('userreg/', views.signupdef, name="signupdef"),

    path('prediction/', views.predictiondef, name="prediction"),    
    path('apidetection/', views.apidetection, name="apidetection"),    
    path('cnndetection/', views.cnndetection, name="cnndetection"),    

]
