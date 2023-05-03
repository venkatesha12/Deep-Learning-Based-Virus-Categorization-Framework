from django.shortcuts import render, redirect
from django.http import HttpResponse, request

from .models import *
from django.core import serializers
from django.template import Context
import numpy as np
import matplotlib.pyplot as plt;

from .SVMAlgo import SVMAlgo
from .NBAlgo import NBAlgo
from .KNNAlgo import KNNAlgo
from .NNAlgo import NNAlgo
from .RFAlgo import RFAlgo
from .Testing import Testing
from .Prediction import Prediction





def home(request):
    return render(request, 'index.html')


def userhomedef(request):
    if "useremail" in request.session:
        uid = request.session["useremail"]
        d = users.objects.filter(email__exact=uid)
        return render(request, 'user_home.html', {'data': d[0]})

    else:
        return render(request, 'user.html')


def adminhomedef(request):
    if "adminid" in request.session:
        uid = request.session["adminid"]
        return render(request, 'admin_home.html')

    else:
        return render(request, 'admin.html')



def userlogoutdef(request):
    try:
        del request.session['useremail']
    except:
        pass
    return render(request, 'user.html')


def adminlogoutdef(request):
    try:
        del request.session['adminid']
    except:
        pass
    return render(request, 'admin.html')


def adminlogindef(request):
    return render(request, 'admin.html')


def userlogindef(request):
    return render(request, 'user.html')


def signupdef(request):
    return render(request, 'signup.html')


def usignupactiondef(request):
    email = request.POST['mail']
    pwd = request.POST['pwd']
    phone = request.POST['ph']
    name = request.POST['name']

    d = users.objects.filter(email__exact=email).count()
    if d > 0:
        return render(request, 'signup.html', {'msg': "Email Already Registered"})
    else:
        d = users(name=name, email=email, pwd=pwd, phone=phone)
        d.save()
        return render(request, 'signup.html', {'msg': "Register Success, You can Login.."})

    return render(request, 'signup.html', {'msg': "Register Success, You can Login.."})


def userloginactiondef(request):
    if request.method == 'POST':
        uid = request.POST['uid']
        pwd = request.POST['pwd']
        d = users.objects.filter(email__exact=uid).filter(pwd__exact=pwd).count()

        if d > 0:
            d = users.objects.filter(email__exact=uid)
            request.session['useremail'] = uid
            request.session['username'] = d[0].name
            return render(request, 'user_home.html', {'data': d[0]})

        else:
            return render(request, 'user.html', {'msg': "Login Fail"})

    else:
        return render(request, 'user.html')


def adminloginactiondef(request):
    if request.method == 'POST':
        uid = request.POST['uid']
        pwd = request.POST['pwd']

        if uid == 'admin' and pwd == 'admin':
            request.session['adminid'] = 'admin'
            return render(request, 'admin_home.html')

        else:
            return render(request, 'admin.html', {'msg': "Login Fail"})

    else:
        return render(request, 'admin.html')


def trainingpage(request):
    if "adminid" in request.session:

        return render(request, 'trainingpage.html')

    else:
        return render(request, 'admin.html')

def svm(request):
    
    SVMAlgo.classification()

    return render(request, 'trainingpage.html', {'msg': "SVM Classifier Training Completed Successfully"})
    
def nbdef(request):
    
    NBAlgo.classification()

    return render(request, 'trainingpage.html', {'msg': "Naive Bayes Classifier Training Completed Successfully"})
def knndef(request):
    
    KNNAlgo.classification()

    return render(request, 'trainingpage.html', {'msg': "KNN Classifier Training Completed Successfully"})
def nndef(request):
    
    NNAlgo.classification()

    return render(request, 'trainingpage.html', {'msg': "Artificial Neural Network Classifier Training Completed Successfully"})
def rfdef(request):
    
    RFAlgo.classification()

    return render(request, 'trainingpage.html', {'msg': "Random Forest Classifier Training Completed Successfully"})



def svmtesting(request):
    
    res=Testing.predict('svm_model.sav')
    d = accuracysc.objects.filter(algo='SVM').delete()
    d = accuracysc(algo='SVM', accuracyv=float(res))
    d.save()
    return render(request, 'trainingpage.html', {'msg': "Testing Completed with SVM, Accuracy : "+str(res)+"%"})
def nbtesting(request):
    
    res=Testing.predict('nb_model.sav')
    d = accuracysc.objects.filter(algo='NB').delete()
    d = accuracysc(algo='NB', accuracyv=float(res))
    d.save()
    return render(request, 'trainingpage.html', {'msg': "Testing Completed with Naive Bayes, Accuracy : "+str(res)+"%"})


def knntesting(request):
    
    res=Testing.predict('knn_model.sav')
    d = accuracysc.objects.filter(algo='KNN').delete()
    d = accuracysc(algo='KNN', accuracyv=float(res))
    d.save()
    return render(request, 'trainingpage.html', {'msg': "Testing Completed with KNN, Accuracy : "+str(res)+"%"})


def nntesting(request):
    
    res=Testing.predict('nn_model.sav')
    d = accuracysc.objects.filter(algo='NN').delete()
    d = accuracysc(algo='NN', accuracyv=float(res))
    d.save()
    return render(request, 'trainingpage.html', {'msg': "Testing Completed with Artificial Neural Network, Accuracy : "+str(res)+"%"})


def rftesting(request):
    
    res=Testing.predict('rf_model.sav')
    d = accuracysc.objects.filter(algo='RF').delete()
    d = accuracysc(algo='RF', accuracyv=float(res))
    d.save()
    return render(request, 'trainingpage.html', {'msg': "Testing Completed with Random Forest, Accuracy : "+str(res)+"%"})

def viewaccuracy(request):
    if "adminid" in request.session:
        d = accuracysc.objects.all()
        
        return render(request, 'viewaccuracy.html', {'data': d})

    else:
        return render(request, 'admin.html')



def viewgraph(request):
    if "adminid" in request.session:
        algos = []
        row = accuracysc.objects.all()
        rlist = []
        for r in row:
            algos.append(r.algo)
            rlist.append(r.accuracyv)


        height = rlist
        # print(height)
        try:

            bars = algos
            y_pos = np.arange(len(bars))
            plt.bar(bars, height, color=['purple','blue','green','yellow', 'cyan'])
            # plt.plot( bars, height )
            plt.xlabel('Algorithms')
            plt.ylabel('Accuracy ')
            plt.title('Accuracy Measure')
            plt.savefig('g1.jpg')
        except:
            pass

        from PIL import Image 
        
        im = Image.open(r"g1.jpg") 
          
        im.show()

        return redirect('viewaccuracy')

def trainingpage2(request):
    if "adminid" in request.session:

        return render(request, 'trainingpage2.html')

    else:
        return render(request, 'admin.html')



def cnntrain(request):
    from .Classification import main
    
    res=main()
    d = accuracycnn.objects.all()
    d.delete()
    d = accuracycnn(accuracyv=res[0], accuracyloss=1-res[0])
    d.save()
    return render(request, 'trainingpage2.html', {'msg': "Training & Testing Completed with CNN Algorithm "})


def cnnacc(request):
    if "adminid" in request.session:
        d = accuracycnn.objects.all()
        algos = ['Accuracy','Loss']
        row = accuracycnn.objects.all()
        rlist = []
        for r in row:
            rlist.append(r.accuracyloss)
            rlist.append(r.accuracyv)


        height = rlist
        plt.clf()
        # print(height)
        try:

            bars = algos
            y_pos = np.arange(len(bars))
            plt.bar(bars, height, color=['purple','red'])
            # plt.plot( bars, height )
            plt.xlabel('')
            plt.ylabel(' ')
            plt.title('CNN Performance')
            plt.savefig('g2.jpg')
        
        except:
            pass

        from PIL import Image 
        im = Image.open(r"g2.jpg") 
          
        im.show()


        
        return render(request, 'viewaccuracy2.html', {'data': d})

    else:
        return render(request, 'admin.html')



def predictiondef(request):
    if "useremail" in request.session:

        return render(request, 'prediction.html')

    else:
        return render(request, 'user.html')



def apidetection(request):
    if request.method == 'POST':
        dataset = [row.split()[1] for row in open('malware_API_dataset.csv').readlines()[1:]]
        file = request.POST['file']
        
        filename="Data\\"+file
        #file=[row.split() for row in open(file).readlines()]
        file = open(filename, 'rt')
        text = file.read()
        file.close()
        # split into words
        from nltk.tokenize import word_tokenize
        tokens = word_tokenize(text)
        # remove all tokens that are not alphabetic
        words = [word for word in tokens if word.isalpha()]
        calls=""
        set3 = set(dataset)&set(words) 
        print(set3)

        if len(set3)>0:
            for s in set3:
                calls=calls+s+" "
            calls=calls.strip()
            print(calls)
            r=Prediction.predict(calls)
            print(r)
            return render(request, 'result.html',{'data': r})
        else:
            return render(request, 'prediction.html',{'msg': 'Not a Malware file'})
               
      
    else:
        return render(request, 'prediction.html')



def cnndetection(request):
    from .Cnn_predict import predict
    file = request.POST['file']
    
    res=predict(file)
    return render(request, 'result2.html', {'data': res})

