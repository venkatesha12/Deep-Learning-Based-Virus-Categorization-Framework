
import numpy as np
import operator
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model
from keras.preprocessing import image

def predict(img):
    # load model
    img_width, img_height = 64, 64
    model_path = 'malware_cnn.h5'

    model = load_model(model_path)
  

    # Prediction on a new picture
    from keras.preprocessing import image as image_utils

    from PIL import Image, ImageTk

    class_labels = ['Adialer.C','Agent.FYI','Allaple.A','Allaple.L','Alueron.gen!J','Autorun.K', 'C2LOP.P','C2LOP.gen!g', 'Dialplatform.B', 'Dontovo.A', 'Fakerean', 'Instantaccess', 'Lolyda.AA1', 'Lolyda.AA2', 'Lolyda.AA3', 'Lolyda.AT', 'Malex.gen!J', 'Obfuscator.AD', 'Rbot!gen', 'Skintrim.N', 'Swizzor.gen!E','Swizzor.gen!I', 'VB.AT','Wintrim.BX','Yuner.A']
    test_image = image.load_img(img, target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)

    test_image /= 255
    result = model.predict(test_image)

    decoded_predictions = dict(zip(class_labels, result[0]))
    decoded_predictions = sorted(decoded_predictions.items(), key=operator.itemgetter(1), reverse=True)
    print(decoded_predictions[0][0])

    count = 1
    for key, value in decoded_predictions[:5]:
        print("{}. {}: {:8f}%".format(count, key, value * 100))
        count += 1

    return decoded_predictions[0][0]

if __name__ == '__main__':
	print(predict("01d3dd5cd0c2c5d08ab3d30d0930113e - Copy (2).png"),'<<<')