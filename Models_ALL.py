from __future__ import print_function
from tensorflow import keras
from tensorflow.keras.datasets import cifar10, mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
import os
from tensorflow.keras import regularizers
from tensorflow.keras.utils import HDF5Matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint


from sklearn.model_selection import train_test_split

batch_size = 32
num_classes = 2
epochs = 1
data_augmentation = False
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
# Save model and weights
if not os.path.isdir(save_dir):
   os.makedirs(save_dir)


# The data, split between train and test sets:
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = HDF5Matrix('camelyonpatch_level_2_split_train_x.h5', 'x')
y_train = HDF5Matrix('camelyonpatch_level_2_split_train_y.h5', 'y')

x_test = HDF5Matrix('camelyonpatch_level_2_split_valid_x.h5', 'x')
y_test = HDF5Matrix('camelyonpatch_level_2_split_valid_y.h5', 'y')

print("Dataset Loaded")
# x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.33, random_state=42)
# print("Data Splitting Done")

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = y_train[:].reshape(-1,1)
y_test =  y_test[:].reshape(-1,1)
# x_train = x_train.reshape(-1,25,25,1)
# x_test =  x_test.reshape(-1,25,25,1)

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print("Categorical Variables Created")


from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.vgg19 import VGG19


from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.densenet import DenseNet169
from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras.applications.nasnet import NASNetLarge
from tensorflow.keras.applications.nasnet import NASNetMobile
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

Models = []
Names = []
model = VGG16(weights=None, include_top=True, input_shape=x_train.shape[1:], classes=2)
Models.append(model)
Names.append("VGG16")
#print("VGG16")
model = ResNet50(weights=None, include_top=True,input_shape=x_train.shape[1:], classes=2)
Models.append(model)
Names.append("ResNet50")
#print("ResNet50")
model = Xception(weights=None, include_top=True,input_shape=x_train.shape[1:], classes=2)
Models.append(model)
Names.append("Xception")
#print("Xception")
model = VGG19(weights=None, include_top=True,input_shape=x_train.shape[1:], classes=2)
Models.append(model)
Names.append("VGG19")
# print("VGG19")


from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.applications.resnet import ResNet101
from tensorflow.keras.applications.resnet import ResNet152
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.applications.resnet_v2 import ResNet101V2
from tensorflow.keras.applications.resnet_v2 import ResNet152V2
model = ResNet50(weights=None, include_top=True,input_shape=x_train.shape[1:], classes=2)
Models.append(model)
Names.append("ResNet50")
#print("ResNet50")
model = ResNet101(weights=None, include_top=True,input_shape=x_train.shape[1:], classes=2)
Models.append(model)
Names.append("ResNet101")
#print("ResNet101")
model = ResNet152(weights=None, include_top=True,input_shape=x_train.shape[1:], classes=2)
Models.append(model)
Names.append("ResNet152")
#print("ResNet152")
model = ResNet50V2(weights=None, include_top=True,input_shape=x_train.shape[1:], classes=2)
Models.append(model)
Names.append("ResNet50V2")
#print("ResNet50V2")
model = ResNet101V2(weights=None, include_top=True,input_shape=x_train.shape[1:], classes=2)
Models.append(model)
Names.append("ResNet101V2")
#print("ResNet101V2")
model = ResNet152V2(weights=None, include_top=True,input_shape=x_train.shape[1:], classes=2)
Models.append(model)
Names.append("ResNet152V2")
#print("ResNet152V2")


model = InceptionV3(weights=None, include_top=True,input_shape=x_train.shape[1:], classes=2)
Models.append(model)
Names.append("InceptionV3")
#print("InceptionV3")
model = InceptionResNetV2(weights=None, include_top=True,input_shape=x_train.shape[1:], classes=2)
Models.append(model)
Names.append("InceptionResNetV2")
#print("InceptionResNetV2")
model = MobileNet(weights=None, include_top=True,input_shape=x_train.shape[1:], classes=2)
Models.append(model)
Names.append("MobileNet")
#print("MobileNet")

model = DenseNet121(weights=None, include_top=True,input_shape=x_train.shape[1:], classes=2)
Models.append(model)
Names.append("DenseNet121")
#print("DenseNet121")
model = DenseNet169(weights=None, include_top=True,input_shape=x_train.shape[1:], classes=2)
Models.append(model)
Names.append("DenseNet169")
#print("DenseNet169")
model = DenseNet201(weights=None, include_top=True,input_shape=x_train.shape[1:], classes=2)
Models.append(model)
Names.append("DenseNet201")
#print("DenseNet201")

model = NASNetLarge(weights=None, include_top=True,input_shape=x_train.shape[1:], classes=2)
Models.append(model)
Names.append("NASNetLarge")
#print("NASNetLarge")
model = NASNetMobile(weights=None, include_top=True,input_shape=x_train.shape[1:], classes=2)
Models.append(model)
Names.append("NASNetMobile")
#print("NASNetMobile")

model = MobileNetV2(weights=None, include_top=True,input_shape=x_train.shape[1:], classes=2)
Models.append(model)
Names.append("MobileNetV2")
#print("MobileNetV2")


# initiate RMSprop optimizer
opt = keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)
sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
# Let's train the model using RMSprop


#x_train = x_train[:].astype('float32')
#x_test = x_test[:].astype('float32')
#x_train /= 255
#x_test /= 255


for m,n in zip(Models,Names):
    #print(m.summary())
    print(n)
    #print(save_dir+ "/"+n+".hdf5")
    #continue
    m.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])
    if not data_augmentation:
       print('Not using data augmentation.')
       #mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
       checkpoint = ModelCheckpoint(save_dir+ "/"+n+".hdf5", monitor='val_acc', verbose=1,save_best_only=True, mode='auto', period=1)
       m.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(x_test, y_test),
                shuffle=True,
                callbacks=[checkpoint])

    else:
       print('Using real-time data augmentation.')
       datagen = ImageDataGenerator(
                  preprocessing_function=lambda x: x/255.)
       #          width_shift_range=4,  # randomly shift images horizontally
       #          height_shift_range=4,  # randomly shift images vertically
       #          horizontal_flip=True,  # randomly flip images
       #          vertical_flip=True)  # randomly flip images
       m.fit_generator(datagen.flow(x_train, y_train,       batch_size=batch_size),steps_per_epoch=len(x_train),epochs=100,use_multiprocessing=True)

       
       model_name = str(n)+'.h5'
       model_path = os.path.join(save_dir, model_name)
       m.save(model_path)
       print('Saved trained model at %s ' % model_path)

       # Score trained model.
       scores = m.evaluate(x_test, y_test, verbose=1)
       print('Test loss:', scores[0])
       print('Test accuracy:', scores[1])
   


