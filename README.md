# assignment01-tandin2000
>Datasets
### Datasets avaliable [here](https://susanqq.github.io/UTKFace/) ###
>Report


1.	INTRODUCTION FOR PROBLEM
```
##Topic – Age classification using Deep learning for low spec device.
How accurate is age prediction from facial images? Is it possible to predict someone's age from their face? 
Age is a crucial factor in determining an individual's health, appearance, and life expectancy. The ability to estimate a person's age accurately would be helpful in many fields such as marketing, law enforcement, cosmetics, and medicine. Facial recognition is becoming prevalent in our day-to-day lives, from unlocking our smartphones to identifying criminals where facial recognition is being used as a vital tool for security aspects. Deep learning techniques have been applied to image-processing tasks such as object detection, semantic segmentation, and action recognition where they have been able to produce high-accuracy results. In order to facilitate these deep-learning techniques and produce highly accurate results, devices should expedite high CUP consumption making the devices very expensive. However, in recent years, the misuse of facial recognition has contributed to raising privacy concerns. Major factors the deep learning and image processing algorithms take into consideration are illumination, head profile, hairstyle, facial expression, mustache, beards, cosmetics, UV rays, etc. which makes it almost impossible to have the same face twice as all the factors contribute to making a face unique at given instance and changes to them might produce an entirely new face. Facial ageing is a collective result of variations in both soft tissues and bone structure of the human face. Hence these age-based variations could be used to analyze and predict facial age.
```


 
2.	METHODOLOGY 
```
A popular dataset named, UTKFace is used in this project. The dataset consists of 23 thousand images and each image is labeled with gender, age, and ethnicity. This dataset is trained by the following methods; 
-	Reading the image files as 3D NumPy arrays. They will be using 3-channeled RGB images for training the model, so each array will have a shape of ‘[ img_width ,img_height , 3 ]’.
-	Splitting the filename so as to parse the age of the person in the corresponding image. We use the ‘tf.strings.split()’ method for performing this task.
-	The maximum value of our target variable is 116 years which is used for normalizing the age variable. Once these operations have been performed, we are left with N samples where each sample consists of an image array [ 200, 200, 3] and its corresponding label, the age of that person, which has a shape [ 1 , ] The use of ‘tf.data.Dataset’  helps us to process the data faster, taking advantage of parallel computing. The above two operations will be mapped on each filename using ‘tf.data.Dataset.map’ method. 
-	Creating two splits from our dataset, one for training the model and another for testing the model. The fraction of the dataset which will be used for testing the model is determined by TRAIN_TEST_SPLIT.
```

3.	SELECTED DEEP LEARNING ARCHITECTURES
```
In this approach, I treat age estimation as a regression problem. The aim was to develop a model which has lesser parameters ( which implies lesser inference time and size ) but powerful enough so that it can generalize better.

-	The model takes in a batch of shape [ None , 200 , 200 , 3 ] and performs a number of convolutions on it as determined by num_blocks.
-	Each block consists of a sequence of layers : Conv2D -> BatchNorm -> LeakyReLU

# Define the conv block.
if lite_model:
x = tf.keras.layers.SeparableConv2D( num_filters ,
kernel_size=kernel_size ,
strides=strides
, use_bias=False ,
kernel_initializer=tf.keras.initializers.HeNormal() ,
kernel_regularizer=tf.keras.regularizers.L2( 1e-5 )
)( x )
else:
x = tf.keras.layers.Conv2D( num_filters ,
kernel_size=kernel_size ,
strides=strides ,
use_bias=False ,
kernel_initializer=tf.keras.initializers.HeNormal() ,
kernel_regularizer=tf.keras.regularizers.L2( 1e-5 )
)( x )

x = tf.keras.layers.BatchNormalization()( x )
x = tf.keras.layers.LeakyReLU( leaky_relu_alpha )( x )

If lite_model is set to True, The Separable Convolutions occurs which have lesser parameters. We could achieve a faster model, compromising its performance. Stack of suchnum_blocks blocks sequentially, where the number of filters for each layer is taken from num_filters. After the adding a number of Dense layers to learn the features extracted by convolutional layers. There adds a Dropout layer that reduce overfitting. The rate for each Dropout layer is decreased subsequently for each layer, so that the learnability of Dense layer with lesser units (neurons) is not affected.

def dense( x , filters , dropout_rate ):
    x = tf.keras.layers.Dense( filters , kernel_regularizer=tf.keras.regularizers.L2( 0.1 ) , bias_regularizer=tf.keras.regularizers.L2( 0.1 ) )( x )
    x = tf.keras.layers.LeakyReLU( alpha=leaky_relu_alpha )( x )
    x = tf.keras.layers.Dropout( dropout_rate )( x )
    return x

The output of the model is a tensor with shape [ None, 1 ]
```

4.	DETAILED ANALYSIS FOR DATASET
```
The UTKFace dataset is a large-scale face-based dataset with over 20,000 images of individuals whose ages range from 0 to 116 years old with annotations of age, gender, and ethnicity. The images cover large variations in pose, facial expression, illumination, occlusion, resolution, etc. This dataset could be used on a variety of tasks, e.g., face detection, age estimation, age progression/regression, landmark localization, etc. Some sample images are shown below 

Highlights 
•	Consists of 20k+ fae images in the wild (only single face in one image)
•	Provides correspondingly aligned and cropped faces
•	Provides corresponding landmarks (68 points)
•	Each image is labeled by age, gender, and ethnicity
Samples

 
Figure 1Dataset examples
 
Labels
The labels of each face image is embedded in the file name, formatted like
•	[age]_[gender]_[race]_[date&time].jpg
•	[age] is an integer from 0 to 116, indicating the age
•	[gender] is either 0 (male) or 1 (female)
•	[race] is an integer from 0 to 4, denoting White, Black, Asian, Indian, and Others (like Hispanic, Latino, Middle Eastern).
•	[date&time] is in the format of yyyymmddHHMMSSFFF, showing the date and time an image was collected to UTKFace

```


5.	FEATURE SELECTION & PREPROCESSING TECHNIQUES
```
For evaluating the performance of our model, we use Mean Absolute Error as a metric. 

Callbacks:
•	tf.keras.callbacks.ModelCheckpoint to save the Keras model as an H5 file after every epoch.

•	tf.keras.callbacks.TensorBoard to visualize the training with TensorBoard.
•	tf.keras.callbacks.LearningRateScheduler to decrease the learning rate over a certain number of epochs, so as to make smaller steps, as the optimizer reaches near the minima of the loss function.

def scheduler( epochs , learning_rate ):
    if epochs < num_epochs * 0.25:
        return learning_rate
    elif epochs < num_epochs * 0.5:
        return 0.0005
    elif epochs < num_epochs * 0.75:
        return 0.0001
    else:
        return 0.000095

tf.keras.callbacks.EarlyStopping to stop the training when the evaluation metric i.e the MAE stops improving on the test dataset.Load a .h5 file for evaluation. Note: Need to run the code cell which creates train_ds and test_ds.

batch_size = 128
model = tf.keras.models.load_model( 'model_lite_age.h5' )
p = model.evaluate( test_ds.batch( batch_size ) )
print( p )

Save the Keras model to the local disk, so that we can resume training if needed.

model_name = 'model_age' #@param {type: "string"}
model_name_ = model_name + '.h5'
model.save( model_name_ )
files.download( model_name_ )

Convert to TensorFlow Lite format 
The model is to be deployed in an Android app, where it uses TF Lite Android package to parse the model and make predictions. It uses the TFLiteConverter API to convert our Keras Model ( .h5 ) to a TF Lite buffer ( .tflite ). We'll produce two TF Lite buffers, one with float16 quantization and other non-quantized model.

converter = tf.lite.TFLiteConverter.from_keras_model( model )
converter.optimizations = [ tf.lite.Optimize.DEFAULT ]
converter.target_spec.supported_types = [ tf.float16 ]
buffer = converter.convert()
open( '{}_q.tflite'.format( model_name ) , 'wb' ).write( buffer )
files.download( '{}_q.tflite'.format( model_name ) )

For conversion to a non-quantized TF Lite buffer.

converter = tf.lite.TFLiteConverter.from_keras_model( model )
buffer = converter.convert()
open( '{}_nonq.tflite'.format( model_name ) , 'wb' ).write( buffer )
files.download( '{}_nonq.tflite'.format( model_name ) )

Utility to zip and download a directory

dir_to_zip = 'tb_logs' #@param {type: "string"}
output_filename = 'logs.zip' #@param {type: "string"}
delete_dir_after_download = "No"  #@param ['Yes', 'No']
os.system( "zip -r {} {}".format( output_filename , dir_to_zip ) )
if delete_dir_after_download == "Yes":
    os.system( "rm -r {}".format( dir_to_zip ) )
files.download( output_filename )

Use this method to delete a directory.

dir_path = ''  #@param {type: "string"}
os.system( f'rm -r {dir_path}')
```
