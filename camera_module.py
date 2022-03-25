from cProfile import label
import cv2
import numpy as np
import tensorflow.keras as keras
import tensorflow as tf

cap = cv2.VideoCapture(0)


# Load the CNN model
 
# Make predictions using the input from the camera

# Show camera text, according to the model's prediction
model = tf.keras.models.load_model('./CNN_Model/TestModelLR1e-5_doubleLayers_softmax_noDropout_noAugmentation_newTestData')

data = np.ndarray( (1, 256, 256), dtype = np.float32)

image_width = 256
image_height = 256

image_size = ( image_height, image_width )

labels = ['no_mask', 'mask']

while True:
    ret, frame = cap.read()


    frame = cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY )

    # resize
    resized = cv2.resize( frame, image_size, fx = image_width / image_height, fy = 1, interpolation = cv2.INTER_NEAREST )

    # normalize 
    normalized = np.float32(resized) 

    # convert to array
    image_array = (np.asarray( normalized ) / 127.0 )- 1

    # reshape

    normalized_reshaped = np.resize( image_array, image_size)

    data[0] = normalized_reshaped

    # predict
    prediction = model.predict( data )

    index = np.argmax(prediction)

    classLabel = labels[index]

    accuracy = prediction[0][index]

    
    cv2.putText(frame, classLabel, (50,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2, cv2.LINE_AA )
    cv2.putText(frame, str(accuracy * 100) + '%', (50,150),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2, cv2.LINE_AA )
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows
