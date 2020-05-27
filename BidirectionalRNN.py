import re
import numpy as np
import tensorflow as tf
import math
import random
import os.path
from os import path
from mlxtend.preprocessing import shuffle_arrays_unison
gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"

# Prepare a directory to store all the checkpoints.
checkpoint_dir = './ckpt'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

def make_model():
    #batch_size =  32
    #n_timesteps = num_inputs_train
    #n_features = 50
    dropout = 0.8
    LSTM = tf.keras.layers.LSTM
    HIDDEN_SIZE = 256
    #BATCH_SIZE = 512
    LAYERS = 3

    model = tf.keras.models.Sequential()

    model.add(LSTM(HIDDEN_SIZE,
                input_shape = (50, len(chars)),
                return_sequences = True,
                dropout = dropout))

    #model.add(tf.keras.layers.RepeatVector(50))

    for _ in range(LAYERS):
        model.add(tf.keras.layers.Bidirectional(LSTM(HIDDEN_SIZE, return_sequences=True, dropout = dropout)))

    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(len(chars), activation='softmax')))
    model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
    model.summary()
    return model

def make_or_restore_model():
    # Either restore the latest model, or create a fresh one
    checkpoints = [checkpoint_dir + '/' + name
                   for name in os.listdir(checkpoint_dir)]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print('Restoring from', latest_checkpoint)
        return tf.keras.models.load_model(latest_checkpoint)
    print('Creating a new model')
    return make_model()

class CharacterTable(object):
    """Given a set of characters:
    + Encode them to a one-hot integer representation
    + Decode the one-hot or integer representation to their character output
    + Decode a vector of probabilities to their character output
    """
    def __init__(self, chars):
        """Initialize character table.
        # Arguments
            chars: Characters that can appear in the input.
        """
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

    def encode(self, C, num_rows):
        """One-hot encode given string C.
        # Arguments
            C: string, to be encoded.
            num_rows: Number of rows in the returned one-hot encoding. This is
                used to keep the # of rows for each data the same.
        """
        x = np.zeros((num_rows, len(self.chars)))

        for i, c in enumerate(C):
            x[i, self.char_indices[c]] = 1
        return x

    def decode(self, x, calc_argmax=True):
        """Decode the given vector or 2D array to their character output.
        # Arguments
            x: A vector or a 2D array of probabilities or one-hot representations;
                or a vector of character indices (used with `calc_argmax=False`).
            calc_argmax: Whether to find the character index with maximum
                probability, defaults to `True`.
        """
        if calc_argmax:
            x = x.argmax(axis=-1)
        arr = np.zeros(50,)
        j = 0
        for i in x:
            arr[j] = self.indices_char[i]
            j+=1

        return arr

def normalize(X,Y):
    sumx = 0
    sumy = 0
    total_elements = num_inputs_train*num_co
    XNorm = np.zeros(100000)
    YNorm = np.zeros(100000)
    for i in range(total_elements):
        sumx = sumx + X[i]
        sumy = sumy + Y[i]


    meanx = sumx/total_elements
    meany = sumy/total_elements

    for i in range(total_elements):
        XNorm[i] = X[i] - meanx
        YNorm[i] = Y[i] - meany

    sdx = 0
    sdy = 0
    sum2x = 0
    sum2y = 0

    for i in range(total_elements):
        sum2x = sum2x + (XNorm[i]*XNorm[i])
        sum2y = sum2y + (YNorm[i]*YNorm[i])

    sdx = math.sqrt(sum2x/total_elements)
    sdy = math.sqrt(sum2y/total_elements)

    for i in range(total_elements):
        XNorm[i] = XNorm[i]/sdx
        YNorm[i] = YNorm[i]/sdy
    return(XNorm,YNorm)



count = 0
with open('trainInput.txt', 'r') as f:
    for line in f:
        count += 1
num_inputs_train = count
count = 0
with open('testInput.txt', 'r') as f:
    for line in f:
        count += 1
num_inputs_test = count
num_co = 25
num_shuffle = 10




def file2arr(fname, num_lines, num_elements, flag):
    i = 0
    j = 0
    k = 0

    X = np.zeros(num_lines*num_elements*2)
    Y = np.zeros(num_lines*num_elements*2)
    input = np.zeros([num_lines,num_elements,2])
    input_fin = np.zeros([num_lines*num_shuffle,num_elements,2])

    for each in fname:
        for element in each:
            if(element != ',' and element != '\n'):
                num = int(element)
                if k%2 == 0:
                    r1 = X[i]
                    X[i] = float((r1*10)+num)

                else:
                    r2 = Y[j]
                    Y[j] = float((r2*10)+num)

            else:
                if element == ',' and element != '\n':
                    if k%2 == 0:
                        i = i+1
                    else:
                        j = j+1
                    k = k+1



    #normalize the Data
    #XNormTrain,YNormTrain = normalize(X,Y)

    i = 0
    j = 0
    k = 0
    s = 0
    t = 0

    while i<num_lines:
        j = 0
        while j<25:
            k = 0
            while k<2:
                if k%2 == 0:
                    input[i][j][k] = X[s]
                    s = s+1
                    k = k+1
                else:
                    input[i][j][k] = Y[t]
                    t = t+1
                    k = k+1

            j = j+1
        i = i+1


    j = 0
    if flag == 1:
        while j<num_shuffle:
            i = 0
            for item in input:
                random.shuffle(item)
                input_fin[(num_lines*j)+i] = item
                #print(i)
                i+=1
            j+=1
        print(input_fin.shape)
        return input_fin
    else:
        return input

#input training data to array
train_input_fin = np.zeros([num_inputs_train*num_shuffle,num_co,2])
f1=open('trainInput.txt', 'r')
train_input_fin = file2arr(f1, num_inputs_train, num_co, 1)
train_input_fin = np.reshape(train_input_fin,(num_inputs_train*num_shuffle,50))
print(train_input_fin.shape)

#output training data to array
train_output_fin = np.zeros([num_inputs_train*num_shuffle,num_co,2])
f2=open('trainBestlist.txt', 'r')
train_output_fin = file2arr(f2, num_inputs_train, num_co, 1)
train_output_fin = np.reshape(train_output_fin,(num_inputs_train*num_shuffle,50))
print(train_output_fin.shape)

#input testing data to array
test_input = np.zeros([num_inputs_test,num_co,2])
f3=open('testInput.txt', 'r')
test_input = file2arr(f3, num_inputs_test, num_co, 0)
test_input = np.reshape(test_input,(num_inputs_test,50))

#output testing data to array

test_output = np.zeros([num_inputs_test,num_co,2])
f4=open('testBestlist.txt', 'r')
test_output = file2arr(f4, num_inputs_test, num_co, 0)
test_output = np.reshape(test_output,(num_inputs_test,50))

#for line in train_input_fin:
#    print(line)

#c = train_input_fin
#r = train_output_fin
#print('Shuffling the dataset...')
#train_input_fin, train_output_fin = shuffle_arrays_unison(arrays=[train_input_fin, train_output_fin], random_seed=3)
#train_input_fin, train_output_fin = shuffle_in_unison(train_input_fin, train_output_fin)
#print(train_input_fin)
#i = 0
#for line in train_input_fin:
#    print(line)
        #print(r)
        #print(line)
        #print(c)
        #print(i)
#    i+=1

print('Vectorization...')

chars = np.zeros(num_inputs_train*50*num_shuffle)
i = 0


for line in train_input_fin:
    for word in line:
        chars[i] = word

        i+=1
chars = sorted(chars)
chars = list(dict.fromkeys(chars))

x = np.zeros((len(train_input_fin), 50, len(chars)), dtype=np.bool)
y = np.zeros((len(train_input_fin), 50, len(chars)), dtype=np.bool)
x_test = np.zeros((len(test_input), 50, len(chars)), dtype=np.bool)
y_test = np.zeros((len(test_output), 50, len(chars)), dtype=np.bool)
ctable = CharacterTable(chars)
i = 0
for line in train_input_fin:
    x[i] = ctable.encode(line, 50)
    i+=1
i = 0

for line in train_output_fin:
    y[i] = ctable.encode(line, 50)
    i+=1

i = 0
for line in test_input:
    x_test[i] = ctable.encode(line, 50)
    i+=1

i = 0

for line in test_output:
    y_test[i] = ctable.encode(line, 50)
    i+=1

#split for validation set

#split_at = len(x) - len(x) // 10
#(x_train, x_val) = x[:split_at], x[split_at:]
#(y_train, y_val) = y[:split_at], y[split_at:]
x_train = x
y_train = y
x_val = x_test
y_val = y_test
print('Training Data:')
print(x_train.shape)
print(y_train.shape)

print('Validation Data:')
print(x_val.shape)
print(y_val.shape)





#model_path = "model.h5"
#if path.exists(model_path):
#    print("Loading weights and model")

#    model = tf.keras.models.load_model('model.h5')
#    print(model.get_weights())
#    model.summary()

#else:
#    print("New model")
    #Build the model

model = make_or_restore_model()

callbacks = [
    # This callback saves a SavedModel every 1000 batches.
    # We include the training loss in the folder name.
    tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_dir + '/ckpt-loss={loss:.2f}',
        save_freq=2000)
]

#model.fit(x_train, y_train, batch_size=BATCH_SIZE,epochs=1)

#y_pred = model.predict(x_test)
#print(y_pred)



#for iteration in range(1, 10):
model.fit(x_train, y_train,
              epochs=100000,
              batch_size = 1024,
              validation_data=(x_val, y_val),
              callbacks=callbacks)
#model.save('model.h5')


sum1 = 0
for i in range(10):
    ind = np.random.randint(0, len(x_val))
    rowx, rowy = x_val[np.array([i])], y_val[np.array([i])]
    preds = model.predict(rowx, verbose=0)
    q = np.zeros(50)
    correct = np.zeros(50)
    guess = np.zeros(50)
    n = 0
    for x in rowx:
        for y in x:
            l = 0
            for z in y:
                if z == True:
                    q[n] = l
                    n+=1
                l+=1

    n = 0
    for x in rowy:
        for y in x:
            l = 0
            for z in y:
                if z == True:
                    correct[n] = l
                    n+=1
                l+=1
    print(rowx.shape)
    print(preds.shape)

    largest = np.zeros(50)
    for x in preds:
        l = 0
        for y in x:
            for z in y:
                if z >= largest[l]:
                    largest[l] = z
            l+=1
    print(largest)
    for x in preds:
        l = 0
        for y in x:
            k = 0
            for z in y:
                if z == largest[l]:
                    guess[l] = k
                k+=1
            l+=1
    print(guess)


    #q = ctable.decode(rowx[0])
    #correct = ctable.decode(rowy[0])
    #guess1 = ctable.decode(preds[0], calc_argmax=False)
    print('Input : ', q)
    print('Expected : ', correct)
    print('Predicted', guess)
    sum = 0
    j = 0
    for j in range(50):
        if correct[j] == guess[j]:
            sum = sum+1
    val_error = (sum/50)*100
    print('Val_error : ',val_error)
    sum1 = sum1 + val_error
print("Average accuracy = " + str(sum1/50))

    # Select 10 samples from the validation set at random so we can visualize
    # errors.
#for i in range(10):
#    ind = np.random.randint(0, len(x_val))
#    rowx, rowy = x_val[np.array([ind])], y_val[np.array([ind])]
#    preds = model.predict_classes(rowx, verbose=0)
#    q = ctable.decode(rowx[0])
#    correct = ctable.decode(rowy[0])
#    guess = ctable.decode(preds[0], calc_argmax=False)
#    print('Input : ', q[::-1])
#    print('Expected : ', correct)
#    #if correct == guess:
        #print(colors.ok + '☑' + colors.close, end=' ')
    #else:
        #print(colors.fail + '☒' + colors.close, end=' ')
#    print('Predicted', guess)
#    sum = 0
#    i = 0
#    for i in range(50):
#        if correct[i] == guess[i]:
#            sum = sum+1
#    val_error = (sum/50)*100
#    print('Val_error : ',val_error)


    #model.save_weights('model.h5')
