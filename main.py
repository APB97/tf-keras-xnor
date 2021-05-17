import numpy
from keras.models import Sequential
from keras.layers import Dense

if __name__ == '__main__':
    dataset1 = numpy.loadtxt('xnor.csv', delimiter=',')
    A = numpy.array(dataset1[:, 0:2], 'uint8')
    B = numpy.array(dataset1[:, 2:3], 'uint8')

    print(A)
    print(B)

    model = Sequential()
    model.add(Dense(2, activation='elu', input_dim=2))
    model.add(Dense(4, activation='elu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(A, B, epochs=1000, verbose=0)

    _, accuracy = model.evaluate(A, B)
    print(f'Accuracy: {accuracy*100}')

    predictions = model.predict_classes(A)
    for i in range(len(A)):
        print(f'{A[i].tolist()} => {predictions[i]} (expected: {B[i]})')

    model.save('XNOR.model')
