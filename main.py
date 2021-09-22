from argparse import ArgumentParser

import numpy
from keras.models import Sequential
from keras.layers import Dense


def main():
    args = parse_args()
    inputs, desired_output = load_dataset(args.dataset, args.data_delimiter)
    model = train_model(inputs, desired_output)
    evaluate(model, inputs, desired_output)
    print_predictions(model, inputs, desired_output)
    model.save(args.model_out)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset', help='csv file with inputs and desired outputs')
    parser.add_argument('--data_delimiter', help='delimiter in dataset csv file')
    parser.add_argument('--model_out', help='model destination file or directory')
    args = parser.parse_args()
    return args


def load_dataset(dataset_file, delimiter):
    dataset1 = numpy.loadtxt(dataset_file, delimiter=delimiter)
    inputs = numpy.array(dataset1[:, 0:2], 'uint8')
    desired_output = numpy.array(dataset1[:, 2:3], 'uint8')
    return inputs, desired_output


def train_model(inputs, desired_output):
    model = Sequential()
    model.add(Dense(2, activation='elu', input_dim=2))
    model.add(Dense(4, activation='elu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(inputs, desired_output, epochs=1000, verbose=0)
    return model


def evaluate(model, inputs, desired_output):
    _, accuracy = model.evaluate(inputs, desired_output)
    print(f'Accuracy: {accuracy * 100}')


def print_predictions(model, inputs, desired_output):
    predictions = model.predict(inputs)
    for i in range(len(inputs)):
        print(f'{inputs[i].tolist()} => {round(predictions[i][0])} (expected: {desired_output[i]})')


if __name__ == '__main__':
    main()
