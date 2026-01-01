import pickle
import numpy as np
import os
from music21 import instrument, note, stream, chord
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, BatchNormalization, Activation
from keras.utils import to_categorical

SEQ_LENGTH = 100


def train():
    with open('data/notes', 'rb') as f:
        notes = pickle.load(f)

    pitchnames = sorted(set(notes))
    n_vocab = len(pitchnames)

    network_input, normalized_input, network_output = prepare_sequences(notes, pitchnames, n_vocab)
    model = create_network(normalized_input, n_vocab)

    model.fit(normalized_input, network_output, epochs=200, batch_size=64)
    model.save_weights('weights.hdf5')


def generate():
    with open('data/notes', 'rb') as f:
        notes = pickle.load(f)

    pitchnames = sorted(set(notes))
    n_vocab = len(pitchnames)

    network_input, normalized_input, _ = prepare_sequences(notes, pitchnames, n_vocab)
    model = create_network(normalized_input, n_vocab)

    prediction_output = generate_notes(model, network_input, pitchnames, n_vocab)
    create_midi(prediction_output)


def prepare_sequences(notes, pitchnames, n_vocab):
    note_to_int = {note: i for i, note in enumerate(pitchnames)}

    network_input = []
    network_output = []
    for i in range(len(notes) - SEQ_LENGTH):
        seq_in = notes[i:i + SEQ_LENGTH]
        seq_out = notes[i + SEQ_LENGTH]
        network_input.append([note_to_int[n] for n in seq_in])
        network_output.append(note_to_int[seq_out])

    n_patterns = len(network_input)
    network_input = np.reshape(network_input, (n_patterns, SEQ_LENGTH, 1))
    normalized_input = network_input / float(n_vocab)
    network_output = to_categorical(network_output)

    return network_input.tolist(), normalized_input, network_output


def create_network(network_input, n_vocab):
    model = Sequential()

    model.add(LSTM(512, input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=True))
    model.add(LSTM(512, return_sequences=True))
    model.add(LSTM(512))

    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    if os.path.exists('weights.hdf5'):
        model.load_weights('weights.hdf5')

    return model


def generate_notes(model, network_input, pitchnames, n_vocab):
    start = np.random.randint(0, len(network_input) - 1)

    int_to_note = {i: note for i, note in enumerate(pitchnames)}
    pattern = network_input[start]

    prediction_output = []

    for _ in range(500):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)
        index = np.argmax(prediction)

        result = int_to_note[index]
        prediction_output.append(result)

        pattern.append(index)
        pattern = pattern[1:]

    return prediction_output


def create_midi(prediction_output):
    output_notes = []
    offset = 0

    for pattern in prediction_output:
        if '.' in pattern or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes_list = []

            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes_list.append(new_note)

            new_chord = chord.Chord(notes_list)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp='output.mid')


if __name__ == '__main__':
    if not os.path.exists('weights.hdf5'):
        print("Training the model...")
        train()
    generate()
