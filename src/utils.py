import pretty_midi
import pandas as pd
import numpy as np
import collections
import tensorflow as tf

def generate_audio(pm: pretty_midi.PrettyMIDI, seconds=30):
    SAMPLING_RATE = 16000
    waveform = pm.synthesize(fs=SAMPLING_RATE)
    waveform_short = waveform[:seconds * SAMPLING_RATE]
    return waveform_short, SAMPLING_RATE

def print_instruments_info(pm: pretty_midi.PrettyMIDI):
    print('Number of instruments:', len(pm.instruments))
    if len(pm.instruments) > 0:
        inst = pm.instruments[0]
        inst_name = pretty_midi.program_to_instrument_name(inst.program)
        print('Instrument name:', inst_name)

def midinotes(midi_file: str) -> pd.DataFrame:
    pm = pretty_midi.PrettyMIDI(midi_file)
    inst = pm.instruments[0]
    notes = collections.defaultdict(list)
    sorted_notes = sorted(inst.notes, key=lambda note: note.start)
    prev_start = sorted_notes[0].start

    for note in sorted_notes:
        start = note.start
        end = note.end
        notes['pitch'].append(note.pitch)
        notes['start'].append(start)
        notes['end'].append(end)
        notes['step'].append(start - prev_start)
        notes['duration'].append(end - start)
        prev_start = start

    return pd.DataFrame(notes)

def create_sequences(dataset: tf.data.Dataset, seq_length: int, vocab_size=128) -> tf.data.Dataset:
    seq_length = seq_length + 1
    windows = dataset.window(seq_length, shift=1, stride=1, drop_remainder=True)
    flatten = lambda x: x.batch(seq_length, drop_remainder=True)
    sequences = windows.flat_map(flatten)

    def scale_pitch(x):
        x = x / [vocab_size, 1.0, 1.0]  
        return x

    def split_labels(sequences):
        inputs = sequences[:-1]
        labels_dense = sequences[-1]
        labels = {key: labels_dense[i] for i, key in enumerate(['pitch', 'step', 'duration'])}
        return scale_pitch(inputs), labels

    return sequences.map(split_labels, num_parallel_calls=tf.data.AUTOTUNE)

def prepare_dataset(files: list, seq_length: int, batch_size: int, vocab_size=128) -> tf.data.Dataset:
    all_notes = []
    for f in files:
        notes = midinotes(f) 
        all_notes.append(notes)

    all_notes = pd.concat(all_notes)
    train_notes = np.stack([all_notes[key].values for key in ['pitch', 'step', 'duration']], axis=1)
    notes_ds = tf.data.Dataset.from_tensor_slices(train_notes)

    seq_ds = create_sequences(notes_ds, seq_length, vocab_size)

    buffer_size = len(all_notes) - seq_length
    train_ds = (seq_ds
                .shuffle(buffer_size)
                .batch(batch_size, drop_remainder=True)
                .cache()
                .prefetch(tf.data.experimental.AUTOTUNE))

    return train_ds


def predict_next_note(
    notes: np.ndarray,
    keras_model: tf.keras.Model,
    temperature: float = 1.0
) -> tuple:
    assert temperature > 0

    inputs = tf.expand_dims(notes, 0)

    predictions = keras_model.predict(inputs)
    pitch_logits = predictions['pitch']
    step = predictions['step']
    duration = predictions['duration']

    pitch_logits /= temperature
    pitch = tf.random.categorical(pitch_logits, num_samples=1)
    pitch = tf.squeeze(pitch, axis=-1)
    duration = tf.squeeze(duration, axis=-1)
    step = tf.squeeze(step, axis=-1)

    step = tf.maximum(0, step)
    duration = tf.maximum(0, duration)

    return int(pitch), float(step), float(duration)

def notes_to_midi(notes: pd.DataFrame, out_file: str, instrument_name: str, velocity: int = 100):
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program(instrument_name))

    prev_start = 0
    for i, note in notes.iterrows():
        start = float(prev_start + note['step'])
        end = float(start + note['duration'])
        midi_note = pretty_midi.Note(velocity=velocity, pitch=int(note['pitch']), start=start, end=end)
        instrument.notes.append(midi_note)
        prev_start = start

    pm.instruments.append(instrument)
    pm.write(out_file)
    return pm