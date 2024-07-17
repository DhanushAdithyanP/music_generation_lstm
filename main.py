import glob
from src.data_preprocessing import download_and_extract_maestro, count_midi_files
from utils import generate_audio, print_instruments_info, midinotes, prepare_dataset, predict_next_note, notes_to_midi
from src.model_training import build_model, train_model, evaluate_model
import pretty_midi
import numpy as np
import pandas as pd
from google.colab import files

def main():
    print("Starting data preparation...")
    download_and_extract_maestro()
    num_files = count_midi_files()
    print(f"Data preparation completed. Number of MIDI files: {num_files}")

    data_dir = 'data/maestro-v3.0.0'
    midi_files = glob.glob(data_dir + '/**/*.mid*', recursive=True)  # Rename files to midi_files
    
    if len(midi_files) == 0:
        print("No MIDI files found in the dataset directory.")
        return
    
    first_file = midi_files[0]
    pm = pretty_midi.PrettyMIDI(first_file)
    audio = generate_audio(pm)
    
    # Print instrument information
    print_instruments_info(pm)

    # Extract and print MIDI note information
    raw_notes = midinotes(first_file)
    get_note_names = np.vectorize(pretty_midi.note_number_to_name)
    note_names = get_note_names(raw_notes['pitch'])

    # Print first 10 note names and their details
    print("First 10 note names and details:")
    for i, (pitch, note_name, start, end, duration) in enumerate(
        zip(raw_notes['pitch'][:10], note_names[:10], raw_notes['start'][:10], raw_notes['end'][:10], raw_notes['duration'][:10])
    ):
        print(f"{i}: pitch={pitch}, note_name={note_name}, start={start:.4f}, end={end:.4f}, duration={duration:.4f}")

    # Prepare dataset for training
    seq_length = 25
    batch_size = 64
    train_ds = prepare_dataset(midi_files[:5], seq_length, batch_size)  # Pass midi_files here

    # Build and train the model
    input_shape = (seq_length, 3)
    model = build_model(input_shape)
    model.summary()

    # Evaluate the model before training
    losses = evaluate_model(model, train_ds)
    print(f"Initial losses: {losses}")

    # Train the model
    history = train_model(model, train_ds)

    # Evaluate the model after training
    final_losses = evaluate_model(model, train_ds)
    print(f"Final losses: {final_losses}")

    # Generate MIDI from generated_notes DataFrame
    temperature = 2.0
    num_predictions = 120
    
    # Define key_order based on your columns
    key_order = ['pitch', 'step', 'duration']
    
    # Define vocab_size based on your dataset
    vocab_size = 128  # Example: Number of unique pitch values in your dataset
    
    sample_notes = np.stack([raw_notes[key] for key in key_order], axis=1)
    input_notes = sample_notes[:seq_length] / np.array([vocab_size, 1, 1])
    
    generated_notes = []
    prev_start = 0
    for _ in range(num_predictions):
        pitch, step, duration = predict_next_note(input_notes, model, temperature)  # Implement predict_next_note function
        start = prev_start + step
        end = start + duration
        input_note = (pitch, step, duration)
        generated_notes.append((*input_note, start, end))
        input_notes = np.delete(input_notes, 0, axis=0)
        input_notes = np.append(input_notes, np.expand_dims(input_note, 0), axis=0)
        prev_start = start
    
    generated_notes = pd.DataFrame(generated_notes, columns=['pitch', 'step', 'duration', 'start', 'end'])

    # Convert generated_notes DataFrame to MIDI
    out_file = 'generated_music.mid'
    instrument_name = 'Acoustic Grand Piano'
    notes_to_midi(generated_notes, out_file, instrument_name)


if __name__ == "__main__":
    main()