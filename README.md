# Piano MIDI Generation Using LSTM

Generate MIDI files of piano music using LSTM (Long Short-Term Memory) neural networks trained on the Maestro v3.0.0 dataset.

## Overview
This project leverages deep learning techniques to create original piano music compositions in MIDI format. The LSTM model, a type of recurrent neural network known for its ability to learn patterns over sequences, is trained on the Maestro v3.0.0 dataset. This dataset consists of meticulously annotated piano performances, providing a rich source of musical data for training.

The model is designed to predict the next note in a sequence given a sequence of notes, enabling it to generate coherent and expressive musical sequences.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/DhanushAdithyanP/music_generation_lstm.git
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
Make sure you have Python 3.x and pip installed on your system. The `requirements.txt` file includes all necessary Python packages required to run the project.

3. **Download the Maestro v3.0.0 dataset:**
Before running the project, you need to download the Maestro v3.0.0 dataset. You can do this by running the data preprocessing script provided (`data_preprocessing.py`) or manually download it from [here](https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip) and extract it into the `data/maestro-v3.0.0/` directory.

4. **Run the project:**
   ```bash
   python main.py
   ```
   
## Future Work
- **Genre-specific Models:** Develop models tailored to specific musical genres, enhancing the diversity and style of generated compositions.
- **Multi-instrument Synthesis:** Extend the model to generate music for multiple instruments, exploring synchronization and harmonization between instruments.

## Contributing
Contributions are welcome! Fork the repository, make improvements, and submit pull requests. Issues and feature requests can be discussed via GitHub issues.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgments
- Maestro v3.0.0 dataset by Google Magenta.
- TensorFlow and PrettyMIDI libraries for deep learning and MIDI manipulation.

## Contact
For questions, feedback, or collaborations, contact [P Dhanush Adithyan](dhanushadithyanp@gmail.com).
   
