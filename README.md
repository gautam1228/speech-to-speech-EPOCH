# Speech-to-Speech Translation Model (English to Hindi)

## Project Overview

This project was developed as part of the EPOCH college hackathon, aimed at addressing Problem Statement 3. Our goal was to create a speech-to-speech translation model that translates English speech to Hindi speech. 

### Problem Statement:

Problem Statement 3: Develop a speech-to-speech translation model for translating English speech to Hindi speech without using an intermediate text representation.

## Project Description

Our solution leverages deep learning techniques to perform direct translation from English speech to Hindi speech, eliminating the need for text representation in between. This allows for more efficient and seamless translation in many scenarios.

### Key Features:

- Speech-to-speech translation from English to Hindi speech.
- No intermediate text representation, ensuring faster translation.
- Developed using deep learning techniques.
- Utilizes cutting-edge speech processing algorithms.
- Custom dataset created for training the model.

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/yourusername/speech-to-speech-translation.git
   ```

2. Install the required dependencies:

   ```
   pip install -r requirements.txt
   ```

## Dataset Generation

We created a custom dataset for training the speech-to-speech translation model. Initially, we utilized the LibriSpeech dataset (https://paperswithcode.com/dataset/librispeech), which contains English speech-to-text data. We employed the Google Translate API to convert this English text into Hindi text. Subsequently, the Hindi text was transformed into Hindi audio using gTTS (Google Text-to-Speech), albeit this task was computationally intensive.

## Challenges

One of the major challenges we encountered was the constraint of not converting speech to text in the lateral space. This constraint added complexity to the development process as it required designing a model capable of directly translating speech signals without relying on intermediate text representation.

## Future Improvements

While our current model demonstrates promising results, there are several areas for future improvement:

- Enhanced model performance through continued training with larger datasets.
- Integration of more advanced speech processing techniques to improve translation accuracy.
- Expansion to support translation between additional languages.

## Contributors

- Mehul Pahuja (@mehulhere)
- Gautam Singh (@gautam1228)
- Aditya Aggarwal (@aytida1165)

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

We would like to express our gratitude to the organizers of the EPOCH hackathon for providing us with the opportunity to work on this challenging problem statement.
