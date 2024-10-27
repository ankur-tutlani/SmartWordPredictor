## Description
This repository contains three different approaches for intelligent word prediction and completion. Originally designed for the Hangman game, these algorithms have broad applications in various language processing tasks.

## Input Data Requirements
The input data for these approaches should be a dictionary of alphabetic keywords in the English language.

## Approaches

## Approach 1: LSTM Based Model
1. Data Segmentation: Divides data into segments with varying percentages of masked characters.
2. Multiple Variations: Includes different masking variations for robust training.
3. Masking and Padding: Utilizes '0' for masking and '#' for padding.
4. Model: Employs a Bidirectional LSTM model.
5. Prediction: Defined a predict function to handle masked words.

## Approach 2: LSTM Based Model (Masked Characters in Output Sequences)
1. Output Sequences: Focuses solely on masked characters in the output sequences.
2. Custom Loss and Accuracy: Implements custom functions excluding padding characters.
3. Prediction Accuracy: Measures success based on the presence of predicted characters in true labels.
4. Model: Uses a Bidirectional LSTM model for letter prediction.

## Approach 3: Backward and Forward Letters Distribution
1. Length-Based Segmentation: Segregates words by length.
2. Frequency Calculation: Computes letter frequency in both directions.
3. Letter Recommendation: Suggests letters based on occurrence percentages.
4. Handling Multiple Masked Characters: Uses highest normalized count for recommendations.

## Applications
- Language Learning Tools: Enhance vocabulary and grammar skills.
- Word Prediction Systems: Improve text editors and predictive text systems.
- Crossword Puzzle Solvers: Assist in solving crosswords.
- Spelling Correction Tools: Refine spelling and grammar checking software.
- AI-Powered Assistants: Enhance virtual assistants for better query understanding.

## Features
- Approach 1: Innovative LSTM-based masking and padding techniques.
- Approach 2: Enhanced LSTM model with custom loss and accuracy functions.
- Approach 3: Statistical letter distribution for precise recommendations.

## License
This project is licensed under the MIT License.
