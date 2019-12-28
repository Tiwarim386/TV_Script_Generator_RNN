# TV_Script_Generator_RNN
Using RNN's on a Seinfeld dataset of scripts from 9 seasons to generate own Seinfeld TV scripts. The Neural Network will generate a new, "fake" TV script based on patterns it recognizes in this training data.

## Project Information

### Contents

- Get the Data
- Explore the Data
- Implement Pre-processing Functions
	- Lookup Table
	- Tokenize Punctuation
- Pre-process all the data and save it
- Check Access to GPU
- Input
	- Batching
	- Test your dataloader
	- Sizes
	- Values
- Build the Neural Network
	- Define forward and backpropagation
- Neural Network Training
	- Train Loop
	- Hyperparameters
	- Train 
- Generate TV Script
	- Generate text
	- Generate a new script

### Model
| Layer | Input Dimension | Output Dimension |
| ----- | --------------- | ---------- |
|Embedding|Vocab Size| 463 |
|LSTM|463|512|
|Fully Connected Layer|512|Vocab Size|

### Hyperparameters
|Data Parameter|Value|
|--------------|------|
|sequence_length| 10|
|batch_size| 256 |

|Training Parameter|Value|
|------------------|-----|
|num_epochs|5|
|learning_rate|0.001|
|embedding_dim|463|
|hidden_dim|512|
|n_layers(Number of RNN Layers)|2|
