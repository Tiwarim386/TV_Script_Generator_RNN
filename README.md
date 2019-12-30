# TV_Script_Generator_RNN
Using RNN's on a Seinfeld dataset of scripts from 9 seasons to generate own Seinfeld TV scripts. The Neural Network will generate a new, "fake" TV script based on patterns it recognizes in this training data.

## Project Information

## Results

I have trained the model over 5 epochs to give me a loss of about 3.2. You can see that there are multiple characters that say (somewhat) complete sentences, but it isn't perfect. It takes quite a while to get good results, and often, you'll have to use a smaller vocabulary (and discard uncommon words), or get more data. The Seinfeld dataset is about 3.4 MB, which is big enough for our purposes; for script generation you'll want more than 1 MB of text, generally.


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

### Sample Script Generated after Training

jerry: i was in my apartment, and i was just sitting there, but, uh, i just can't get out of here. i mean i can't take this anymore.

george:(to jerry) oh, yeah.

jerry:(on phone) yeah?

george: yeah!

jerry: yeah, you know what i want you to do.

george: what?

jerry: i don't know why you want.

jerry: well, i don't know.(he hangs up phone)

jerry: i'm not gonna go to the hospital and get it.(kramer enters with the remote.)

kramer: hey, i got the machine.

jerry: oh, i know. it's a good idea.

jerry: oh.(to george) you got a big salad, huh?

kramer: well, i don't have to talk to her!

kramer:(looking at the menu) what?

george: i don't know, i don't know how to get back to the hospital.

george: i know. it's a very unusual idea.

george: i can't believe it.

elaine: oh, come on, take it off, come up with it.

george: i don't know. i don't think we should do it. you know, it's the only way i could get back.

jerry:(to elaine) hey... i know what i'm gonna do with it, but you gotta be honest.

jerry: you're gonna have a good time.

elaine: well, i don't want to know if i'm not going to see the movie.

jerry: what is this?

kramer: i think i should take a cab.

elaine: i can't take it off.

jerry: i don't have any idea.(she leaves)

jerry: oh, yeah.(to jerry) what are we talking about?

kramer: oh, yeah.(to jerry) you know, the only thing is a scam.

### Look at complete scripts Generated [Here](https://github.com/Tiwarim386/TV_Script_Generator_RNN/tree/master/Generated%20Scripts)


### Question:How did I decide on my model hyperparameters?

Answer: I tried running for 15 epochs. but it was taking a lot of time to train.So, I reduced the no. of epochs to 10 but my GPU workspace used to disconect/sleep (the limit is 30 mins of inactivity). So i used the workspace_utilities.py file provided. But since it was still taking a lot of time to train I finally reduced no. of epochs to 5. Which was good enough for a Loss of less than "3.5".

I arbitrarily chose 150 as my sequence length initially but loss started with huge number. Then I thought if in a plot of TV Script a normal conversation would be minimum 5 words and maximum 10 words (maybe 20-25 if not a monologue? ) .So, I chose 10 as my sequence length this time. Although, model's training speed was slow but the loss started with very less number(5.11) and I reached objective of less than 3.5 ( EVEN BEFORE 5th Epoch!! ) .

About the hidden dim I choose 512. Its a standard practice, we need to choose values in powers of 2 (i.e. 64,128,256,512) etc. More the value better the training.

About the n_layers, as we are using the LSTM cells its standard practice to use between 1-3 layers as we go more deeper there will high computational complexity but this is reverse in case of CNN. I used 2 as n_layers. Moreover I wanted to Use Dropout & Because of Dropout Constraints n_layers must be >= 2

As expected, We get better results with larger hidden and n_layer dimensions, but larger models take a longer time to train.



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


