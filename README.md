# MISTI_Brazil_dPASP_research
Explanation of folders/files in root directory (plp_files_and_implementation and pure_nn_approach are directly useful, and the other two are more for reference):
- plp_files_and_implementation: all the experiments that I have created during this internship
- pure_nn_approach: all neural networks that correspond to the tasks that the plp files.
- useful_python_code: general code to help make neural networks.
- pasp_examples: examples from dpasp tutorial that helped me write my plp files.
- model.pth: example of how to save models to use in pretrained____.plp files

## Types of Experiments
Each task has several variations regarding the task, or the input to the algorithm. Some variations are just for the pure neural network approach, while other variations are just for the dPASP approach.

### For dPASP 

#### Three Digits 
- There are three digits, and the task is to guess the last digit. The first two digits should.

- three_digits.plp - three images of digits are fed in, and nn guesses a number between 0-9. The pasp code guesses the last digit.

- three_digit_implicit.plp - same as above except for test time. The training is done by giving the algorithm the ground truth of the last digit, but the test is checking the accuracy of identifying the first digit.

- three_digits_biased.plp - same as three_digits.plp except for the distribution of training and test set. For training, the digits sum to between 0 and 10, and the digits sum to 8-18 for test.

- pretrained_three_digits.plp - exactly the same as three_digits.plp except instead of training through dpasp, downloads a pretrained model (mnist_net.pth) to identify mnist digits. You can train this yourself separately, or download a pretrained model from the internet.

#### Guess Equal Row Sum or Equal Column Sum

#### Five Digits
There are three digits, and the task is to guess the last digit. The first two digits should sum (or differ) to the same number as the second two. The difficulty is that this takes too long for exact inference.

- five_digit.plp - five images of digits are fed in, and nn guesses a number between 0-9. The pasp code guesses the last digit.

- guess_operation.plp - same as five_digit.plp except the logic part also has to guess whether the operation that is equal between the pairs is a multiplication, subtraction or addition (could also try adding modulo as a possible operation). 

- mod3_guess_op.plp - The variation here is that the pair have equal sums or differences modulo 3. The logic program also has to guess whether it is a sum or a difference.

- mod3_guess_op_biased.plp - Same as mod3_guess_op, but the training set is only "sum", while the test set contains both "sum" and "dif" cases.

- pretrained_five_digits.plp - pretrained version of five_digit.plp just as in pretrained_three_digits.plp

### For Neural Nets
- for more details, read the mns_dataloaders.py and mns_models.py files.
#### Three Digits
- three_channels - three channels, each an images of 28*28 are fed-in. The nn outputs the guess of the last.
- two_concat - The model has two channels. The first channel is dim 28x64 which shows one digit on the top and one on the bottom concatenated with 8 line gap in between. The second image is a digit on the top and a blank on the bottom. The task is to guess the last digit. (what goes in the bottom)
- **two_concat_biased - same as two_concat except for the train and test data. The train data are numbers that sum to between (0,10) and the test are (8,18).**

#### Five Digits
- five_channels - same as three_channels but for five digits case.
- five_concat - five digits are concatenated to form 2x140 image. The neural output is a guess of the last digit.
- **guess_opp - stil implementing, but should correspond to mod3_guess_op.plp. We want to make the train and test data different, where for training it is always the sum operator, but for test, the operator is sum and dif with equal probability.**
#### Guess Equal Row Sum or Equal Column Sum
- four_gathered - four images of dimension 28x28 are concatenated to make an image of dim 56x56. Either each row sums up to an equal number, or each column sums up to an equal number. The algorithm outputs a length two vector with confidence of "row" vs "column".

## File types
### For Neural Net approach
All neural net algorithms will be run through the execution.ipynb file. This reads in the mns_dataloaders.py and mns_models.py files, containing the dataloaders and neural net models respectively for each specific task. Each task has a specific dataloader (e.g. guess_op), and the name of the corresponding neural net is obtained by adding "_Net" at the end (e.g. guess_op_Net).

### For dPASP approach.
Each pasp algorithm has its own .plp file. These can all be run through evaluate_pasp.py by changing the file names fed into pasp.parse(".."). The file name passed into read_csv should match the file the corresponding pasp code writes into using the note_label_for_test function.
