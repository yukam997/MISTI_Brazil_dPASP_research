# MISTI_Brazil_dPASP_research

## Types of Experiments
Each task has several variations regarding the task, or the input to the algorithm. Some variations are just for the pure neural network approach, while other variations are just for the dPASP approach.
### Three Digits
There are three digits, and the task is to guess the last digit. The first two digits should.

1.

### Guess Equal Row Sum or Equal Column Sum

### Five Digits

## File types
### For Neural Net approach
All neural net algorithms will be run through the execution.ipynb file. This reads in the mns_dataloaders.py and mns_models.py files, containing the dataloaders and neural net models respectively for each specific task. Each task has a specific dataloader (e.g. guess_op), and the name of the corresponding neural net is obtained by adding "_Net" at the end (e.g. guess_op_Net).

### For dPASP approach.
Each pasp algorithm has its own .plp file. These can all be run through evaluate_pasp.py by changing the file names fed into pasp.parse(".."). The file name passed into read_csv should match the file the corresponding pasp code writes into using the note_label_for_test function.
