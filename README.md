# Heuristic interpretation as rational inference: A computational model of the N400 and P600 in language processing
README

This directory contains four files:

  noisy_channel.py -- the code for calculating n400 and p600 elicited by a given stimuli
  stats_analysis.R -- the code of linear mixed effect models to run statistical tests for simulated 		ERP effect across eight experiments
  test_input.csv -- sample input file for noisy_channel.py
  test_output.csv -- sample output file generated by noisy_channel.py

%%%%%%%%%%%%%%%%%%%


main.py takes two arguments:

1) The name (and path) of an input .csv file. The input file should have at least four columns: Item Condition, Literal, Alternative. See 'test_input.csv' for detailed format
2) The name (and path) of an output .csv file. See 'test_output.csv' for more details.

Install required packages:
  torch
  pytorch_pretrained_bert
  sentence_transformers
  fastDamerauLevenshtein
  
Run the code as follows:
    python3 noisy_channel.py -i <input_filename> -o <output_filename>

%%%%%%%%%%%%%%%%%%%%
