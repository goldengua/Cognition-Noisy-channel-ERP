<h1> README </h1>

<p> This repository includes sample test data and codes of paper <i> Heuristic interpretation as rational inference: A computational model of the N400 and P600 in language processing </i>. </p>

<h4> List of Files </h4>

  <li> noisy_channel.py -- the code for calculating n400 and p600 elicited by a given stimuli </li>
  
  <li> stats_analysis.R -- the code of linear mixed effect models to run statistical tests for simulated ERP effect across nine experiments </li>
  
  <li> test_input.csv -- sample input file for noisy_channel.py </li>
  
  <li> test_output.csv -- sample output file generated by noisy_channel.py </li>


<h4> How to run <i> noisy_channel.py </i> </h4>
<p><i> noisy_channel.py </i> takes two arguments: </p>

<li> <input_filename>: The name (and path) of an input .csv file. The input file should have at least four columns: Item, Condition, Literal, Alternative. See 'test_input.csv' for detailed format </li>

<li> <output_filename>: The name (and path) of an output .csv file. See 'test_output.csv' for more details. </li>

<p> Install required packages:</p>
  <li> torch </li> 
  <li> pytorch_pretrained_bert </li>
  <li> sentence_transformers </li>
  <li> fastDamerauLevenshtein </li>
  
<p> Run the code as follows: </p>
    
    python3 noisy_channel.py -i <input_filename> -o <output_filename>

%%%%%%%%%%%%%%%%%%%%
