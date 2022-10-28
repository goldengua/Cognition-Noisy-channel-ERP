###install the following packages:
####!pip install pytorch_pretrained_bert
####!pip install sentence_transformers
####!pip install fastDamerauLevenshtein
import torch
import sys, getopt
import math 
from statistics import *
import matplotlib.pyplot as plt
import numpy as np
from math import log
import torch.nn.functional as F
import pandas as pd
import argparse
from pytorch_pretrained_bert import OpenAIGPTTokenizer, OpenAIGPTModel, OpenAIGPTLMHeadModel
from sentence_transformers import SentenceTransformer, util
import nltk
from fastDamerauLevenshtein import damerauLevenshtein

#import gpt 
model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
model.eval()
#import distil bert
model_distil = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
model_distil.eval()

#parse sentence into context and target (argument: string)
def preprocess(sentence):

  target = sentence.split()[-1].replace('.','')
  context = ' '.join(sentence.split()[:-1])

  return context,target

#calculate perplexity of sentence (argument: string)
def score(sentence):

    tokenize_input = tokenizer.tokenize(sentence)
    tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
    loss = model(tensor_input, lm_labels=tensor_input)

    return math.exp(loss)

#calculate perplexity for a list of sentences (argument: list)
def perplexity(lit,alt):

    per_lit = [score(item)/len(item.split()) for item in lit]
    per_alt = [score(item)/len(item.split()) for item in alt]

    return per_lit,per_alt

#calculate edit distance between two strings and return 1/edit_distance as likelihood (argument: two strings)
def editDistDP(str1,str2):
  return damerauLevenshtein(str1, str2, similarity=True, deleteWeight = 1, insertWeight=2, replaceWeight = 2, swapWeight=1) 


#calculate conditional probability with GPT (argument: string)
def prediction(sentence):
    text,target = preprocess(sentence)
    tokenized_text = tokenizer.tokenize(text)
    tokenized_target = tokenizer.tokenize(target)
    # Convert token to vocabulary indices
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    target_index =  tokenizer.convert_tokens_to_ids(tokenized_target)
    
    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    with torch.no_grad():
        predictions = model(tokens_tensor)
        
    predicted_score = predictions[0, -1, :][target_index]
    
    return predicted_score[0].item()

#calculate conditional probability (normalized) (argument: string)
def normalized_prediction(sentence):
    text,target = preprocess(sentence)
    tokenized_text = tokenizer.tokenize(text)
    tokenized_target = tokenizer.tokenize(target)
    # Convert token to vocabulary indices
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    target_index =  tokenizer.convert_tokens_to_ids(tokenized_target)
    
    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    with torch.no_grad():
        predictions = model(tokens_tensor)
    predicted_score = F.softmax(F.log_softmax(predictions[0, -1, :], dim=0), dim=0)[target_index]  
    return predicted_score[0].item()

#calculate consine similarity between sentence embeddings with distilbert (argument: two lists of strings)
def similarity(sentences1,sentences2):

    #Compute embedding for both lists
    embeddings1 = model_distil.encode(sentences1, convert_to_tensor=True)
    embeddings2 = model_distil.encode(sentences2, convert_to_tensor=True)

    #Compute cosine-similarits
    cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
    return cosine_scores

#calculate conditional probability between context and target (argument: two lists of strings)
def probing_conditional_probability(lit,alt):
 
  pred_lit = [normalized_prediction(lit[i]) for i in range(len(lit))]
  pred_alt = [normalized_prediction(alt[i]) for i in range(len(alt))]

  return pred_lit, pred_alt

#calculate conditional probability between two lists of sentences (argument: two lists of strings)
def probing_similarity(lit,alt):
    cosine_scores = similarity(lit,alt)
    return [cosine_scores[i][i].item() for i in range(len(lit))]

def probing_distance(lit,alt):
    return [editDistDP(lit[i],alt[i]) for i in range(len(lit))]

#calculate posterior probability and return the probability for 'not corrected' and 'corrected' (argument: two lists of strings)
def posterior(lit,alt):
    distance = [editDistDP(lit[i],alt[i]) for i in range(len(lit))]
    lit_ori = [1/score(lit[i])*(1/(distance[i]+1)) for i in range(len(lit))]
    lit_cor = [1/score(alt[i])*(distance[i]/(distance[i]+1)) for i in range(len(lit))]
    return lit_ori,lit_cor

#calculate N400 (argument: output from posterior (lit, alt), and two lists of sentences)
def cal_n400(lit_ori,lit_cor,lit,alt):

    n400 = [prediction(alt[i]) if lit_ori[i] < lit_cor[i] else prediction(lit[i]) for i in range(len(lit))]

    return n400

#calculate P600 (argument: output from posterior (lit, alt), and two lists of sentences)
def cal_p600(lit_ori,lit_cor,lit,alt):

    cosine_scores = similarity(lit,alt)
    p600 = [cosine_scores[i][i].item() if lit_ori[i] < lit_cor[i] else 1 for i in range(len(lit)) ]

    return p600
#how to use: 'noisy_channel.py -i <input_filename> -o <output_filename>'
def main(argv):
    opts, args = getopt.getopt(argv, 'hi:o:', ['ifile=', 'ofile='])
    for opt, arg in opts:
        if opt == '-h':
            print_usage()
            sys.exit(2)
        elif opt in ("-i", "--ifile"):
            input_filename = arg
        elif opt in ("-o", "--ofile"):
            output_filename = arg

    df_input = pd.read_csv(input_filename)
    lit,alt = df_input['Literal'],df_input['Alternative']
    lit_per, alt_per = perplexity(lit,alt)
    distance = probing_distance(lit,alt)
    cosine_similarity = probing_similarity(lit,alt)
    prob_lit,prob_alt = probing_conditional_probability(lit,alt)
    lit_ori,lit_cor = posterior(lit,alt)
    n400 = cal_n400(lit_ori,lit_cor,lit,alt)
    p600 = cal_p600(lit_ori,lit_cor,lit,alt)
            
    correction_rate = len([i for i in range(len(lit)) if lit_ori[i]< lit_cor[i]])/len(lit)
    highest_posterior = ['false' if lit_ori[i]>lit_cor[i] else 'true' for i in range(len(lit))]

    df = pd.DataFrame({'Item':list(df_input['Item']),'Condition':list(df_input['Condition']),'Literal':lit,'Alternative':alt,
        'Perplexity_Literal':lit_per,'Perplexity_Alternative':alt_per,'Distance':distance,
        'Posterior (Not Corrected)':lit_ori,'Posterior (Corrected)':lit_cor,'Corrected':highest_posterior,
        'CondProb_Literal':prob_lit,'CondProb_Alternative':prob_alt,'Similarity':cosine_similarity,'N400':n400,'P600':p600})
    df.to_csv(output_filename)
    return lit_per,alt_per,correction_rate,n400,p600

if __name__ == "__main__":
    main(sys.argv[1:])
    