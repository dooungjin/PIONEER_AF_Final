import pandas as pd
import glob
import os
import json
import sys
import time
from pioneer.compile_features import *
from pioneer.prediction import *
from pioneer.data_generation import *
from pioneer.integration import *
from pioneer.AF_incorporation import *
from pioneer.models_to_use_cleaning import *
from timeit import default_timer as timer
from datetime import timedelta

Uni2Seq = pd.read_pickle('/local/storage/sl2678/interface_pred/pipeline/pipeline_data/uniprot_seq_dict.pkl')
pred_dir = '/local/storage/dl953/interface_pred/prediction/Models/predictions/'
preds = glob.glob(pred_dir + '*.pkl')
pio_inters = []
for inter in preds:
    pio_inters.append(os.path.basename(inter)[:-6])
    
# Interactions already predicted by PIONEER
pio_inters = list(set(pio_inters))

Input = pd.read_csv('./Input.txt', sep = '\t', header = 0)
Input['UniA'] = Input.apply(lambda x: x['uniprot_a'].split('-')[0], axis = 1)
Input['UniB'] = Input.apply(lambda x: x['uniprot_b'].split('-')[0], axis = 1)
Input['P1'] = Input.apply(lambda x: sorted([str(x['UniA']), str(x['UniB'])])[0], axis = 1)
Input['P2'] = Input.apply(lambda x: sorted([str(x['UniA']), str(x['UniB'])])[1], axis = 1)
Inters = Input[['P1', 'P2']] # 102 rows
Inters = Inters.drop_duplicates() # 98 rows
Inters = (Inters.apply(lambda x: x['P1'] + '_' + x['P2'], axis = 1)).values.tolist()

Inter2pred = list(filter(lambda x: x not in pio_inters, Inters))

int_list = []
seq_dict = {}
seqs = seq_dict.keys()
for inter in Inter2pred:
    p1, p2 = inter.split('_')
    int_list.append((p1, p2))
    if p1 not in seqs:
        seq_dict[p1] = Uni2Seq[p1]
    if p2 not in seqs:
        seq_dict[p2] = Uni2Seq[p2]

print("Number of interactions to predict:", len(int_list))

struct_data_dir = './feats_prediction/structs'
seq_feat_dir = './feats_prediction/seq_feats'
full_feat_dir = './feats_prediction/full_feats'
seq_seq_prediction_dir = './feats_prediction/seq_seq_predictions'
seq_str_prediction_dir = './feats_prediction/seq_str_predictions'
str_seq_prediction_dir = './feats_prediction/str_seq_predictions'
str_str_prediction_dir = './feats_prediction/str_str_predictions'
PIONEER_prediction_dir = './feats_prediction/PIONEER_prediction'

# parameters
AF_threshold = 80
# end parameters

# Initialize (empty) all directories
dirs = [struct_data_dir, seq_feat_dir, full_feat_dir, seq_seq_prediction_dir, seq_str_prediction_dir, str_seq_prediction_dir, str_str_prediction_dir]
for direc in dirs:
    files = glob.glob(os.path.join(direc, '*'))
    for f in files:
        os.remove(f)
print("Directories initialized")
    
compile_pioneer_seq_feats(int_list, seq_dict, seq_feat_dir)
compile_pioneer_full_feats(int_list, seq_dict, full_feat_dir, exclude_hom_struct=False)
print("Feature generated")

models_to_use = generate_models_to_use(int_list, seq_dict)
models_to_use = models_to_use_cleaning(models_to_use)
models_to_use = AF_incort(int_list, models_to_use)
print("models to use collected")

seq_seq, seq_str, str_seq, str_str = obtain_all_structures(int_list, models_to_use, struct_data_dir)
print("Structures collected")
   
print("Deep Learning input generation")
seq_seq_data = generate_seq_seq_data(seq_seq, seq_feat_dir) # seq seq prediction
# seq str prediction for interactions that only p2 has structure
seq_str_data_p1 = generate_seq_str_data(seq_str, seq_feat_dir, full_feat_dir, struct_data_dir, p2=False, AF_threshold = AF_threshold)
# seq str prediction for interactions that p1 and p2 both have structure. We need this for residues in p1 that are not predicted by str str model (interactions p1 and p2 both have structure)
seq_str_data_p1_backup = generate_seq_str_data(str_str, seq_feat_dir, full_feat_dir, struct_data_dir, p2=False, AF_threshold = AF_threshold)
# seq str prediction for interactions that only p1 has structure (We predict for residues in p2). So, a list of str_seq is an input
seq_str_data_p2 = generate_seq_str_data(str_seq, seq_feat_dir, full_feat_dir, struct_data_dir, p2=True, AF_threshold = AF_threshold)
# seq str prediction for interactions that p1 and p2 both have structure. We need this for residues in p2 that are not predicted by str str model (interactions p1 and p2 both have structure)
seq_str_data_p2_backup = generate_seq_str_data(str_str, seq_feat_dir, full_feat_dir, struct_data_dir, p2=True, AF_threshold = AF_threshold)
# str seq prediction for interactions that only p1 has structure
str_seq_data_p1 = generate_str_seq_data(str_seq, seq_feat_dir, full_feat_dir, struct_data_dir, p2=False, AF_threshold = AF_threshold)
# str seq prediction for interactions that only p2 has structure
str_seq_data_p2 = generate_str_seq_data(seq_str, seq_feat_dir, full_feat_dir, struct_data_dir, p2=True, AF_threshold = AF_threshold)
str_str_data = generate_str_str_data(str_str, full_feat_dir, struct_data_dir, AF_threshold = AF_threshold)

device = 'cpu' # GPU 0. If you want to use CPU, change this to 'cpu'. If you want to use GPU 1, change this to 'cuda:1'.
print("Prediction start")
seq_seq_prediction(seq_seq_data, seq_seq_prediction_dir, device)
seq_str_prediction(seq_str_data_p1, seq_str_prediction_dir, device)
seq_str_prediction(seq_str_data_p1_backup, seq_str_prediction_dir, device)
seq_str_prediction(seq_str_data_p2, seq_str_prediction_dir, device)
seq_str_prediction(seq_str_data_p2_backup, seq_str_prediction_dir, device)
str_seq_prediction(str_seq_data_p1, str_seq_prediction_dir, device)
str_seq_prediction(str_seq_data_p2, str_seq_prediction_dir, device)
str_str_prediction(str_str_data, str_str_prediction_dir, device)

integration(seq_seq_prediction_dir, seq_str_prediction_dir, str_seq_prediction_dir, str_str_prediction_dir, PIONEER_prediction_dir)
print("End PIONEER")
