import pandas as pd
import glob
import os
import json
import pickle

# New code to add AF predictions
def Add_AF(inter, UniProt, models_to_use, AF_predictions, Str = True):
    UniProt_AF = list(filter(lambda x: UniProt in x, AF_predictions))
    UniProt_AF.sort(key=lambda x: int(x.split('-')[2][1:]))
    if Str == True:
        if UniProt_AF:
            for AF in UniProt_AF:
                models_to_use[inter][UniProt].append(['AF', AF])
    else: # for protein without structure information where its partner has structure information or both proteins do not have structure information
        Str_list = []
        inter_with_str = set(models_to_use.keys())
        if inter in inter_with_str: # for protein without structure information where its partner has structure information
            if UniProt_AF:
                for AF in UniProt_AF:
                    Str_list.append(['AF', AF])
                models_to_use[inter][UniProt] = Str_list
        else: # for protein where none of proteins in interaction has structure information
            if UniProt_AF:
                models_to_use[inter] = {}
                for AF in UniProt_AF:
                    Str_list.append(['AF', AF])
                models_to_use[inter][UniProt] = Str_list
    return models_to_use

def AF_incort(int_list, models_to_use):
    # All AF predictions
    """
    AF_predictions = []
    AF_prediction_path = '/fs/cbsuhyfs1/storage/resources/alphafold/data/'
    species = os.listdir(AF_prediction_path)
    for s in species:
        predictions = glob.glob(AF_prediction_path + s + '/*.pdb.gz')
        AF_predictions += predictions
    """
    with open('/fs/cbsuhyfs1/storage/dl953/AF_predictions.pkl', 'rb') as f:
        AF_predictions = pickle.load(f)

    inter_with_str = models_to_use.keys()
    for i, inter in enumerate(int_list):
        p1, p2 = inter[0], inter[1]
        if inter in inter_with_str: # at least one protein already has structure information
            if len(models_to_use[inter]) == 2: # Heterodimer, both proteins have structure information
                models_to_use = Add_AF(inter, p1, models_to_use, AF_predictions, Str = True)
                models_to_use = Add_AF(inter, p2, models_to_use, AF_predictions, Str = True)
            else: # Only one protein has structure information or homodimer
                if p1 == p2: # Homodimer
                    models_to_use = Add_AF(inter, p1, models_to_use, AF_predictions, Str = True)
                else:
                    key_prot = list(models_to_use[inter].keys())
                    if p1 == key_prot[0]: # if only p1 has structure information
                        models_to_use = Add_AF(inter, p1, models_to_use, AF_predictions, Str = True)
                        models_to_use = Add_AF(inter, p2, models_to_use, AF_predictions, Str = False)
                    else:
                        models_to_use = Add_AF(inter, p1, models_to_use, AF_predictions, Str = False)
                        models_to_use = Add_AF(inter, p2, models_to_use, AF_predictions, Str = True)
        else: # both proteins do not have structure information
            models_to_use = Add_AF(inter, p1, models_to_use, AF_predictions, Str = False)
            models_to_use = Add_AF(inter, p2, models_to_use, AF_predictions, Str = False)
    return models_to_use
