import os
import glob
import pickle
import dill
import shutil
import tempfile
import pkg_resources
import numpy as np
import pandas as pd
from collections import defaultdict
from multiprocessing import Pool
from .utils import *
from .zdock import *
from .config import *
from .pair_potential import *
from .coevolution import sca, dca
from .js_divergence import calculate_js_div_from_msa
from .srescalc import calculate_SASA, gather_SASA, naccess
from .msa import generate_single_msa, parse_fasta, write_fasta, generate_clustal_input, run_clustal, format_clustal, msa_join
from shutil import copyfile

def calculate_expasy(seq_dict):
    """
    Calculate ExPaSy features.
    
    Args:
        seq_dict (dict): Dictionary mapping protein identifiers to sequences.
    
    Returns:
        A dictionary of dictionaries, feat -> protid -> feature array.
    
    """
    # Load ExPaSy data
    with open(pkg_resources.resource_filename(__name__, 'data/expasy.pkl'), 'rb') as f:
        expasy_dict = pickle.load(f)

    # Calculate ExPaSy features
    expasy_feat_dict = {}
    for feat in ['ACCE', 'AREA', 'BULK', 'COMP', 'HPHO', 'POLA', 'TRAN']:
        expasy_feat_dict[feat] = {}
        for prot in seq_dict:
            try:
                expasy_feat_dict[feat][prot] = calc_expasy(seq_dict[prot], feat, expasy_dict)
            except:
                print('Error calculating ExPaSy feature %s for %s' % (feat, prot))
                continue
    return expasy_feat_dict

def calculate_conservation(seq_dict):
    """
    Calculate JS divergence feature.
    
    Args:
        seq_dict (dict): Dictionary mapping protein identifiers to sequences.
        
    Returns:
        A dictionary mapping protein identifiers fo JS divergence feature arrays.
    
    """
    #DL: bg_distr [0.078 0.051 0.041 0.052 0.024 0.034 0.059 0.083 0.025 0.062 0.092 0.056 0.024 0.044 0.043 0.059 0.055 0.014 0.034 0.072]
    bg_distr = np.load(pkg_resources.resource_filename(__name__, 'data/bg_aa_distr.npy'))
    tmpdir = tempfile.mkdtemp()
    single_msa_file_dict = {} # id -> path
    js_feat_dict = {} # id -> js_array
    uniprot_seq_dict = {}
    
    # Step 1: Generate single MSAs for all proteins queried.
    for prot, sequence in seq_dict.items():
        if os.path.exists(os.path.join(JS_CACHE, '%s.npy' % prot)): # Retrieve results from cache if possible
            js_feat_dict[prot] = np.load(os.path.join(JS_CACHE, '%s.npy' % prot))
            continue
        if os.path.exists(os.path.join(MSA_CACHE, '%s_clustal_aligned.msa' % prot)): # Retrieve MSA if possible
            single_msa_file_dict[prot] = os.path.join(MSA_CACHE, '%s_clustal_aligned.msa' % prot)
            continue
        
        # If no JS and MSA cached, generate MSA.
        msa_file = generate_single_msa(prot, sequence, tmpdir) # Search for homologs using PSIBLAST
        
        if not os.path.exists(msa_file):
            print('PSIBLAST search failed for %s' % prot_id)
        if not uniprot_seq_dict:
            with open(UNIPROT_SEQ_PICKLE_PATH, 'rb') as f:
                uniprot_seq_dict = pickle.load(f)
        clustal_input_file = generate_clustal_input(prot, sequence, msa_file, uniprot_seq_dict, tmpdir) # Generate CLUSTAL input
        if clustal_input_file is None or not os.path.exists(clustal_input_file):
            continue
        clustal_output_file = run_clustal(prot, clustal_input_file, tmpdir) # Run CLUSTAL Omega
        if clustal_output_file is None or not os.path.exists(clustal_output_file):
            continue
        formatted_clustal_file = format_clustal(prot, clustal_output_file, tmpdir) # Format CLUSTAL output
        if formatted_clustal_file is not None or not os.path.exists(formatted_clustal_file):
            single_msa_file_dict[prot] = formatted_clustal_file
            os.system('cp %s %s' % (formatted_clustal_file, MSA_CACHE)) # Copy to MSA cache
    
    # Step 2: Calculate JS divergence from single MSAs generated.
    for prot in single_msa_file_dict:
        try:
            js_array = calculate_js_div_from_msa(single_msa_file_dict[prot], bg_distr, 0.0000001, 3, 0.5) # PARAM
            js_feat_dict[prot] = js_array
            np.save(os.path.join(JS_CACHE, '%s.npy' % prot), js_array) # Save to JS cache
        except:
            print('Error calculating JS divergence for %s' % prot)
            continue
    
    # Step 3: Remove temporary folder and files.
    shutil.rmtree(tmpdir)
    return js_feat_dict

def calculate_coevolution(int_list, seq_dict):
    """
    Calculate coevolution features.
    
    Args:
        int_list (list): A list of all interactions (tuple of two identifiers) to calculate coevolution features for.
        seq_dict (dict): Dictionary mapping protein identifiers to sequences.
        
    Returns:
        A tuple of two dictionaries. The first dictionary maps (id1, id2) -> feat -> (sca1, sca2) and the second dictionary maps
        (id1, id2) -> feat -> (dca1, dca2).
    
    """
    tmpdir = tempfile.mkdtemp()
    sca_feat_dict = {'max': {}, 'mean': {}, 'top10': {}}
    dca_feat_dict = {'DDI_max': {}, 'DDI_mean': {}, 'DDI_top10': {}, 'DMI_max': {}, 'DMI_mean': {}, 'DMI_top10': {}}
    
    ## Modified by Ben, reviewed by Dongjin
    # Step 1: Generate joined MSA files and compile a dictionary from interactions to joined MSA file paths.
    for i in int_list:        
        id1_aligned_msa = os.path.join(MSA_CACHE, '%s_clustal_aligned.msa' % i[0])
        id2_aligned_msa = os.path.join(MSA_CACHE, '%s_clustal_aligned.msa' % i[1])
        if (not os.path.isfile(id1_aligned_msa)) or (not os.path.isfile(id2_aligned_msa)):
            # If MSA is not already in cache, these features cannot be generated, DL: It is because coevolution is calculated after conservation
            continue

        sca_fn = os.path.join(SCA_CACHE, '_'.join(i) + '.sca')
        dca_fn = os.path.join(DCA_CACHE, '_'.join(i) + '.dca')

        if os.path.exists(sca_fn) and os.path.exists(dca_fn):
            continue # If SCA and DCA are already in cache, no need to generate joined MSA
        joined_msa = msa_join(id1_aligned_msa, id2_aligned_msa, tmpdir)
        if joined_msa is None:
            continue
    
        # Step 2: Calculate SCA and DCA result files from joined MSA files generated.
        # Each protein must have more than 10 residues # PARAM
        if (not os.path.isfile(sca_fn)) and (len(seq_dict[i[0]]) > 10) and (len(seq_dict[i[1]]) > 10):
            # Don't need to recalculate if the file already exists
            try:
                sca(joined_msa, os.path.join(tmpdir, '_'.join(i) + '.sca'))
                os.system('cp %s %s' % (os.path.join(tmpdir, '_'.join(i) + '.sca'), sca_fn)) # Copy SCA to cache
            except:
                print('Error calculating SCA for %s' % '_'.join(i))
                continue

        if len(seq_dict[i[0]]) + len(seq_dict[i[1]]) <= 1000 and not os.path.isfile(dca_fn):
            # Maximum length of joined MSA supported for DCA is 1000 # PARAM
            # Don't need to recalculate if the file already exists
            try:
                dca(joined_msa, os.path.join(tmpdir, '_'.join(i) + '.dca'))
                os.system('cp %s %s' % (os.path.join(tmpdir, '_'.join(i) + '.dca'), dca_fn)) # Copy DCA to cache
            except:
                print('Error calculating DCA for %s' % '_'.join(i))
                continue

    # Step 3: Calculates final SCA and DCA results from their result files.
    for i in int_list:
        if os.path.exists(os.path.join(SCA_CACHE, '_'.join(i) + '.sca')):
            id1_max, id2_max, id1_mean, id2_mean, id1_top10, id2_top10 = aggregate_sca(os.path.join(SCA_CACHE, '_'.join(i) + '.sca'), seq_dict)
            sca_feat_dict['max'][i] = (id1_max, id2_max)
            sca_feat_dict['mean'][i] = (id1_mean, id2_mean)
            sca_feat_dict['top10'][i] = (id1_top10, id2_top10)
        
        if os.path.exists(os.path.join(DCA_CACHE, '_'.join(i) + '.dca')):
            # Split DCA result into an MI file and a DI file
            split_dca(os.path.join(DCA_CACHE, '_'.join(i) + '.dca'), tmpdir)
            # Calculate DMI features from *.mi files
            id1_max, id2_max, id1_mean, id2_mean, id1_top10, id2_top10 = aggregate_dca(os.path.join(tmpdir, '_'.join(i) + '.mi'), seq_dict)
            dca_feat_dict['DMI_max'][i] = (id1_max, id2_max)
            dca_feat_dict['DMI_mean'][i] = (id1_mean, id2_mean)
            dca_feat_dict['DMI_top10'][i] = (id1_top10, id2_top10)
            # Calculate DDI features from *.di files
            id1_max, id2_max, id1_mean, id2_mean, id1_top10, id2_top10 = aggregate_dca(os.path.join(tmpdir, '_'.join(i) + '.di'), seq_dict)
            dca_feat_dict['DDI_max'][i] = (id1_max, id2_max)
            dca_feat_dict['DDI_mean'][i] = (id1_mean, id2_mean)
            dca_feat_dict['DDI_top10'][i] = (id1_top10, id2_top10)
            
    # Step 4: Remove temporary folder and files.
    shutil.rmtree(tmpdir)
    return sca_feat_dict, dca_feat_dict

def calculate_excluded_pdb_dict(int_list, seq_dict, uniprot_seq_dict):
    """
    Calculate dictionary containing PDBs to exclude for each interaction.
    
    Args:
        int_list (list): A list of all interactions (tuple of two identifiers) to calculate coevolution features for.
        seq_dict (dict): Dictionary mapping protein identifiers to sequences.
        uniprot_seq_dict (dict): Dictionary containing all UniProt sequences, uniprot_id -> sequence.
        
    Returns:
        A dictionary mapping each interaction to a list of PDB structures to exclude.
    
    """
    # Step 1: Filter out query proteins with a UniProt ID.
    tmpdir = tempfile.mkdtemp()
    seq_dict_subset = {}
    
    for prot, seq in seq_dict.items():
        if prot in uniprot_seq_dict:
            seq_dict[prot] = seq
    if seq_dict_subset:
        write_fasta(seq_dict_subset, os.path.join(tmpdir, 'query.fasta'))
    
    # Step 2: Build a sequence search database from the SIFTS info file.
    sifts_info_file = pkg_resources.resource_filename(__name__, 'data/sifts_uniprot_info.txt')
    sifts_dict = {}
    with open(sifts_info_file, 'r') as f:
        for line in f:
            if line.startswith('UniProt'):
                continue
            try:
                prot, _, _, _, _, seq = line.strip().split('\t') # There may be faulty lines
            except:
                continue
            sifts_dict[prot] = seq
    write_fasta(sifts_dict, os.path.join(tmpdir, 'sifts.fasta'))
    os.system('%s -in %s -dbtype prot' % (MAKEBLASTDB, os.path.join(tmpdir, 'sifts.fasta')))
    
    # Step 3: Run BLASTP for the query sequences against the SIFTS search database.
    if os.path.exists(os.path.join(tmpdir, 'query.fasta')):
        os.system('%s -query %s -db %s -num_threads 12 -outfmt 6 > %s' % (BLASTP, os.path.join(tmpdir, 'query.fasta'), os.path.join(tmpdir, 'sifts.fasta'), os.path.join(tmpdir, 'blastp_output.txt'))) # PARAM
    
    # Step 4: Parse BLASTP output and generate a dictionary specifying the homologs of all proteins.
    if os.path.exists(os.path.join(tmpdir, 'blastp_output.txt')):
        homologs = defaultdict(set)
        with open(os.path.join(tmpdir, 'blastp_output.txt'), 'r') as f:
            for line in f:
                query, target, _, _, _, _, _, _, _, _, e_value, _ = line.strip().split('\t')
                if float(e_value) < 1.0: # PARAM
                    homologs[query].add(target)
                    homologs[target].add(query)
        # Include proteins as their own homologs.
        for i in int_list:
            homologs[i[0]].add(i[1])
            homologs[i[1]].add(i[0])
            
        # Step 5: Build a file specifying excluded PDBs for each interaction.
        pdb2uniprots = defaultdict(set)  # Store all uniprots seen in each PDB
        pdbuniprot2count = defaultdict(int)  # Store number of times a uniprot is seen in each PDB
        uniprot2pdb = defaultdict(set)  # All pdbs associted with a uniprot and its homologs (reduce the set of uniprots to check for each interaction)
        with open(PDBRESMAP_PATH, 'r') as f:
            for line in f:
                if line.startswith('PDB'):
                    continue
                pdb, _, uniprot = line.strip().split('\t')[:3]
                pdb2uniprots[pdb].add(uniprot)
                pdbuniprot2count[(pdb, uniprot)] += 1
                homologs[uniprot].add(uniprot)
                for prot in homologs[uniprot]:
                    uniprot2pdb[prot].add(pdb)
        
        # Write the file specifying excluded PDBs.
        with open(os.path.join(tmpdir, 'excluded_pdb.txt'), 'w') as f:
            f.write('\t'.join(['UniProtA', 'UniProtB', 'hasCC', 'excludedPDBs'])+'\n')
            for idx, i in enumerate(int_list):
                id1, id2 = i
                excluded_pdbs = set()	
                has_CC = 'N'
                for pdb in uniprot2pdb[id1].union(uniprot2pdb[id2]):
                    if id1 == id2: # Homodimers
                        if pdbuniprot2count[(pdb, id1)] > 1:
                            excluded_pdbs.add(pdb)
                            has_CC = 'Y'
                        num_homologs_in_pdb = sum([pdbuniprot2count[(pdb, h)] for h in homologs[id1]])
                        if num_homologs_in_pdb > 1:
                            excluded_pdbs.add(pdb)
                    else: # Heterodimers
                        if id1 in pdb2uniprots[pdb] and id2 in pdb2uniprots[pdb]:
                            excluded_pdbs.add(pdb)
                            has_CC = 'Y'
                        if len(homologs[id1].intersection(pdb2uniprots[pdb])) > 0 and len(homologs[id2].intersection(pdb2uniprots[pdb])) > 0:
                            excluded_pdbs.add(pdb)
                f.write('%s\t%s\t%s\t%s\n' % (id1, id2, has_CC, ';'.join(sorted(excluded_pdbs))))
        excluded_pdb_dict = generate_excluded_pdb_dict(os.path.join(tmpdir, 'excluded_pdb.txt'), int_list)
        shutil.rmtree(tmpdir)
        return excluded_pdb_dict
    else:
        shutil.rmtree(tmpdir)
        return defaultdict(set)

def calculate_sasa(int_list, seq_dict, excluded_pdb_dict):
    """
    Calculate SASA feature.
    
    Args:
        int_list (list): A list of all interactions (tuple of two identifiers) to calculate coevolution features for.
        seq_dict (dict): Dictionary mapping protein identifiers to sequences.
        excluded_pdb_dict (dict): Dictionary mapping each interaction to a list of PDB structures to exclude.
        
    Returns:
        A tuple two dictionaries. The first maps (id1, id2) -> (sasa_max_1, sasa_max_2) and the second maps (id1, id2) -> (sasa_mean_1, sasa_mean_2).
    """
  
    prot_with_sasa = set() # Proteins that already have SASA information from ModBase
    #DL: /local/storage/sl2678/interface_pred/pipeline/pipeline_data/modbase/parsed_files/SASA_modbase_human.txt has pre-calculated sasa values
    with open(os.path.join(MODBASE_CACHE, 'parsed_files', 'SASA_modbase_human.txt'), 'r') as f:
        for i, line in enumerate(f):
            if i > 0:
                prot_with_sasa.add(line.strip().split('\t')[0])
    if os.path.exists(os.path.join(MODBASE_CACHE, 'parsed_files', 'SASA_modbase_other.txt')):
        with open(os.path.join(MODBASE_CACHE, 'parsed_files', 'SASA_modbase_other.txt'), 'r') as f:
            for i, line in enumerate(f):
                if i > 0:
                    prot_with_sasa.add(line.strip().split('\t')[0])
    else: # Create non-human modbase file if it does not exist.
        with open(os.path.join(MODBASE_CACHE, 'parsed_files', 'SASA_modbase_other.txt'), 'w') as f:
            f.write('\t'.join(['uniprot', 'template_length', 'target_length', 'template_pdb', 'target_begin', 'target_end', 'sequence_identity', 'model_score', 'modpipe_quality_score', 'zDOPE', 'eVALUE', 'modbase_modelID', 'SASA']) + '\n')
    tmpdir = tempfile.mkdtemp()
    
    # Step 1: download, parse, fix and filter ModBase models that do not exist in our folder, and create a summary file.
    for prot in seq_dict:
        if prot in prot_with_sasa:
            continue
        # Download models from ModBase
        download_modbase(prot, os.path.join(MODBASE_CACHE, 'models', 'uniprot'))
        modbase_uniprot_pdb = os.path.join(MODBASE_CACHE, 'models', 'uniprot', '%s.pdb' % prot) # prot_id.pdb in _MODBASE_CACHE/models/uniprot/
        if os.path.exists(modbase_uniprot_pdb):
            # Parse ModBase models
            model_hashes = parse_modbase(modbase_uniprot_pdb, os.path.join(MODBASE_CACHE, 'models', 'hash'), os.path.join(MODBASE_CACHE, 'models', 'header'), len(seq_dict[prot]))
            for hash_file in model_hashes:
                # Fix hash files
                fix_modbase(hash_file)
    
    # Filter modbase models and generate a summary file.
    filter_modbase(os.path.join(MODBASE_CACHE, 'models', 'header'), set(seq_dict) - prot_with_sasa, os.path.join(tmpdir, 'select_modbase_models.txt'))
    if not os.path.exists(os.path.join(MODBASE_CACHE, 'parsed_files', 'select_modbase_models.txt')):
        with open(os.path.join(MODBASE_CACHE, 'parsed_files', 'select_modbase_models.txt'), 'w') as sel_f:
            sel_f.write('\t'.join(['uniprot', 'template_length', 'target_length', 'template_pdb', 'template_chain', 'target_begin', 'target_end', 'sequence_identity', 'model_score', 'modpipe_quality_score', 'zDOPE', 'eVALUE', 'modbase_modelID']) + '\n')
    with open(os.path.join(MODBASE_CACHE, 'parsed_files', 'select_modbase_models.txt'), 'a') as sel_f:
        with open(os.path.join(tmpdir, 'select_modbase_models.txt'), 'r') as f:
            for i, line in enumerate(f):
                if i != 0:
                    sel_f.write(line)
    
    # Step 2: Calculate SASA for all selected models.
    os.mkdir(os.path.join(tmpdir, 'SASA'))
    selected_df = pd.read_csv(os.path.join(tmpdir, 'select_modbase_models.txt'), sep='\t')
    model_path_list, length_list, header_info_list = [], [], []
    for _, row in selected_df.iterrows():
        model_path_list.append(os.path.join(os.path.join(MODBASE_CACHE, 'models', 'hash'), '%s.pdb' % row['modbase_modelID']))
        length_list.append(len(seq_dict[row['uniprot']]))
        header_info_list.append('\t'.join([str(x) for x in row.values]))
    pool = Pool(10) # PARAM
    pool.starmap(calculate_SASA, zip(model_path_list, length_list, header_info_list, [os.path.join(tmpdir, 'SASA')] * len(model_path_list)))
    pool.close()
    pool.join()
    
    # Step 3: Gather SASA values and calculate SASA features.
    gather_SASA(os.path.join(tmpdir, 'SASA'), os.path.join(MODBASE_CACHE, 'parsed_files', 'SASA_modbase_other.txt'))
    uniprot2chains = generate_uniprot2chains(SASA_PDB_PATH, os.path.join(MODBASE_CACHE, 'parsed_files', 'SASA_modbase_human.txt'), os.path.join(MODBASE_CACHE, 'parsed_files', 'SASA_modbase_other.txt'))
    sasa_max_dict, sasa_mean_dict = {}, {}
    
    for i in int_list:
        id1, id2 = i
        id1_sasas = []
        for pdb in uniprot2chains[id1]:
            if pdb not in excluded_pdb_dict[i]:
                id1_sasas += [sasas for sasas in uniprot2chains[id1][pdb] if len(sasas) == len(seq_dict[id1])]
        id2_sasas = []
        for pdb in uniprot2chains[id2]:
            if pdb not in excluded_pdb_dict[i]:
                id2_sasas += [sasas for sasas in uniprot2chains[id2][pdb] if len(sasas) == len(seq_dict[id2])]
        if len(id1_sasas) == 0 and len(id2_sasas) == 0:
            continue
        if len(id1_sasas) == 0:
            id1_means, id1_max = [np.nan] * len(seq_dict[id1]), [np.nan] * len(seq_dict[id1])
        else:
            try:
                mdat = np.ma.masked_array(id1_sasas, np.isnan(id1_sasas))
                id1_means = np.mean(mdat, axis=0)
                id1_means = [id1_means.data[r] if id1_means.mask[r]==False else np.nan for r in range(len(id1_means.data))]
                id1_max = np.max(mdat, axis=0)
                id1_max = [id1_max.data[r] if id1_max.mask[r]==False else np.nan for r in range(len(id1_max.data))]
            except:
                continue
        if len(id2_sasas) == 0:
            id2_means, id2_max = [np.nan] * len(seq_dict[id2]), [np.nan] * len(seq_dict[id2])
        else:
            try:
                mdat = np.ma.masked_array(id2_sasas, np.isnan(id2_sasas))
                id2_means = np.mean(mdat, axis=0)
                id2_means = [id2_means.data[r] if id2_means.mask[r]==False else np.nan for r in range(len(id2_means.data))]
                id2_max = np.max(mdat, axis=0)
                id2_max = [id2_max.data[r] if id2_max.mask[r]==False else np.nan for r in range(len(id2_max.data))]
            except:
                continue
        if len(id1_means) != len(seq_dict[id1]) or len(id2_means) != len(seq_dict[id2]):
            continue
        sasa_max_dict[i] = (np.array(id1_max), np.array(id2_max))
        sasa_mean_dict[i] = (np.array(id1_means), np.array(id2_means))
        
    # Step 4: Remove temporary folder and files.
    #shutil.rmtree(tmpdir)
    return sasa_max_dict, sasa_mean_dict

def calculate_zdock(int_list, seq_dict, excluded_pdb_dict):
    """
    Calculate ZDOCK features.
    
    Args:
        int_list (list): A list of all interactions (tuple of two identifiers) to calculate coevolution features for.
        seq_dict (dict): Dictionary mapping protein identifiers to sequences.
        excluded_pdb_dict (dict): Dictionary mapping each interaction to a list of PDB structures to exclude.
        
    Returns:
        A dictionary of the ZDOCK features mapping (id1, id2) -> feat -> (zdock1, zdock2).
    
    """
    # Step 1: Generate PDB, ModBase and mixed docking sets.
    tmpdir = tempfile.mkdtemp()
    os.mkdir(os.path.join(tmpdir, 'docking_set'))
    #DL: generate_pdb_docking_set is defined in utils.py
    pdb_data = generate_pdb_docking_set(int_list, PDBRESMAP_PATH, seq_dict, excluded_pdb_dict, os.path.join(tmpdir, 'docking_set', 'pdb_docking_set.txt'))
    #DL: pdb_data example
    '''
             PDB Chain UniProt  ...                       UniProtResSecondaryStructure                                        Covered_Res Coverage
362802  6ICZ     V  Q9HCG8  ...  HHHHHHHHHHHHHHHHHHHHHHHHHTTTHHHHHHHHHHHTTTTTHH...  [149, 150, 151, 152, 153, 154, 155, 156, 157, ...      452
332080  5Z58     V  Q9HCG8  ...  HHHHHHHHHHHHHHHHHHHHHHHHHTTTHHHHHHHHHHHHTTTTHH...  [149, 150, 151, 152, 153, 154, 155, 156, 157, ...      451
82671   2J0S     A  P38919  ...  TTTTTTTTTTTTTTTTTTHHHHHHTHHHHHHHHHHHHTTTTHHHHH...  [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 3...      391
80211   2HYI     C  P38919  ...  TTTTTTTTTTTTTTTTTHHHHHTTHHHHHHHHHHHHTTTTHHHHHH...  [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 3...      390
80215   2HYI     I  P38919  ...  TTTTTTTTTTTTTTTTTHHHHHTTHHHHHHHHHHHHTTTTTTHHHH...  [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 3...      390
82663   2J0Q     A  P38919  ...  TTTTTTTTTTTTTTTTTTHHHHHTHHHHHHHHHHHHTTTTHHHHHH...  [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 3...      390
82664   2J0Q     B  P38919  ...  HHHHHTTTTTTTTTTTTTHHHHHTHHHHHHHHHHHHTTTTHHHHHH...  [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 3...      390
108802  2XB2     A  P38919  ...  TTTTTTTTTTTTTTTTTTHHHHHTHHHHHHHHHHHHTTTTHHHHHH...  [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 3...      390
108809  2XB2     X  P38919  ...  TTTTTTTTTTTTTTTTTTHHHHHTHHHHHHHHHHHTTTTTHHHHHH...  [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 3...      390
130257  3EX7     C  P38919  ...  TTTTTTTTTTTTTTTTTHHHHHTTHHHHHHHHHHHHTTTTTTHHHH...  [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 3...      390
130261  3EX7     H  P38919  ...  TTTTTTTTTTTTTTTTTHHHHHTTHHHHHHHHHHHTTTTTHHHHHH...  [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 3...      390
193915  4C9B     A  P38919  ...  THHHHHTEEETTTTTTTTHHHHHTTHHHHHHHHHHHHTTTTTTHHH...  [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 3...      389
362827  6ICZ     u  P38919  ...  TTTTTTTTTTTTTTTTTHHHHHTTHHHHHHHHHHHHTTTTHHHHHH...  [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 3...      386
80163   2HXY     B  P38919  ...  TTHHHHHTHHHHHHHHHHHHTTTTHHHHHHHHHHHHHHTTEEEETT...  [38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 4...      374
80164   2HXY     C  P38919  ...  TTHHHHHTHHHHHHHHHHHHTTTTHHHHHHHHHHHHHHTTEEEETT...  [38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 4...      374
80165   2HXY     D  P38919  ...  TTHHHHHTHHHHHHHHHHHHTTTTHHHHHHHHHHHHHHTTEEEETT...  [38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 4...      374
82681   2J0U     A  P38919  ...  TTHHHHHTHHHHHHHHHHHTTTTTHHHHHHHHHHHHHHTTEEEETT...  [38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 4...      362
82682   2J0U     B  P38919  ...  TTHHHHHTHHHHHHHHHHHTTTTTHHHHHHHHHHHHHTTTEEETTT...  [38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 4...      341
193916  4C9B     B  Q9HCG8  ...  TTTTTTTTTTTTHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHTT...  [123, 124, 125, 126, 127, 128, 129, 130, 131, ...      277
407929  6YVH     A  Q9HCG8  ...  TTTTTHHHHHHHTTTTTTHHHHHHHHHHHHHHHHHHHHHHHTTTHH...  [130, 131, 132, 133, 134, 135, 136, 137, 138, ...      274
407930  6YVH     B  Q9HCG8  ...  TTTTTTHHHHHHTTTTTTHHHHHHHHHHHHHHHHHHHHHHHTTTHH...  [130, 131, 132, 133, 134, 135, 136, 137, 138, ...      274
407932  6YVH     D  Q9HCG8  ...  TTTTTHHHHHHHTTTTTTHHHHHHHHHHHHHHHHHHHHHHHTTTHH...  [130, 131, 132, 133, 134, 135, 136, 137, 138, ...      274
407939  6YVH     K  P38919  ...  TTTTTTEEEEEEEEHHHHHHHHHHHHHHHHTTTEEEEETTHHHHHH...  [246, 247, 248, 249, 250, 251, 252, 253, 254, ...      164
407936  6YVH     H  P38919  ...  TTTTTTEEEEEEEETHHHHHHHHHHHHHHHTTTEEEEETTHHHHHH...  [246, 247, 248, 249, 250, 251, 252, 253, 254, ...      162
407938  6YVH     J  P38919  ...  TTTTTTEEEEEEEETHHHHHHHHHHHHHHHTTTEEEEETTHHHHHH...  [246, 247, 248, 249, 250, 251, 252, 253, 254, ...      162
407940  6YVH     L  P38919  ...  TTTTTTEEEEEEEEHHHHHHHHHHHHHHHHTTTEEEEETTHHHHHH...  [246, 247, 248, 249, 250, 251, 252, 253, 254, ...      162
   '''
    mb_data = generate_modbase_docking_set(int_list, os.path.join(MODBASE_CACHE, 'parsed_files', 'SASA_modbase_human.txt'), os.path.join(MODBASE_CACHE, 'parsed_files', 'SASA_modbase_other.txt'), seq_dict, excluded_pdb_dict, os.path.join(tmpdir, 'docking_set', 'modbase_docking_set.txt'))
    generate_mixed_docking_set(int_list, pdb_data, mb_data, seq_dict, excluded_pdb_dict, os.path.join(tmpdir, 'docking_set', 'mixed_docking_set.txt'))
    
    # Step 2: Perform Docking with ZDOCK.
    # Step 2.1: Perform PDB docking.
    docking_df = pd.read_csv(os.path.join(tmpdir, 'docking_set', 'pdb_docking_set.txt'), sep='\t')
    receptor_pdb_list, ligand_pdb_list, receptor_chain_list, ligand_chain_list = [], [], [], []
    pdb_receptor_ligand = set()
    for _, row in docking_df.iterrows():
        if row['CovA'] >= row['CovB']:
            receptor, ligand = row['SubA'], row['SubB']
        else:
            receptor, ligand = row['SubB'], row['SubA']
        pdb_receptor_ligand.add((receptor, ligand))
        if os.path.exists(os.path.join(ZDOCK_CACHE, 'pdb_docked_models', 'PDB%s.ENT_%s--' % tuple(receptor.split('_')) + 'PDB%s.ENT_%s--ZDOCK.out' % tuple(ligand.split('_')))):
            continue
        receptor_pdb, receptor_chain = receptor.split('_')
        ligand_pdb, ligand_chain = ligand.split('_')
        receptor_pdb = os.path.join(PDB_DATA_DIR, receptor_pdb[1:-1].lower(), 'pdb%s.ent.gz' % receptor_pdb.lower())
        ligand_pdb = os.path.join(PDB_DATA_DIR, ligand_pdb[1:-1].lower(), 'pdb%s.ent.gz' % ligand_pdb.lower())
        receptor_pdb_list.append(receptor_pdb)
        ligand_pdb_list.append(ligand_pdb)
        receptor_chain_list.append(receptor_chain)
        ligand_chain_list.append(ligand_chain)
    os.system('taskset -p 0xffffffff %d' % os.getpid())
    my_pool = Pool(10) # PARAM
    my_pool.starmap(zdock, zip(receptor_pdb_list, ligand_pdb_list, receptor_chain_list, ligand_chain_list, [os.path.join(os.path.join(ZDOCK_CACHE, 'pdb_docked_models'))] * len(receptor_pdb_list)))
    my_pool.close()
    my_pool.join()
    
    # Step 2.2: Perform ModBase docking.
    docking_df = pd.read_csv(os.path.join(tmpdir, 'docking_set', 'modbase_docking_set.txt'), sep='\t')
    receptor_pdb_list, ligand_pdb_list = [], []
    modbase_receptor_ligand = set()
    for _, row in docking_df.iterrows():
        if row['CovA'] >= row['CovB']:
            receptor, ligand = row['SubA'], row['SubB']
        else:
            receptor, ligand = row['SubB'], row['SubA']
        modbase_receptor_ligand.add((receptor, ligand))
        if os.path.exists(os.path.join(ZDOCK_CACHE, 'modbase_docked_models', '%s--%s--ZDOCK.out' % (receptor.upper(), ligand.upper()))):
            continue
        receptor_pdb = os.path.join(MODBASE_CACHE, 'models', 'hash', receptor + '.pdb')
        ligand_pdb =  os.path.join(MODBASE_CACHE, 'models', 'hash', ligand + '.pdb')
        receptor_pdb_list.append(receptor_pdb)
        ligand_pdb_list.append(ligand_pdb)
    os.system('taskset -p 0xffffffff %d' % os.getpid())
    my_pool = Pool(10) # PARAM
    my_pool.starmap(zdock, zip(receptor_pdb_list, ligand_pdb_list, ['_'] * len(receptor_pdb_list), ['_'] * len(receptor_pdb_list), [os.path.join(os.path.join(ZDOCK_CACHE, 'modbase_docked_models'))] * len(receptor_pdb_list)))
    my_pool.close()
    my_pool.join()
    
    # Step 2.3: Perform Mixed docking.
    docking_df = pd.read_csv(os.path.join(tmpdir, 'docking_set', 'mixed_docking_set.txt'), sep='\t')
    receptor_pdb_list, ligand_pdb_list, receptor_chain_list, ligand_chain_list = [], [], [], []
    mixed_receptor_ligand = set()
    for _, row in docking_df.iterrows():
        if row['CovA'] >= row['CovB']:
            receptor, ligand = row['SubA'], row['SubB']
        else:
            receptor, ligand = row['SubB'], row['SubA']
        mixed_receptor_ligand.add((receptor, ligand))
        if len(receptor) == 32:
            if os.path.exists(os.path.join(ZDOCK_CACHE, 'mixed_docked_models', '%s--PDB%s.ENT_%s--ZDOCK.out' % (receptor.upper(), ligand.split('_')[0], ligand.split('_')[1]))):
                continue
        else:
            if os.path.exists(os.path.join(ZDOCK_CACHE, 'mixed_docked_models', 'PDB%s.ENT_%s--%s--ZDOCK.out' % (receptor.split('_')[0], receptor.split('_')[1], ligand.upper()))):
                continue
        if len(receptor) == 32:
            receptor_chain = '_'
            receptor_pdb = os.path.join(MODBASE_CACHE, 'models', 'hash', receptor + '.pdb')
        else:
            receptor_pdb, receptor_chain = receptor.split('_')
            receptor_pdb = os.path.join(PDB_DATA_DIR, receptor_pdb[1:-1].lower(), 'pdb%s.ent.gz' % receptor_pdb.lower())
        if len(ligand) == 32:
            ligand_chain = '_'
            ligand_pdb = os.path.join(MODBASE_CACHE, 'models', 'hash', ligand + '.pdb')
        else:
            ligand_pdb, ligand_chain = ligand.split('_')
            ligand_pdb = os.path.join(PDB_DATA_DIR, ligand_pdb[1:-1].lower(), 'pdb%s.ent.gz' % ligand_pdb.lower())
        receptor_pdb_list.append(receptor_pdb)
        ligand_pdb_list.append(ligand_pdb)
        receptor_chain_list.append(receptor_chain)
        ligand_chain_list.append(ligand_chain)
    os.system('taskset -p 0xffffffff %d' % os.getpid())
    my_pool = Pool(10) # PARAM
    my_pool.starmap(zdock, zip(receptor_pdb_list, ligand_pdb_list, receptor_chain_list, ligand_chain_list, [os.path.join(os.path.join(ZDOCK_CACHE, 'mixed_docked_models'))] * len(receptor_pdb_list)))
    my_pool.close()
    my_pool.join()
    
    # Step 2.4: Write docking sets as temporary files.
    with open(os.path.join(tmpdir, 'pdb_docked_models.txt'), 'w') as f:
        for r, l in pdb_receptor_ligand:
            f.write(r + '\t' + l + '\n')
            
    with open(os.path.join(tmpdir, 'modbase_docked_models.txt'), 'w') as f:
        for r, l in modbase_receptor_ligand:
            f.write(r + '\t' + l + '\n')
            
    with open(os.path.join(tmpdir, 'mixed_docked_models.txt'), 'w') as f:
        for r, l in mixed_receptor_ligand:
            f.write(r + '\t' + l + '\n')
    
    # Step 3: Calculate ZDOCK features for each docked model.
    # Step 3.1: Calculate for ModBase docked models first.
    os.system('python -m pioneer.calculate_zdock_feats -c 10 -f dist3d -i %s -o %s -s modbase' % (os.path.join(tmpdir, 'modbase_docked_models.txt'), os.path.join(tmpdir, 'dist3d_modbase_docking.txt'))) # PARAMS
    
    # Step 3.2: Calculate for PDB docked models.
    os.system('python -m pioneer.calculate_zdock_feats -c 10 -f dist3d -i %s -o %s -s pdb' % (os.path.join(tmpdir, 'pdb_docked_models.txt'), os.path.join(tmpdir, 'dist3d_pdb_docking.txt'))) # PARAMS
    
    # Step 3.3: Calculate for mixed docked models.
    os.system('python -m pioneer.calculate_zdock_feats -c 10 -f dist3d -i %s -o %s -s mixed' % (os.path.join(tmpdir, 'mixed_docked_models.txt'), os.path.join(tmpdir, 'dist3d_mixed_docking.txt'))) # PARAMS
    
    # Step 4: Aggregate ZDOCK feature.
    interaction2dist3d = defaultdict(lambda: defaultdict(lambda: ([], [])))  # (p1, p2) -> (subA, subB) -> ([dsasas1...], [dsasas2...])
    pdb2uniprots = defaultdict(set)                                          # Store all uniprots seen in each PDB
    dist3d_lists = [parse_dictionary_list(os.path.join(tmpdir, f)) for f in ['dist3d_pdb_docking.txt', 'dist3d_modbase_docking.txt', 'dist3d_mixed_docking.txt'] if os.path.exists(os.path.join(tmpdir, f))]
    #DL: See dist3d_lists.txt for examples of this list 
    for e in sum(dist3d_lists, []):
        interaction = (e['UniProtA'], e['UniProtB'])
        dock_pair = (e['SubunitA'], e['SubunitB'])
        zdock_score = float(e['ZDOCK_Score'])
        if interaction not in int_list:
            continue
        if interaction in interaction2dist3d and dock_pair not in interaction2dist3d[interaction]:
            continue
        if zdock_score < 0: # PARAMS, docking_score_cutoff
            continue
        dist3ds1 = np.array([float(r) if r != 'nan' else np.nan for r in e['UniProtA_dist3d'].split(';')])
        dist3ds2 = np.array([float(r) if r != 'nan' else np.nan for r in e['UniProtB_dist3d'].split(';')])
        if all((dist3ds1 == 0.0) | np.isnan(dist3ds1)) or all((dist3ds2 == 0.0) | np.isnan(dist3ds2)):
            continue
        if e['UniProtA'] == e['UniProtB']:  # Homodimers have the same features for both proteins, save minimum dist3d for either subunit.
            both_dist3ds = [dist3ds1, dist3ds2]
            nan_mask = np.ma.masked_array(both_dist3ds, np.isnan(both_dist3ds))
            dist3d_min = np.min(nan_mask, axis=0)
            dist3d_min = np.array([dist3d_min.data[r] if dist3d_min.mask[r] == False else np.nan for r in range(len(dist3d_min.data))])
            interaction2dist3d[interaction][dock_pair][0].append(dist3d_min)
            interaction2dist3d[interaction][dock_pair][1].append(dist3d_min)
        else:
            interaction2dist3d[interaction][dock_pair][0].append(dist3ds1)
            interaction2dist3d[interaction][dock_pair][1].append(dist3ds2)
            
    zdock_feat_dict = {'top1': {}, 'max': {}, 'mean': {}, 'min': {}}
    for i in interaction2dist3d:
        id1_dist3ds, id2_dist3ds = [], []
        for docki, dock_pair in enumerate(interaction2dist3d[i]):
            if docki == 0:
                zdock_feat_dict['top1'][i] = (interaction2dist3d[i][dock_pair][0][0], interaction2dist3d[i][dock_pair][1][0])
                id1_dist3ds += interaction2dist3d[i][dock_pair][0]
                id2_dist3ds += interaction2dist3d[i][dock_pair][1]

        mdat = np.ma.masked_array(id1_dist3ds, np.isnan(id1_dist3ds))
        id1_mean = np.mean(mdat, axis=0)
        id1_mean = [id1_mean.data[r] if id1_mean.mask[r]==False else np.nan for r in range(len(id1_mean.data))]
        id1_max = np.max(mdat, axis=0)
        id1_max = [id1_max.data[r] if id1_max.mask[r]==False else np.nan for r in range(len(id1_max.data))]
        id1_min = np.min(mdat, axis=0)
        id1_min = [id1_min.data[r] if id1_min.mask[r]==False else np.nan for r in range(len(id1_min.data))]
        mdat = np.ma.masked_array(id2_dist3ds, np.isnan(id2_dist3ds))
        id2_mean = np.mean(mdat, axis=0)
        id2_mean = [id2_mean.data[r] if id2_mean.mask[r]==False else np.nan for r in range(len(id2_mean.data))]
        id2_max = np.max(mdat, axis=0)
        id2_max = [id2_max.data[r] if id2_max.mask[r]==False else np.nan for r in range(len(id2_max.data))]
        id2_min = np.min(mdat, axis=0)
        id2_min = [id2_min.data[r] if id2_min.mask[r]==False else np.nan for r in range(len(id2_min.data))]
        zdock_feat_dict['max'][i] = (np.array(id1_max), np.array(id2_max))
        zdock_feat_dict['mean'][i] = (np.array(id1_mean), np.array(id2_mean))
        zdock_feat_dict['min'][i] = (np.array(id1_min), np.array(id2_min))
    
    # Step 4: Remove temporary folder and files.
    shutil.rmtree(tmpdir)
    return zdock_feat_dict

def calculate_raptorX(seq_dict):
    """
    Calculate RaptorX features.
    
    Args:
        seq_dict (dict): Dictionary mapping protein identifiers to sequences.
        
    Returns:
        A dictionary of RaptorX features mapping id -> feature -> raptorx_array.
    
    """
    # Step 1: Run RaptorX for all proteins.
    #tmpdir = tempfile.mkdtemp()
    tmpdir = RAPTORX_CACHE
    cur_dir = os.getcwd()
    os.chdir(RAPTORX)
    for prot in seq_dict: 
        if os.path.exists(RAPTORX_CACHE + prot):
            continue

        with open(os.path.join(tmpdir, prot + '.fasta'), 'w') as f:
            f.write('>' + prot + '\n')
            f.write(seq_dict[prot] + '\n')
        os.system(os.path.join('./oneline_command.sh' + ' ' + os.path.join(tmpdir, prot + '.fasta') + ' ./tmp 20 1')) # PARAMS
        os.system('mv ./tmp/' + prot + ' ' + tmpdir)
        os.system('rm '+ os.path.join(tmpdir, prot + '.fasta'))
    os.chdir(cur_dir)
    
    # Step 2: Aggregate RaptorX features.
    raptorx_dict = {}
    for feat in ['SS_H_prob', 'SS_E_prob', 'SS_C_prob', 'ACC_B_prob', 'ACC_M_prob', 'ACC_E_prob']:
        raptorx_dict[feat] = {}
    for prot in seq_dict:
        prot_length = len(seq_dict[prot])
        if os.path.exists(os.path.join(tmpdir, prot, '%s.ss3' % prot)):
            ss3_df = pd.read_csv(os.path.join(tmpdir, prot, '%s.ss3' % prot), skiprows=2, sep='\s+', header=None)
            ss3_length = ss3_df[3].values.shape[0]
            if prot_length > ss3_length:
                nan_array = np.empty((prot_length - ss3_length))
                nan_array[:] = np.nan
                raptorx_dict['SS_H_prob'][prot] = np.concatenate((ss3_df[3].values, nan_array), axis = None)
                raptorx_dict['SS_E_prob'][prot] = np.concatenate((ss3_df[4].values, nan_array), axis = None)
                raptorx_dict['SS_C_prob'][prot] = np.concatenate((ss3_df[5].values, nan_array), axis = None)
            else:
                raptorx_dict['SS_H_prob'][prot] = ss3_df[3].values
                raptorx_dict['SS_E_prob'][prot] = ss3_df[4].values
                raptorx_dict['SS_C_prob'][prot] = ss3_df[5].values
        if os.path.exists(os.path.join(tmpdir, prot, '%s.acc' % prot)):
            try:
                acc_df = pd.read_csv(os.path.join(tmpdir, prot, '%s.acc' % prot), skiprows=3, sep='\s+', header=None)
                raptorx_dict['ACC_B_prob'][prot] = acc_df[3].values
                raptorx_dict['ACC_M_prob'][prot] = acc_df[4].values
                raptorx_dict['ACC_E_prob'][prot] = acc_df[5].values
            except:
                nan_array = np.empty((prot_length))
                nan_array[:] = np.nan
                raptorx_dict['ACC_B_prob'][prot] = nan_array
                raptorx_dict['ACC_M_prob'][prot] = nan_array
                raptorx_dict['ACC_E_prob'][prot] = nan_array
    #shutil.rmtree(tmpdir)
    return raptorx_dict

def calculate_pair_potential(int_list, seq_dict):
    """
    Calculate pair potential features.
    
    Args:
        int_list (list): A list of all interactions (tuple of two identifiers) to calculate coevolution features for.
        seq_dict (dict): Dictionary mapping protein identifiers to sequences.
        
    Returns:
        A tuple two dictionaries. The first maps (id1, id2) -> (pp_1, pp_2) and the second maps (id1, id2) ->
        (pp_normed_1, pp_normed_2).
    
    """
    # Load pair potential background matrices
    g_homo = np.load(pkg_resources.resource_filename(__name__, 'data/G_homo.npy'))
    g_hetero = np.load(pkg_resources.resource_filename(__name__, 'data/G_hetero.npy'))
    
    # Step 1: Calculate pair potential features if they do not exist in the cache.
    pp = PairPotential()
    for id1, id2 in int_list:
        if id1 == id2:
            if os.path.exists(os.path.join(PAIRPOTENTIAL_CACHE, '%s_%s_0.pkl' % (id1, id2))):
                continue
            pp_array = pp.get_feature(seq_dict[id1], seq_dict[id2], g_homo)
            pp_normed = normalize_feat(pp_array) # Yilin Liu found it. Dongjin edited it.
            data = {'pair_potential': pp_array, 'pair_potential_norm': pp_normed}
            pd.DataFrame(data).to_pickle(os.path.join(PAIRPOTENTIAL_CACHE, '%s_%s_0.pkl' % (id1, id2)))
        else:
            if os.path.exists(os.path.join(PAIRPOTENTIAL_CACHE, '%s_%s_0.pkl' % (id1, id2))) and os.path.exists(os.path.join(PAIRPOTENTIAL_CACHE, '%s_%s_1.pkl' % (id1, id2))):
                continue
            # For protein 1
            pp_array = pp.get_feature(seq_dict[id1], seq_dict[id2], g_hetero)
            pp_normed = normalize_feat(pp_array)
            data = {'pair_potential': pp_array, 'pair_potential_norm': pp_normed}
            pd.DataFrame(data).to_pickle(os.path.join(PAIRPOTENTIAL_CACHE, '%s_%s_0.pkl' % (id1, id2)))
            # For protein 2
            pp_array = pp.get_feature(seq_dict[id2], seq_dict[id1], g_hetero)
            pp_normed = normalize_feat(pp_array)
            data = {'pair_potential': pp_array, 'pair_potential_norm': pp_normed}
            pd.DataFrame(data).to_pickle(os.path.join(PAIRPOTENTIAL_CACHE, '%s_%s_1.pkl' % (id1, id2)))
    
    # Step 2: Fetch all pair potential features.
    pp_dict, pp_norm_dict = {}, {}
    for id1, id2 in int_list:
        if id1 == id2:
            if os.path.exists(os.path.join(PAIRPOTENTIAL_CACHE, '%s_%s_0.pkl' % (id1, id2))):
                df = pd.read_pickle(os.path.join(PAIRPOTENTIAL_CACHE, '%s_%s_0.pkl' % (id1, id2)))
                pp_dict[(id1, id2)] = (df['pair_potential'].values, df['pair_potential'].values)
                pp_norm_dict[(id1, id2)] = (df['pair_potential_norm'].values, df['pair_potential_norm'].values)
        else:
            if os.path.exists(os.path.join(PAIRPOTENTIAL_CACHE, '%s_%s_0.pkl' % (id1, id2))) and os.path.exists(os.path.join(PAIRPOTENTIAL_CACHE, '%s_%s_1.pkl' % (id1, id2))):
                df1 = pd.read_pickle(os.path.join(PAIRPOTENTIAL_CACHE, '%s_%s_0.pkl' % (id1, id2)))
                df2 = pd.read_pickle(os.path.join(PAIRPOTENTIAL_CACHE, '%s_%s_1.pkl' % (id1, id2)))
                pp_dict[(id1, id2)] = (df1['pair_potential'].values, df2['pair_potential'].values)
                pp_norm_dict[(id1, id2)] = (df1['pair_potential_norm'].values, df2['pair_potential_norm'].values)
    
    return pp_dict, pp_norm_dict
