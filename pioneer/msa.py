import os
from .config import PSIBLAST, UNIPROT_ALL_DB, CLUSTALO


def generate_fasta(prot_id, sequence, out_dir):
    """
    Generate a FASTA file for a protein sequence.
    
    Args:
        prot_id (str): Identifier of the protein.
        sequence (str): Sequence of the protein.
        out_dir (str): Path to the directory to store the generated FASTA file.
        
    Returns:
        A string of the path to the generated FASTA file.
        
    """
    with open(os.path.join(out_dir, prot_id + '.fasta'), 'w') as f:
        f.write('>' + prot_id + '\n')
        f.write(sequence + '\n')
    return os.path.join(out_dir, prot_id + '.fasta')


def generate_single_msa(prot_id, sequence, out_dir):
    """
    Generate an MSA and a PSSM file for a protein sequence.
    
    Args:
        prot_id (str): Identifier of the protein.
        sequence (str): Sequence of the protein.
        out_file (str): Path to the directory to store the output MSA and PSSM files.
        
    Returns:
        A string of the path to the output MSA file.
        
    """
    generate_fasta(prot_id, sequence, out_dir)
    command = '%s -query %s -db %s -out %s -evalue 0.001 -matrix BLOSUM62 -num_iterations 3 -num_threads 6' % (PSIBLAST, os.path.join(out_dir, prot_id + '.fasta'), UNIPROT_ALL_DB, os.path.join(out_dir, prot_id + '.rawmsa'))
    os.system(command)
    os.system('rm %s' % os.path.join(out_dir, prot_id + '.fasta')) # Remove intermediary FASTA file.
    # TODO: add lines to check if output is successfully generated.
    return os.path.join(out_dir, prot_id + '.rawmsa')


def parse_fasta(fasta_file):
    """
    Parse a FASTA file. Written by Michael Meyer and copied over from `mjm_parsers.py`.
    
    Args:
        fasta_file (str): Path to the FASTA file to parse.
        
    Returns:
        A tuple of a list and a dict. The list contains all identifiers in the FASTA file in the order
        by which they show up, and the dict maps the identifiers to their protein sequence.
    
    """
    assert os.path.exists(fasta_file)
    cur_key = ''
    fasta_dict = {}
    keys_ordered = []
    for line in open(fasta_file, 'r'):
        if line[0] == '>':
            cur_key = line.strip().replace('>', '').split()[0]
            keys_ordered.append(cur_key)
            fasta_dict[cur_key] = ''
        else:
            fasta_dict[cur_key] += line.strip()	
    return keys_ordered, fasta_dict


def write_fasta(fasta_dict, fasta_file):
	'''
    Write contents of a dictionary mapping identifiers to sequences to a fasta file. Written by Michael
    Meyer and copied over from `mjm_parsers.py`.
    
    Args:
        fasta_dict (dict): Dictionary mapping identifiers to sequences.
        fasta_file (str): Path to the output FASTA file.
    
    Returns:
        None.
    
    '''
	output = open(fasta_file, 'w')
	for k, v in sorted(fasta_dict.items()):
		output.write('>%s\n%s\n' %(k, v))
	output.close()
    
    
def generate_clustal_input(prot_id, sequence, msa_file, seq_dict, out_dir):
    """
    Generate a file as input to CLUSTAL.
    
    Args:
        prot_id (str): Identifier of the protein.
        sequence (str): Sequence of the protein.
        msa_file (str): Path to the raw MSA file of the protein.
        seq_dict (dict): Dictionary mapping all UniProt identifiers to sequences.
        out_dir (str): Path to the directory to store the generated CLUSTAL input file.
        
    Returns:
        A string of the path to the generated CLUSTAL input FASTA file.
    
    """
    assert os.path.exists(msa_file)
    identifiers_to_align = set()
    with open(msa_file, 'r') as f:
        for line in f:
            if line.startswith('>'):
                tmp_id = line.strip().split()[0]
                if tmp_id.split('|')[1] != prot_id:
                    identifiers_to_align.add(tmp_id[1:])
    # Write the CLUSTAL input file in fasta format. The protein of our interest is listed first.
    with open(os.path.join(out_dir, prot_id + '_clustal_input.fasta'), 'w') as f:
        f.write('>' + prot_id + '\n' + sequence + '\n')
        for identifier in identifiers_to_align:
            f.write('>' + identifier + '\n' + seq_dict[identifier.split('|')[1]] +'\n')
    return os.path.join(out_dir, prot_id + '_clustal_input.fasta')
    
    
def run_clustal(prot_id, clustal_input_file, out_dir):
    """
    Run a protein sequence alignment with CLUSTAL Omega and generates aligned output.
    
    Args:
        prot_id (str): Identifier of the protein.
        clustal_input_file (str): Path to the CLUSTAL input file.
        out_dir (str): Path to the directory to store the CLUSTAL output file.
        
    Returns:
        A string of the path to the CLUSTAL Omega output MSA file. If IOError is encountered when loading
        the input file, return None.
    
    """
    numseqs = 0
    try:
        with open(clustal_input_file, 'r') as f:
            for i, l in enumerate(f):
                numseqs = i
        f.close()
    except IOError:
        return None
    numseqs /= 2
    output_file = os.path.join(out_dir, prot_id + '_clustal.msa')
    if numseqs > 1:
        # Run Clustal Omega
        os.system('%s -i %s -o %s --force --threads 6' % (CLUSTALO, clustal_input_file, output_file))
    else:
        # Clustal Omega will fail, copy single sequence to output file
        os.system('cp %s %s' % (clustal_input_file, output_file))
    return output_file


def format_clustal(prot_id, clustal_output_file, out_dir):
    """
    Format CLUSTAL Omega alignment results such that positions where the reference sequence is a gap
    are trimmed.
    
    Args:
        prot_id (str): Identifier of the protein.
        clustal_output_file (str): Path to the unprocessed CLUSTAL output file.
        out_dir (str): Path to the directory to store the formatted CLUSTAL output file.
        
    Returns:
        A string of the path to the formatted CLUSTAL output file. If IOError is encountered during file
        processing, return None.
    
    """
    msa_info = []
    with open(clustal_output_file, 'r') as f:
        seq_name = ''
        seq = ''
        for line in f:
            if line.startswith('>'):
                if seq_name:
                    msa_info.append(seq_name)
                    msa_info.append(seq)
                seq_name = line.strip()
                seq = ''
            else:
                seq += line.strip()
        msa_info.append(seq_name)
        msa_info.append(seq)
    # Make a temporary MSA file where the sequences are not split across multiple lines.
    clustal_oneline_file = os.path.join(out_dir, prot_id + '_clustal_oneline.msa')
    with open(clustal_oneline_file, 'w') as f:
        for line in msa_info:
            f.write(line + '\n')
        f.close()
    # Generate the formatted CLUSTAL output.
    formatted_output_file = os.path.join(out_dir, prot_id + '_clustal_aligned.msa')
    try:
        # Read clustal MSA
        with open(clustal_oneline_file, 'r') as f:
            outtxt = ''
            gaps = []
            # Iterate over each line
            for idx, line in enumerate(f):
                line = line.strip()
                # Add Header lines as they are
                if idx % 2 == 0:
                    outtxt += line
                    outtxt += '\n'
                # Special case for the first entry in the alignment
                # Find all of the gaps in the alignment since we only care about using the MSA with regard to the current UniProt
                # query. We don't care about any of the positions where the query has a gap
                elif idx == 1: # Query
                    for i in range(len(line)): # Find all the Gaps
                        gaps.append(line[i] == '-')
                # For all matches
                if idx % 2 == 1:
                    # Update the sequence by removing all of the positions that were a gap in the current UniProt alignment
                    newseq = ''
                    for i in range(len(gaps)):
                        if not gaps[i]:
                            if i < len(line):
                                newseq += line[i]
                            else:
                                newseq += '-'
                    # Write the formatted alignment sequence
                    outtxt += newseq
                    outtxt += '\n'
        # Write all of the formatted alignment lines to the final alignment output
        with open(formatted_output_file, 'w') as f:
            f.write(outtxt)
    except IOError:
        return None
    os.system('rm %s' % clustal_oneline_file)
    return formatted_output_file
    
    
def msa_join(aligned_msa_file_1, aligned_msa_file_2, out_dir, min_sequence_coverage=0.0):
    """
    Generate a joined MSA file from two single aligned MSAs.
    
    Args:
        aligned_msa_file_1 (str): Path to the first aligned MSA file.
        aligned_msa_file_2 (str): Path to the second aligned MSA file.
        out_dir (str): Path to the directory to store the output MSA file.
        min_sequence_coverage (float): Minimum coverage by single MSAs required for each protein.
        
    Returns:
        A string of the path to the output joined MSA file. If not successful, return None.
    
    """
    assert os.path.exists(aligned_msa_file_1)
    assert os.path.exists(aligned_msa_file_2)
    id1 = os.path.basename(aligned_msa_file_1).split('_')[0]
    id2 = os.path.basename(aligned_msa_file_2).split('_')[0]
    
    id1_msa, id2_msa = {}, {}
    with open(aligned_msa_file_1, 'r') as f:
        lines1 = f.readlines()
    with open(aligned_msa_file_2, 'r') as f:
        lines2 = f.readlines()
    reference_seq = lines1[1].strip() + lines2[1]
    lines1, lines2 = lines1[2:], lines2[2:]
    for i in range(0, len(lines1), 2):
        id1_msa[lines1[i].strip()[1:]] = lines1[i+1].strip()
    for i in range(0, len(lines2), 2):
        id2_msa[lines2[i].strip()[1:]] = lines2[i+1].strip()
    
    # Normalize keys by organism name AND remove last residues of each (UCSC uses terminal 'Z' residues to mean stop codons)
    for k in list(id1_msa.keys()):
        id1_msa[k.split('|')[2].split('_')[-1]] = id1_msa[k].replace('U', '-')
        del id1_msa[k]
    for k in list(id2_msa.keys()):
        id2_msa[k.split('|')[2].split('_')[-1]] = id2_msa[k].replace('U', '-')
        del id2_msa[k]
    
    # Sanity check: some MSAs could be empty
    if len(id1_msa) == 0:
        print('The MSA of %s does not contain any sequences.' % id1)
        return
    if len(id2_msa) == 0:
        print('The MSA of %s does not contain any sequences.' % id2)
        return
    
    # Begin generating the joined MSA
    joined_msa = {}
    for k in set(id1_msa.keys()).intersection(set(id2_msa.keys())):
        id1_perc_missing = id1_msa[k].count('-') / float(len(id1_msa[k]))
        id2_perc_missing = id1_msa[k].count('-') / float(len(id2_msa[k]))
        if id1_perc_missing > (1.0 - min_sequence_coverage) or id2_perc_missing > (1.0 - min_sequence_coverage):
            continue
        joined_msa[k] = id1_msa[k] + id2_msa[k]
    
    # Write output to file
    joined_msa_file = os.path.join(out_dir, '%s_%s.msa' % (id1, id2))
    if len(joined_msa) > 0:
        with open(joined_msa_file, 'w') as f:
            f.write('>' + id1 + '_' + id2 + '\n' + reference_seq)
            for k in joined_msa.keys():
                f.write('>' + k + '\n' + joined_msa[k] + '\n')
        return joined_msa_file
    else:
        return None