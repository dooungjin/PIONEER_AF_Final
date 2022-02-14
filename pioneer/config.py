# Binaries # # COPIED!
PSIBLAST = '/local/storage/sl2678/applications/ncbi-blast-2.11.0+/bin/psiblast'
CLUSTALO = '/local/storage/sl2678/applications/clustalo'
BLASTP = '/local/storage/sl2678/applications/ncbi-blast-2.11.0+/bin/blastp'
MAKEBLASTDB = '/local/storage/sl2678/applications/ncbi-blast-2.11.0+/bin/makeblastdb'
NACCESS = '/local/storage/sl2678/interface_pred/Deep-Interface-Pipeline/bin/naccess2.1.1/naccess'
ZDOCK = '/local/storage/sl2678/interface_pred/Deep-Interface-Pipeline/bin/zdock/zdock3.0.2'
CREATE_PL = '/local/storage/sl2678/interface_pred/Deep-Interface-Pipeline/bin/zdock/create.pl'
CREATE_LIG = '/local/storage/sl2678/interface_pred/Deep-Interface-Pipeline/bin/zdock/create_lig'

# Application directories # # COPIED!
RAPTORX = '/local/storage/sl2678/applications/RaptorX_Property_Fast/'

# Data files #
UNIPROT_ALL_DB = '/local/storage/dx38/local_resource/uniprot_db_v20200226/uniprot_all.fasta'
UNIPROT_SEQ_PICKLE_PATH = '/local/storage/sl2678/interface_pred/pipeline/pipeline_data/uniprot_seq_dict.pkl'
UNIPROT_LENGTH_PICKLE_PATH = '/local/storage/sl2678/interface_pred/pipeline/pipeline_data/uniprot_length_dict.pkl'

# Cache directories #
# Use this set of paths to test the pipeline with existing cache files.
JS_CACHE = '/local/storage/sl2678/interface_pred/pipeline/pipeline_data/pipeline_js_div/'
SCA_CACHE = '/local/storage/sl2678/interface_pred/pipeline/pipeline_data/SCA/'
DCA_CACHE = '/local/storage/sl2678/interface_pred/pipeline/pipeline_data/DCA/'
MSA_CACHE = '/local/storage/sl2678/interface_pred/pipeline/empty_cache/msa_cache/'
MODBASE_CACHE = '/local/storage/sl2678/interface_pred/pipeline/pipeline_data/modbase/'
ZDOCK_CACHE = '/local/storage/sl2678/interface_pred/pipeline/pipeline_data/zdock/docked_models/'
PAIRPOTENTIAL_CACHE = '/local/storage/sl2678/interface_pred/pipeline/empty_cache/pairpotential_cache/'
RAPTORX_CACHE = '/fs/cbsuhyfs1/storage/dl953/large_scale_prediction/raptorx_cache/'
DISTANCE_MATRIX_CACHE = '/fs/cbsuhyfs1/storage/dl953/large_scale_prediction/distance_matrix/'

"""
# Use this set of paths to test the pipeline without existing cache files (except data that is necessary).
JS_CACHE = '/local/storage/sl2678/interface_pred/pipeline/empty_cache/js_cache/'
SCA_CACHE = '/local/storage/sl2678/interface_pred/pipeline/empty_cache/sca_cache/'
DCA_CACHE = '/local/storage/sl2678/interface_pred/pipeline/empty_cache/dca_cache/'
MSA_CACHE = '/local/storage/sl2678/interface_pred/pipeline/empty_cache/msa_cache/'
MODBASE_CACHE = '/local/storage/sl2678/interface_pred/pipeline/empty_cache/modbase_cache/'
ZDOCK_CACHE = '/local/storage/sl2678/interface_pred/pipeline/empty_cache/zdock_cache/'
PAIRPOTENTIAL_CACHE = '/local/storage/sl2678/interface_pred/pipeline/empty_cache/pairpotential_cache/'
DISTANCE_MATRIX_CACHE = '/local/storage/sl2678/interface_pred/pipeline/empty_cache/dm_cache/'
"""

# Data directories and files # 
PDB_DATA_DIR = '/fs/cbsuhyfs1/storage/resources/pdb/data/'
SASA_PDB_PATH = '/local/storage/sl2678/interface_pred/data/ires/SASA_perpdb_alltax.txt'
PDBRESMAP_PATH = '/local/storage/sl2678/interface_pred/data/sifts/pdbresiduemapping.txt'
