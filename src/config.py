###### MODIFY ######
SAVING_DIR = "/scratch/home/"
ARCHITECTURE = 'dfanet'
DATASET = 'SECOND'
ROOT_DIRECTORY = f"{SAVING_DIR}/SECOND-dataset-tif-split"

NUM_WORKERS = 8
BATCH_SIZE = 2 

########## DO NOT MODIFY BELOW ##########
CD_DIR = "cd2_Output"
NUM_EPOCHS = 150
CLASSES = [
    'no_change','lowVeg_to_Nvgsurface', 'lowVeg_to_tree', 'lowVeg_to_water',
    'lowVeg_to_buildings', 'lowVeg_to_playground','Nvgsurface_to_lowVeg',
    'Nvgsurface_to_tree','Nvgsurface_to_water','Nvgsurface_to_buildings',
    'Nvgsurface_to_playground','tree_to_lowVeg','tree_to_Nvgsurface',
    'tree_to_water','tree_to_buildings','tree_to_playground',
    'water_to_lowVeg','water_to_Nvgsurface','water_to_tree','water_to_buildings',
    'water_to_playground','buildings_to_lowVeg','buildings_to_Nvgsurface',
    'buildings_to_tree','buildings_to_water','buildings_to_playground',
    'playground_to_lowVeg','playground_to_Nvgsurface','playground_to_tree',
    'playground_to_water','playground_to_buildings']
SEMANTIC_CLASSES = ['no_change', 'lowVeg', 'Nvgsurface', 'tree',
                    'water', 'buildings', 'playground']