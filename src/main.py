from config import *
from dataset.dataloader import ChangeDetectionDatasetTIF, describe_loader
from torch.utils.data import Dataset, DataLoader
import torch
from train import train_model, save_training_history
from models.dfanet import DFANet
from test import test_model_pcc, save_test_metrics

def main():
    # Create datasets
    train_dataset = ChangeDetectionDatasetTIF(
        t2019_dir=f"{ROOT_DIRECTORY}/train/Images/T2019",
        t2024_dir=f"{ROOT_DIRECTORY}/train/Images/T2024",
        sem_2019_dir=f"{ROOT_DIRECTORY}/train/Masks/T2019",
        sem_2024_dir=f"{ROOT_DIRECTORY}/train/Masks/T2024",
        mask_dir=f"{ROOT_DIRECTORY}/train/{CD_DIR}",
        classes=CLASSES,
        semantic_classes=SEMANTIC_CLASSES
    )

    val_dataset = ChangeDetectionDatasetTIF(
        t2019_dir=f"{ROOT_DIRECTORY}/val/Images/T2019",
        t2024_dir=f"{ROOT_DIRECTORY}/val/Images/T2024",
        sem_2019_dir=f"{ROOT_DIRECTORY}/val/Masks/T2019",
        sem_2024_dir=f"{ROOT_DIRECTORY}/val/Masks/T2024",
        mask_dir=f"{ROOT_DIRECTORY}/val/{CD_DIR}",
        classes=CLASSES,
        semantic_classes=SEMANTIC_CLASSES
    )

    test_dataset = ChangeDetectionDatasetTIF(
        t2019_dir=f"{ROOT_DIRECTORY}/test/Images/T2019",
        t2024_dir=f"{ROOT_DIRECTORY}/test/Images/T2024",
        sem_2019_dir=f"{ROOT_DIRECTORY}/test/Masks/T2019",
        sem_2024_dir=f"{ROOT_DIRECTORY}/test/Masks/T2024",
        mask_dir=f"{ROOT_DIRECTORY}/test/{CD_DIR}",
        classes=CLASSES,
        semantic_classes=SEMANTIC_CLASSES
    )

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,num_workers=NUM_WORKERS, pin_memory=True)

    print("------------Train------------")
    describe_loader(train_loader)
    print("------------Val------------")
    describe_loader(val_loader)
    print("------------Test------------")
    describe_loader(test_loader)

    ## Model train

    # Initialize model and device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_cd_classes = len(CLASSES)   # Change Detection classes (3 for cd1, 13 for cd2)
    num_semantic_classes = len(SEMANTIC_CLASSES)   # Semantic segmentation LCM classes (4 for both)
    weighted_metrics = True if num_cd_classes > 5 else False   #True for 13 classes,False for 3 classes

    name = f'{ARCHITECTURE}-{num_cd_classes}_classes_{NUM_EPOCHS}_{DATASET}'
    checkpoint_path = f'{SAVING_DIR}/best_{name}_epochs.pt'

    if ARCHITECTURE == 'dfanet':
        model = DFANet(input_nbr=3,sem_classes=num_semantic_classes,
                cd_classes=num_cd_classes,with_softmax=True,base_depth=16).to(device)

    # Train model
    model2, history = train_model(model=model,train_loader=train_loader,val_loader=val_loader,
                                    num_epochs=NUM_EPOCHS,device=device,checkpoint_path=checkpoint_path,
                                    num_cd_classes=num_cd_classes,
                                    weighted_metrics=weighted_metrics)

    #Save model files
    save_history_path=f'{SAVING_DIR}/{name}_history.json'
    save_bestepoch_path=f'{SAVING_DIR}/{name}_best_epoch.json'
    save_training_history(history, checkpoint_path=checkpoint_path,
                        save_path=save_history_path, save_path_bestepoch=save_bestepoch_path)
    
    # Test model with computational metrics
    test_metrics = test_model_pcc(model, test_loader, checkpoint_path=checkpoint_path,
                              device=device, num_samples_to_plot=3, num_cd_classes=num_cd_classes,
                              weighted_metrics=weighted_metrics)

    # Save test metrics (now including computational metrics)
    save_test_metrics(test_metrics, save_path=f'{SAVING_DIR}/{name}_test_metrics.json')

if __name__ == '__main__':
    main()