from tqdm import tqdm
from geom.data import GeomDataModule

if __name__ == "__main__":
    print("Initializing Dataset")
    datamodule = GeomDataModule(
       batch_size=128,
       num_workers=4,
       dataset="drugs",
       env_in_init=True,
       shuffle_train=True,
       subset_frac=0.1,
       pin_memory=True,
       persistent_workers=True,
    )
    datamodule.setup()
     
    print(f"Train set size {len(datamodule.train_dataset)}")
    print(f"Val set size {len(datamodule.val_dataset)}")
    print(f"Test set size {len(datamodule.test_dataset)}")
    data = datamodule.train_dataset[0]
    loader = iter(datamodule.train_dataloader(shuffle=True))
    data = next(loader)

    loader = datamodule.val_dataloader(shuffle=True)
    for data in tqdm(loader, total=len(loader)):
       pass
   