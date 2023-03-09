from tqdm import tqdm
from geom.data import GeomDataModule
import click

@click.command(
    help="Dataloading debugging"
)
@click.option("--dataset", "-d", default="drugs")
@click.option("--batch_size", "-b", default=256)
@click.option("--max_num_conformers", "-m", default=30)
@click.option("--num_workers", "-n", default=4)
@click.option("--split", "-s", default="validation")
def main(dataset: str = "drugs", batch_size: int = 256, num_workers: int = 4,  max_num_conformers: int = 30, split: str = "validation"):
   print("Initializing Dataset")
   datamodule = GeomDataModule(
       batch_size=batch_size,
       num_workers=num_workers,
       dataset=dataset,
       env_in_init=True,
       shuffle_train=True,
       max_num_conformers=max_num_conformers,
       pin_memory=True,
       persistent_workers=True,
    )
   datamodule.setup()
   print(f"Train set size {len(datamodule.train_dataset)}")
   print(f"Val set size {len(datamodule.val_dataset)}")
   print(f"Test set size {len(datamodule.test_dataset)}")
   
   if split == "validation":
       loader = datamodule.val_dataloader(shuffle=False)
   elif split == "training":
       loader = datamodule.train_dataloader(shuffle=False)
   elif split == "test":
       loader = datamodule.test_dataloader(shuffle=False)
   else:
       raise ValueError 
   
   print(f"Iterating over {split} loader")
   for _ in tqdm(loader, total=len(loader)):
      pass
    
if __name__ == "__main__":
    main()