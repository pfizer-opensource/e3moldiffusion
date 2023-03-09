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
def main(dataset: str = "drugs", batch_size: int = 256, num_workers: int = 4,  max_num_conformers: int = 30):
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
   data = datamodule.train_dataset[0]
   loader = iter(datamodule.train_dataloader(shuffle=True))
   data = next(loader)
   
   print("Only iterating over validation loader")
   loader = datamodule.val_dataloader(shuffle=True)
   for data in tqdm(loader, total=len(loader)):
      pass
    
    
if __name__ == "__main__":
    main()