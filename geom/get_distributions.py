from tqdm import tqdm
from geom.data import GeomDataModule
from collections import defaultdict
import pickle
import click

@click.command(
    help="Getting the dataset distributions"
)
@click.option("--dataset", "-d", default="drugs")
@click.option("--batch_size", "-b", default=256)
@click.option("--max_num_conformers", "-m", default=30)
@click.option("--num_workers", "-n", default=4)
@click.option("--pin_memory", "-pm", default=True)
@click.option("--persistent_workers", "-pw", default=True)
def main(batch_size: int = 256, num_workers: int = 4,
         dataset: str = "drugs", max_num_conformers: int = 30,
         pin_memory: bool = True, persistent_workers: bool = True):
    datamodule = GeomDataModule(
        batch_size=batch_size,
        num_workers=num_workers,
        dataset=dataset,
        env_in_init=True,
        shuffle_train=False,
        max_num_conformers=max_num_conformers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    datamodule.setup()

    num_nodes_dict = defaultdict(int)
    atom_types_dict = defaultdict(int)

    print(f"Train set size {len(datamodule.train_dataset)}")
    print(f"Val set size {len(datamodule.val_dataset)}")
    print(f"Test set size {len(datamodule.test_dataset)}")
    # data = datamodule.train_dataset[0]
    # print(data)
    loader = datamodule.train_dataloader(shuffle=False)
    print("Iterating....")
    print(f"Length dataloader = {len(loader)}")
    for i, batch in tqdm(enumerate(loader), total=len(loader)):
        num_atoms = batch.batch.bincount().tolist()
        for n in num_atoms:
            num_nodes_dict[n] += 1
        atom_types = batch.xgeom.tolist()
        for a in atom_types:
            atom_types_dict[a] += 1
        if i % 1000 == 0:
            print(f"{i}/{len(loader)}")

    with open(f'{dataset}_num_nodes.pickle', 'wb') as f:
        pickle.dump(num_nodes_dict, f)

    with open(f'{dataset}_atom_types.pickle', 'wb') as f:
        pickle.dump(atom_types_dict, f)
        
        
if __name__ == "__main__":
    main()
    