import torch

from uspto.molbart import BARTModel
from uspto.tokeniser import MolEncTokeniser


class ChemformerEmbedding:
    REGEX = "\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9]"
    DEFAULT_CHEM_TOKEN_START = 272

    def __init__(
        self,
        model_path: str,
        vocab_path: str,
        chem_token_start_idx: int = DEFAULT_CHEM_TOKEN_START,
    ):
        self.tokeniser = MolEncTokeniser.from_vocab_file(
            vocab_path, self.REGEX, chem_token_start_idx
        )
        self.model = BARTModel.load_from_checkpoint(model_path, decode_sampler=None)
        self.model.eval()
        self.model.freeze()

    def encode_smiles(self, smi: str):
        tokenised_smi = self.tokeniser.tokenise([smi])["original_tokens"]
        enc_token_ids = self.tokeniser.convert_tokens_to_ids(tokenised_smi)
        enc_token_ids = torch.tensor(enc_token_ids).transpose(0, 1)
        embs = self.model._construct_input(enc_token_ids)
        embs = self.model.encoder(embs, src_key_padding_mask=None)
        return embs.sum(dim=0)


if __name__ == "__main__":
    f = ChemformerEmbedding(
        model_path="fined-tuned/uspto_mixed/last.ckpt",
        vocab_path="vocabs/bart_vocab.txt",
    )
    smi = "COC(=O)[C@H](CCCCNC(=O)OCc1ccccc1)NC(=O)Nc1cc(OC)cc(C(C)(C)C)c1O"
    print(f.encode_smiles(smi))
