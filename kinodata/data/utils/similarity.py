import itertools as itr
from typing import Sequence

import numpy as np
import pandas as pd
from rdkit.Chem import AllChem, MolFromSmiles
from rdkit.DataStructs.cDataStructs import BulkTanimotoSimilarity
from tqdm import tqdm

try:
    from biotite.sequence.align import SubstitutionMatrix
except ImportError:
    SubstitutionMatrix = None


def _normalize_score_matrix(matr: np.ndarray) -> np.ndarray:
    """Normalize a square score matrix `X` such that all entries are in `[0,1]`
    and diagonal entries are equal to 1:

    `X' = X - min(X)`
    `X''[i,j] = X'[i,j] / sqrt(X'[i,i] * X'[j,j])`

    Parameters
    ----------
    matr : np.ndarray
        A score square matrix `X`.

    Returns
    -------
    np.ndarray
        Normalized matrix `X''` as described above.
    """
    matr = matr - matr.min()

    diag = np.diag(matr)
    norm = np.sqrt(diag[:, None] * diag[None, :])
    matr = matr / norm
    return matr


def _translate_rescale_substitution_matrix(
    substitution_matrix: SubstitutionMatrix,
) -> pd.DataFrame:
    """
    Translate and rescale substitution matrix.

    Parameters
    ----------
    substitution_matrix : biotite.sequence.align.SubstitutionMatrix
        A substitution matrix specific to amino acids.
        The default is align.SubstitutionMatrix.std_protein_matrix()
        from biotite, which represents BLOSUM62.

    Returns
    -------
    pd.DataFrame
        Translated and rescaled substitution matrix as DataFrame
        (index/columns contain letters).
    """
    # Retrieve np.array from substitution matrix
    score_matrix = substitution_matrix.score_matrix()
    normalized_score_matrix = _normalize_score_matrix(score_matrix)

    # Create DataFrame from matrix with letters as index/column names
    normalized_score_matrix = pd.DataFrame(
        normalized_score_matrix,
        columns=substitution_matrix.get_alphabet1(),
        index=substitution_matrix.get_alphabet1(),
    )

    # Check for symmetry
    symmetric = (
        normalized_score_matrix.values == normalized_score_matrix.values.transpose()
    ).all()
    if not symmetric:
        raise ValueError(f"Translated/rescaled matrix is not symmetric.")

    return normalized_score_matrix


def substitution_score(
    sequence1,
    sequence2,
    substitution_matrix_df,
) -> np.ndarray:
    """
    Retrieve the match score given the substitution matrix.

    Parameters
    ----------
    sequence1 : np.array
        An array of characters describing the first sequence.
    sequence2 : np.array
        An array of characters describing the second sequence.
    substitution_matrix_df :
       Data frame substitution matrix specific to amino acids.

    Returns
    -------
    np.array
        The vector of match score
        using the normalized substitution matrix.
    """
    match_score_array = np.zeros(len(sequence1))
    for i, (character_seq1, character_seq2) in enumerate(zip(sequence1, sequence2)):
        match_score_array[i] = substitution_matrix_df.loc[
            character_seq1, character_seq2
        ]
    return match_score_array


def sequence_similarity(sequence_1, sequence_2, substitution_matrix_df) -> float:
    """
    Compares two sequences using a given metric.

    Parameters
    ----------
    sequence_1, sequence_2 : str
        The two sequences of strings for comparison.
    substitution_matrix_df :
       Data frame substitution matrix specific to amino acids.


    Returns
    -------
    float :
        The similarity between the pocket sequences of the two kinases.
    """

    # Replace possible unavailable residue
    # noted in KLIFS with "-"
    # by the symbol "*" for biotite
    sequence_1 = sequence_1.replace("_", "*")
    sequence_2 = sequence_2.replace("_", "*")

    if len(sequence_1) != len(sequence_1):
        raise ValueError(f"Mismatch in sequence lengths.")
    else:
        seq_array1 = np.array(list(sequence_1))
        seq_array2 = np.array(list(sequence_2))

        match_score_array = substitution_score(
            seq_array1, seq_array2, substitution_matrix_df
        )
        similarity_normed = sum(match_score_array) / len(sequence_1)

        return similarity_normed


class BLOSUMSubstitutionSimilarity:
    """
    Computes BLOSUM substitution similarity given protein sequences.
    See TeachopenCADD T028.
    """

    def __init__(self) -> None:
        if SubstitutionMatrix is None:
            raise RuntimeError(
                "Biotite package was not imported, consider installing it."
            )
        self.subst_matr_df = _translate_rescale_substitution_matrix(
            SubstitutionMatrix.std_protein_matrix()
        )

    def __call__(self, sequences: Sequence[str]) -> np.ndarray:
        # initialize such that diagonal (self-similarity) is 1
        similarities = np.eye(len(sequences))
        # only computer upper/lower triangle
        total = (len(sequences) * (len(sequences) - 1)) // 2
        print(f"Computing {total} sequence similarities...")
        pbar = tqdm(itr.combinations(enumerate(sequences), 2), total=total)
        for (i, seq_i), (j, seq_j) in pbar:
            sim_ij = sequence_similarity(seq_i, seq_j, self.subst_matr_df)
            similarities[i, j] = sim_ij
            similarities[j, i] = sim_ij
        return similarities


def pairwise_tanimoto_similarity(
    smiles: Sequence[str], nbits: int = 1028
) -> np.ndarray:
    mols = [MolFromSmiles(sm) for sm in smiles]
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nbits) for m in mols]
    similarities = np.eye(len(smiles))
    for i, fp in enumerate(fps):
        similarities_i = BulkTanimotoSimilarity(fp, fps[i:])
        similarities[i, i:] = similarities_i
        similarities[i:, i] = similarities_i
    return similarities


if __name__ == "__main__":
    seq = [
        "KPLGR____QVIEVAVKMLALMSELKILIHIGLNVVNLLGAMVIVEFCKFGNLSTYLRSFLASRKCIHRDLAARNILLICDF___",
        "VKLGQG__GEVWMVAIKTLAFLQEAQVMKKLREKLVQLYAVYIVTEYMNKGSLLDFLKGYVERMNYVHRDLRAANILVVAD____",
        "VKLGQGCFGEVWMVAIKTLAFLQEAQVMKKLREKLVQLYAVYIVGEYMSKGSLLDFLKGYVERMNYVHRDLRAANILVVADFGLA",
        "KPLGR____QVIEVAVKMLALMSELKILIHIGLNVVNLLGAMVIVEFCKFGNLSTYLRSFLASRKCIHRDLAARNILLICDF___",
        "KVLGSGAFGTVYKVAIKELEILDEAYVMASVDPHVCRLLGIQLITQLMPFGCLLDYVREYLEDRRLVHRDLAARNVLVITDFGLA",
        "KPLGR____QVIEVAVKMLALMSELKILIHIGLNVVNLLGAMVIVEFCKFGNLSTYLRSFLASRKCIHRDLAARNILLICDF___",
        "KPLGR____QVIEVAVKMLALMSELKILIHIGLNVVNLLGAMVIVEFCKFGNLSTYLRSFLASRKCIHRDLAARNILLICDF___",
        "VKLGQGCFGEVWMVAIKTLAFLQEAQVMKKLREKLVQLYAVYIVGEYMSKGSLLDFLKGYVERMNYVHRDLRAANILVVADFGLA",
        "KPLGR____QVIEVAVKMLALMSELKILIHIGLNVVNLLGAMVIVEFCKFGNLSTYLRSFLASRKCIHRDLAARNILLICDF___",
        "VKLGQG__GEVWMVAIKTLAFLQEAQVMKKLREKLVQLYAVYIVTEYMNKGSLLDFLKGYVERMNYVHRDLRAANILVVAD____",
    ]

    sim = BLOSUMSubstitutionSimilarity()(seq)
    print(seq[3], seq[4])
    print(sim[3, 4], sim[4, 3])
