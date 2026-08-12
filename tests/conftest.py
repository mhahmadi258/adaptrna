import pytest
import torch

from rinalmo.data.alphabet import Alphabet
from tests.helpers import SEQUENCES


@pytest.fixture(scope="session")
def alphabet():
    return Alphabet()


@pytest.fixture(scope="session")
def sequences():
    return list(SEQUENCES)


@pytest.fixture
def tokens(alphabet, sequences):
    return torch.tensor(alphabet.batch_tokenize(sequences), dtype=torch.long)
