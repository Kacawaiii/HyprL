from decimal import localcontext

import pytest

from scripts.trading_lab.platform.datasets import synthetic_dataset


@pytest.fixture(scope="session")
def dataset():
    with localcontext() as context:
        context.prec = 34
        return synthetic_dataset(products=["BTC-USD"], bars=120)
