import unittest

import numpy as np
from fsiltok.utils.data import DocumentTapeDataset

class TestDataUtils(unittest.TestCase):
    
    def test_initialization(self):
        # Test if the dataset can be initialized correctly
        
        dataset = DocumentTapeDataset(prefix="test", token_dtype=np.uint16, chunk_size=4096, eod_token_id=0)
        self.assertIsNotNone(dataset)
        self.assertEqual(dataset._chunk_size, 4096)
        self.assertEqual(dataset._token_dtype, np.uint16)

    def test_get_item(self):

        # Test if we can retrieve an item from the dataset
        dataset = DocumentTapeDataset(prefix="test", token_dtype=np.uint16, chunk_size=4096, eod_token_id=0)
        item = dataset[0]
        self.assertIsInstance(item, dict)
        self.assertTrue('input_ids' in item)
        self.assertTrue('position_ids' in item)
        self.assertTrue('mask' in item)

    def test_dataloader(self):
        # Test if the DataLoader can be created and iterated
        dataset = DocumentTapeDataset(prefix="test", token_dtype=np.uint16, chunk_size=4096, eod_token_id=0)
        dataloader = DocumentTapeDataset.get_dataloader(
            dataset,
            batch_size=32
        )

        self.assertIsNotNone(dataloader)

        for _ in dataloader:
            pass

