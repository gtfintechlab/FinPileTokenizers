import unittest

import numpy as np
from fsiltok.utils.data import DocumentTapeDataset

class TestDataUtils(unittest.TestCase):
    
    def test_initialization(self):
        # Test if the dataset can be initialized correctly
        
        dataset = DocumentTapeDataset(prefix="test/test_data", token_dtype=np.uint16, chunk_size=4096, eod_token_id=0)
        self.assertIsNotNone(dataset)
        self.assertEqual(dataset._chunk_size, 4096)
        self.assertEqual(dataset._token_dtype, np.uint16)

    def test_get_item(self):

        # Test if we can retrieve an item from the dataset
        dataset = DocumentTapeDataset(prefix="test/test_data", token_dtype=np.uint16, chunk_size=4096, eod_token_id=0)
        item = dataset[0]
        self.assertIsInstance(item, dict)
        self.assertTrue('input_ids' in item)
        self.assertTrue('position_ids' in item)
        self.assertTrue('mask' in item)

    def test_dataloader(self):
        # Test if the DataLoader can be created and iterated
        dataset = DocumentTapeDataset(prefix="test/test_data", token_dtype=np.uint16, chunk_size=4096, eod_token_id=0)
        dataloader = DocumentTapeDataset.get_dataloader(
            dataset,
            batch_size=2
        )

        self.assertIsNotNone(dataloader)

        for _ in dataloader:
            pass

        dataloader = DocumentTapeDataset.get_dataloader(
            dataset,
            batch_size=2
        )

        batch = next(iter(dataloader))

        self.assertTrue('input_ids' in batch)
        self.assertTrue('position_ids' in batch)
        self.assertTrue('mask' in batch)

        B, L = batch['input_ids'].shape
        self.assertEqual(batch['input_ids'].shape, (B, L))
        self.assertEqual(batch['position_ids'].shape, (B, L))
        self.assertEqual(batch['mask'].shape, (B, L, L))

    def test_position_ids(self):
        # Test if position IDs are calculated correctly
        dataset = DocumentTapeDataset(prefix="test/test_data", token_dtype=np.uint16, chunk_size=4096, eod_token_id=0)
        item = dataset[0]
        position_ids = item['position_ids']
        input_ids = item['input_ids']

        expected_position = 0
        for token, position in zip(input_ids, position_ids):
            self.assertEqual(position, expected_position)
            expected_position += 1
            if token == 0:
                expected_position = 0
        
    def test_attention_mask(self):
        # Test if the attention mask is created correctly
        dataset = DocumentTapeDataset(prefix="test/test_data", token_dtype=np.uint16, chunk_size=4096, eod_token_id=0)
        item = dataset[0]
        input_ids = item['input_ids']
        mask = item['mask']

        seq_len = 0
        start = 0
        for index, token in enumerate(input_ids):
            seq_len += 1
            self.assertTrue(mask[index, start:start+seq_len].all())
            if start != 0:
               self.assertFalse(mask[index, :start].all())
            if token == 0:  # Reset condition
                start = index + 1
                seq_len = 0
