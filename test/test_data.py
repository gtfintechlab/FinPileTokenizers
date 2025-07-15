import os
import unittest

import numpy as np
import tempfile
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
        self.assertTrue('attention_mask' in item)

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
        self.assertTrue('attention_mask' in batch)

        B, L = batch['input_ids'].shape
        self.assertEqual(batch['input_ids'].shape, (B, L))
        self.assertEqual(batch['position_ids'].shape, (B, L))
        self.assertEqual(batch['attention_mask'].shape, (B, L, L))

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
        
    def test_attention_attention_mask(self):
        # Test if the attention attention_mask is created correctly
        dataset = DocumentTapeDataset(prefix="test/test_data", token_dtype=np.uint16, chunk_size=4096, eod_token_id=0)
        item = dataset[0]
        input_ids = item['input_ids']
        attention_mask = item['attention_mask']

        seq_len = 0
        start = 0
        for index, token in enumerate(input_ids):
            seq_len += 1
            self.assertTrue(attention_mask[index, start:start+seq_len].all())
            self.assertFalse(attention_mask[index, start+seq_len:].any())
            if start != 0:
               self.assertFalse(attention_mask[index, :start].any())
            if token == 0:  # Reset condition
                start = index + 1
                seq_len = 0

    def test_eod_as_start_token(self):
        # Test if EOD token is treated correctly as a start token
        tmp_path = tempfile.mkdtemp()
        data = np.memmap(
            os.path.join(tmp_path, "test_data.bin"),
            dtype=np.uint16,
            mode='w+',
            shape=(4096 * 10,)
        )
        idx = np.memmap(
            os.path.join(tmp_path, "test_data.idx"),
            dtype=np.uint32,
            mode='w+',
            shape=(10,)
        )
        idx = np.arange(0, 10)
        data[:] = np.random.randint(1, 100, size=(4096 * 10,))
        data[0] = 0  # Set EOD token at the start

        dataset = DocumentTapeDataset(
            prefix=os.path.join(tmp_path, "test_data"),
            token_dtype=np.uint16,
            chunk_size=4096,
            eod_token_id=0
        )

        item = dataset[0]
        self.assertTrue(item['input_ids'][0] == 0)  # EOD
        self.assertTrue(item['position_ids'][0] == 0)  # Position ID
        self.assertTrue(item['position_ids'][1] == 0)  # Next token position ID

        # Clean up
        del data
        del idx
        os.remove(os.path.join(tmp_path, "test_data.bin"))
        os.remove(os.path.join(tmp_path, "test_data.idx"))