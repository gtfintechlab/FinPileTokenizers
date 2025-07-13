import numpy as np
from fsiltok.utils.data import DocumentTapeDataset
from tqdm import tqdm

if __name__ == "__main__":

    dataset = DocumentTapeDataset(
        prefix="test/test_data",
        token_dtype=np.uint16,
        chunk_size=4096,
        eod_token_id=0 # EleutherAI/gpt-neox-20b uses 0 as EOD token
    )

    dataloader = DocumentTapeDataset.get_dataloader(
        dataset,
        batch_size=2,
        pin_memory=True,
        num_workers=4
    )

    for batch in tqdm(dataloader, desc="Iterating over DataLoader", unit="batch"):
        
        assert "input_ids" in batch
        assert "position_ids" in batch
        assert "mask" in batch

        input_ids = batch["input_ids"]
        position_ids = batch["position_ids"]
        mask = batch["mask"]

        B = input_ids.shape[0]
        L = input_ids.shape[1]

        assert input_ids.shape == (B, L)
        assert position_ids.shape == (B, L)
        assert mask.shape == (B, L, L)

        # You might want to turn this mask into an additive mask

        mask = ~mask * torch.finfo(torch.float32).min

        #  Train your model here




        #  CE Loss Here

        # target = input_ids[:, 1:]  # Shifted input_ids
        # logits = logits[:, :-1, :]  # Shifted logits

        
