# FinPileTokenizers

A collection of tools to tokenize text for pretraining.

## Usage

`fsiltok --input [file-or-folder] --prefix tokenized_data --tokenizer EleutherAI/gpt-neox-20b --threads 8`

See `-h` for all instructions.


The generated artifacts can be loaded using the `fsiltok.utils.data.DocumentTapeDataset` class:

```
from fsiltok.utils.data import DocumentTapeDataset

dataset = DocumentTapeDataset(
    prefix="prefix",
    token_dtype=np.uint16,
    chunk_size=4096,
    eod_token_id=0
)

dataloader = DocumentTapeDataset.get_dataloader(
    dataset,
    batch_size=2,
    pin_memory=True,
    ...,
)

for batch in dataloader:
    ...
```
