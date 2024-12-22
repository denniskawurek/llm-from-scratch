import torch
from instruction.instruction_data import InstructionDataset
from torch.utils.data import DataLoader

def create_dataloader(data, tokenizer, batch_size, customized_collate_fn, num_workers):
    torch.manual_seed(123)
    train_dataset = InstructionDataset(data, tokenizer)
    return DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers
    )