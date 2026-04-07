import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
from tqdm import tqdm
import argparse
from utils import text_to_tokens, tokens_to_text, prepare_shakspear, generate_text, generate_text_beamsearch
from transformer import Transformer


"""
Checklist for a well functioning transformer architecture
Things to keep in mind in the transformer architecture that you might make a mistake:
- Tensors and Models should be either on cpu or gpu
- The sequence of operations that need to happen after forward and during the update process are: optimizer.zero_grad() > loss.backward > optimizer.step()
- Positional encoding needs to be of the shape (1, max_len, embedding_dim), since we need to sum along the batch dimension
- Note that depending on the sequence length of the input you might have to slice part of the positional encoding x = x + pe[:,:seq_len, :]
- We need to register the positional encoding in the buffer since we need it when we save the model and later on reload it.
    It also doesn't get updated during the backpropagation process
- For the layernorm the input shape needs to be model dim
- Inside the attention layer softmax happens along the last axis
- Before returning the values from the attention layer you need to run output.transpose(1, 2).contiguous().view(B, T, self.dmodel)
- For the Key tensor you need to transpose the last two dimensions of it for Softmax(Q.KT/sqrt(dhead))
- Pay close attention to the shapes and reshapes operations happening here:
    kv = self.kv(x)
    q = self.q(x)  # shape: (B, T, dmodel)
    Q = q.view(B, T, self.nheads, self.dhead).transpose(1, 2)  # (B, nheads, T, dhead)
    kv = kv.view(B, -1, self.nheads, 2 * self.dhead).transpose(1, 2)  # (B, nheads, T, 2*dhead)
    K, V = torch.chunk(kv, 2, dim=-1)
    attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.dhead)
    attn_weights = self.softmax(attention_scores)
    output = torch.matmul(attn_weights, V)  # (B, nheads, T, dhead)
    output = output.transpose(1, 2).contiguous().view(B, T, self.dmodel)  # (B, T, dmodel)
- The result of Q.KT needs to get divided by sqrt(dhead) before running the softmax
- Make sure that output projections and MLP are using the right dimensions.
- Note that at the end of attention block, we have output projection with the shape dmodel x dmodel
- Towards the end of individual transformer blocks we have MLP which consists of: Linear + Gelu + Linear with dmodel x 4dmodel and  4dmodel x dmodel
- For crossentropyloss(output, target), the output(dtype float) shape is (B , Label) and target(dtype int) is integers with the shape (B)
- For BCEWithLogitsLoss(output, target), the output(dtype float) shape is (B ) and target(dtype float) is float with the shape (B)
- No argmax before passing the output of the model to the cross entropy loss
- Note that if you are using KV caching, sometimes you might have to reset the cache at the beginning of next batch.
- Note that output logits get divided by the temperature. Higher temperature leads to a more smoothed distribution.
- GELU() is typically better for transformers (used in GPT, BERT) due to smoother gradient flow.
    ReLU() may cause dead neurons or worse training convergence.
- While we have not implemented gradient checkpointing. However, it can be useful for really deep models in order to save memory.
    Instead of storing all intermediate activations during the forward pass, we only store a subset of them and compute the missing
    ones during the backward pass.
    When to use gradient checkpointing:
        - The model is deep
        - Training with long sequences or large batch sizes and running out of memory
        - When using FP32 precision and can't fit the model in GPU memory otherwise
    Requirements are:
        - The function/module must be pure (i.e., deterministic, no side effects like modifying cache).
        - Inputs must be tensors (no booleans or non-tensor args unless wrapped).
        - You'll trade compute for memory: longer training time but lower GPU memory use.
    
    In pytorch, you can utilize it via:

    import torch.utils.checkpoint as cp
    def forward(self, x):
        x = cp.checkpoint(self.decoders[0], x, causal_mask, padding_mask)
"""
parser = argparse.ArgumentParser(description="Training.")
parser.add_argument("--train", action="store_true", help="Training")
parser.add_argument("--eval", action="store_true", help="Eval Phase")
args = parser.parse_args()
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("device = ", device)
seq_len = 40
batch_size = 32
d_model = 512
nhead = 8
num_layers = 6
learning_rate = 5e-4
num_epochs = 10
inputdata, vocab_size, str_to_id, id_to_str = prepare_shakspear(input_file='tinyshakespeare.txt',max_seq_len=seq_len)
PAD_ID = str_to_id["<pad>"] # 0
START_ID = str_to_id["<start>"] # 1
END_ID = str_to_id["<end>"] # 2

class SimpleDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn_dynamic(batch, padding_value=PAD_ID):
    max_len = max(len(x) for x in batch)
    padded_batch = []
    for item in batch:
        if len(item) > max_len:
            # Remove existing <end> if present at cutoff, then add one at the very end
            trimmed = item[:max_len]
            if trimmed[-1] != END_ID:  # 2 is <end>
                trimmed[-1] = END_ID
            padded_item = torch.tensor(trimmed)
        else:
            item = torch.tensor(item)
            pad_size = max_len - len(item)
            padded_item = torch.cat([item, torch.full((pad_size,), padding_value, dtype=item.dtype)])
        padded_batch.append(padded_item)
    
    return torch.stack(padded_batch)


# TODO: Fix the dataset, the loss is not going down low enough
inputdata = inputdata
dataset = SimpleDataset(inputdata)
dataloader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        collate_fn=lambda x: collate_fn_dynamic(x, padding_value=PAD_ID))

model = Transformer(vocabsize=vocab_size, dmodel=d_model, nheads=nhead, nlayers=num_layers)
model.to(device)
num_params = sum(p.numel() for p in model.parameters())
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# Add after optimizer definition
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, 
    max_lr=learning_rate,
    steps_per_epoch=len(dataloader),
    epochs=num_epochs
)
# adding label smoothing ε is the smoothing factor (e.g. 0.1),
# For target class i, y_i=1−ε, For all other classes j, y_j=ε/(K-1)
# the implementation of cross entropy loss is something like below:
'''
def crossentropy(logits, targets):
    # logits are directly coming from the output of the dense layer
    # target 
    # logit shape is (B x Labels)
    # targets shape is (Labels,) and it includes class indices
    log_probs = F.log_softmax(logits, dim=1)
    # note that log_probs[range(logits.size(0)),targets] is different from 
    # log_probs[:,targets], the latter extract the whole column instead of
    # individual elements
    loss = -log_probs[range(logits.shape[0]),targets]
    loss = loss.mean()
def BCEWithLogitsLoss(logits, targets):
    # logits shape is [B] or [Bx1]
    # target shape is float labels [B] or [Bx1]
    # note that logits and targets need to be exactly same shape
    # logit is also float and not integers unlike corssentropy function above
    prob = torch.sigmoid(logits)
    loss = F.binary_cross_entropy(prob, target)
'''
loss_fn = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.02)
print('vocab_size = ', vocab_size)
print("number of model parameters = ", num_params)

if args.train:
    model.train()
    pbar = tqdm(range(num_epochs), desc="Training")
    for epoch in pbar:
        epoch_losses = []
        for src_batch in dataloader:
            # adding model.train to fix the behavior of dropout and batchnorm layers since
            # they do behave differently depending on eval mode or train mode
            x = src_batch[:,0:-1].to(device)
            y = src_batch[:,1:].to(device)
            non_pad = (y != 0).float().sum().item()
            total = y.numel()
            # gradients accumulates by default. So each call to the backward method adds to the 
            # existing gradients stored in .grad tensors of model parameters. zero_grad clears
            # the old gradients. Its needed to prevent gradient accumulation across batches.
            # you typically zero the gradients once per training batch. You can also not zero
            # it and accumulate the gradients over multiple steps. For examples, your batch is
            # too big and cannot fit into the memory so you pick a smaller batch size and instead
            # update the gradients after num_steps

            optimizer.zero_grad()
            output, _ = model(x)
            loss = loss_fn(output.view(-1, vocab_size), y.reshape(-1))
            loss.backward()
            # Add after loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(loss.item()) # <<<< Note how it tries to get the loss value
            pbar.set_description(f"Loss: {loss.item():.4f}")        
            scheduler.step()
if args.train:
    # Save the checkpoint
    # Create a checkpoint dictionary
    checkpoint = {
        'epoch': 10,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': 0.5  # Example loss value
    }
    torch.save(checkpoint, 'model_checkpoint.pth')
    # To save only the model's state_dict:
    torch.save(model.state_dict(), 'model_state_dict.pth')

if args.eval:
    model.load_state_dict(torch.load('model_state_dict.pth', map_location=torch.device(device)))
    start_tokens = ['I speak from', 'Some parcels of']
    numerical_tokens = text_to_tokens(start_tokens, str_to_id)
    # generated_tokens = generate_text_batch(model, numerical_tokens)
    # generated_tokens = generate_text(model, numerical_tokens)
    generated_tokens = generate_text_beamsearch(model, numerical_tokens)
    generated_text = tokens_to_text(generated_tokens, id_to_str)
    print("generated_text = \n", generated_text)
