import torch
import torch.nn.functional as F
from transformer import Cache
def prepare_shakspear(input_file='tinyshakespeare.txt', max_seq_len=30):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    str_to_id = {"<start>": 1, "<end>": 2, "<pad>": 0}
    id_to_str = {1: "<start>", 2: "<end>", 0: "<pad>"}
    id = 3
    for line in lines:
        for st in list(line):
            if st not in str_to_id:
                str_to_id[st] = id
                id_to_str[id] = st
                id += 1

    vocab_size = len(id_to_str)
    all_sentences = []
    for line in lines:
        chars = list(line.strip())
        if len(chars) > max_seq_len - 2:
            chars = chars[:max_seq_len - 2]
        if len(chars) > 0:
            tokens = [str_to_id["<start>"]] + [str_to_id[ch] for ch in chars] + [str_to_id["<end>"]]
            all_sentences.append(tokens)
    return all_sentences, vocab_size, str_to_id, id_to_str

def tokens_to_text(tokens_list, id_to_str):
    tokens_list = [[t for t in tokens if t not in [1, 2]] for tokens in tokens_list]  # Remove start, end tokens
    return [''.join([id_to_str[id] for id in tokens]) for tokens in tokens_list]

def text_to_tokens(texts, str_to_id):
    # include the start of the sequence as well
    return [[str_to_id["<start>"]]+[str_to_id[st] for st in text] for text in texts]

def top_k_sampling(logits, k, pad_token_id=None):
    if pad_token_id is not None:
        logits[pad_token_id] = float('-inf')
    if torch.all(logits == float('-inf')):  # <--- fallback
        return torch.tensor(pad_token_id)   # fallback to pad token
    values, indices = torch.topk(logits, k)
    # Apply softmax to get probabilities
    probs = F.softmax(values, dim=-1)
    # Sample from the distribution
    # binomial dist: P(x) = (n, x)p^x.q^(n-x)
    # multinomial dist: P(X1=x1,X2=x2,...)=(n!/(x1!x2!..)).p1^x1.p2^x2...
    next_token = indices[torch.multinomial(probs, num_samples=1)]
    return next_token
def clean_generated(seq, eos_token_id=2, pad_token_id=0):
    out = []
    for token in seq:
        if token == eos_token_id:
            break
        if token != pad_token_id:
            out.append(token)
    return out

def top_k_beams(candidates, beam_width):
    return sorted(candidates, key=lambda x: x[1], reverse=True)[0:beam_width] # <<<<< This sort command is pretty handy in python
    
def generate_text(
    model,
    sentences,
    max_length=30,
    temperature=1.0,
    top_k=10,
    eos_token_id=2,
    pad_token_id=0,
):
    model.eval()
    device = next(model.parameters()).device
    batch_tokens = []
    with torch.no_grad():
        for sentence in sentences:
            is_end_seq = False
            current_tokens = torch.tensor(sentence, dtype=torch.long).unsqueeze(dim=0).to(device)
            # TODO: get the number of layers prorammatically
            num_layers = len(model.decoders)
            cache = Cache.empty(num_layers=num_layers)
            while current_tokens.shape[1] < max_length and (not is_end_seq):
                output, cache = model(current_tokens[:,cache.length:], cache)
                logits = output[:, -1, :] / temperature
                logits[:, pad_token_id] = float('-inf')
                logits = logits[0] # since it's a single batch we take the index 0
                next_token = top_k_sampling(logits, top_k, pad_token_id=pad_token_id)
                is_end_seq = next_token.item() == eos_token_id
                current_tokens = torch.cat([current_tokens,next_token.unsqueeze(0)],dim=1)
            else:
                batch_tokens.append(current_tokens)
        
    generated_sequences = [clean_generated(seq.tolist()[0]) for seq in batch_tokens]

    return generated_sequences




"""
Input:
    - model: a function that scores the next tokens given a sequence
    - beam_width: the number of beams to keep at each step
    - max_length: maximum length of the output sequence
    - start_token: token that starts the sequence
    - end_token: token that ends the sequence

Initialize:
    - beams = [(start_token, 0)]   # Each beam is (sequence, score)

For step in 1 to max_length:
    - all_candidates = []

    For seq, score in beams:
        If seq ends with end_token:
            all_candidates.append((seq, score))
            continue

        next_tokens = model.predict_next(seq)  # returns list of (token, log_prob)

        For token, log_prob in next_tokens:
            new_seq = seq + [token]
            new_score = score + log_prob   # usually add log probs for total score
            all_candidates.append((new_seq, new_score))

    # Select top `beam_width` sequences by score
    beams = top_k(all_candidates, beam_width)

    If all beams end with end_token:
        Break

Return:
    - the sequence in beams with the highest score
"""
def generate_text_beamsearch(
    model,
    sentences,
    max_length=30,
    temperature=1.0,
    top_k=10,
    eos_token_id=2,
    pad_token_id=0,
    beam_width=5,
):
    model.eval()
    device = next(model.parameters()).device
    batch_tokens = []
    num_layers = len(model.decoders)
    with torch.no_grad():
        for sentence in sentences:
            cache = Cache.empty(num_layers=num_layers)
            beams = [(sentence, 0, cache)]
            for step in range(max_length):
                all_candidates = []
                for seq, score, cache in beams: # <<<<< Note that when you have beam search at each time step 
                    # you are doing beam_width number of inferences instead of a single one. So in pratice
                    # you end up doing beam_width times more inference per time step
                    if seq[-1] == eos_token_id:
                        all_candidates.append((seq, score, cache))
                        continue
                    current_tokens = torch.tensor(seq, dtype=torch.long).unsqueeze(dim=0).to(device)
                    output, new_cache = model(current_tokens[:, cache.length:], cache=cache)
                    logits = output[:, -1, :] / temperature # <<<<< Note that during inference you take the output for the last time step
                    logits[:, pad_token_id] = float('-inf')
                    log_probs = F.log_softmax(logits, dim=-1) # note this is not probability. For getting the probablity you should do torch.exp(log_probs)
                    candidates = [(seq+[tok], score+new_score, new_cache) for tok,new_score in enumerate(log_probs.squeeze().tolist())]
                    all_candidates = all_candidates + candidates

                # top k
                beams = top_k_beams(all_candidates, beam_width)
                
                # if all beams end with end of sentence token then terminate
                end_seq_counter = 0
                for seq, score, _ in beams:
                    if seq[-1] == eos_token_id:
                        end_seq_counter += 1
                if end_seq_counter == len(beams):
                    break
            # return the sequence with the highest score
            beams = top_k_beams(all_candidates, beam_width=1)
            batch_tokens.append(beams[0][0])
        
    generated_sequences = [clean_generated(seq) for seq in batch_tokens]

    return generated_sequences

    