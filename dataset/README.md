# Dataset

## 1. PretrainDataset

```python
sample['text'] = "Hello world"
max_length = 6
tokenizer.encode("Hello world") → [101, 7592, 2088, 102]
pad_token_id = 0
```

### Step 1 - After tokenization

```python
input_ids : [101, 7592, 2088, 102, 0, 0]
loss_mask = input_ids != pad_token_id: [1, 1, 1, 1, 0, 0]
```

### Step 2 - Build X and Y

```python
X = input_ids[:-1] # Drop the last token
Y = input_ids[1:]  # Drop the first token (next-token labels)

X = [101, 7592, 2088, 102, 0]
Y = [7592, 2088, 102,   0, 0]
```

### Step 3 - Adjust loss mask and convert to Tensor

```python
loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)
loss_mask = [1, 1, 1, 0, 0]
```

<br>

## 2. SFTDataset

### _generate_loss_mask

- Count loss only for tokens inside a BOS–EOS span
- Do NOT apply loss 
    - before BOS
    - on EOS
    - on padding
- Skip first token after BOS (special rule)

<br>

```python
<BOS> user question <EOS> <BOS> model answer <EOS>
```

##### Why skip first token after BOS?
Because for causal LM, the model should not predict the first token - it must be given.

<br>

```python
bos_id = [1]
eos_id = [2]
max_length = 20
input_ids = [5, 5, 1, 10, 11, 12, 2, 7, 1, 20, 21, 2, 0, 0]

Two BOS/EOS segments:
1, 10, 11, 12, 2
1, 20, 21, 2

Resulting loss mask:

Index:   0  1  2   3   4   5  6  7  8   9  10  11  12  13
Tokens: [5  5  1  10  11  12  2  7  1  20  21  2   0  0]
Mask:   [0  0  0  0    1   1  1  0  0   0   1  1   1  0]

BOS = [1]
EOS = [2]

input_ids =
[5, 5, 1, 10, 11, 12, 2, 7, 1, 20, 21, 2, 0, 0]
       |--------------|     |----------|
       BOS            EOS   BOS        EOS


index:       0 1 2 3 4 5 6 7 8  9  10 11 12 13
input_ids:  [5 5 1 10 11 12 2 7 1 20 21  2  0  0]
                 ^          ^
                 BOS        EOS

mask init:   [0 0 0 0 0 0 0 0 0 0  0  0  0  0]

# STEP 1: find BOS at index 2
content span = indices 3–6

skip index 3 !!!
apply mask to 4, 5, 6

mask now:    [0 0 0 0 1 1 1 0 0 0  0  0  0  0]

# STEP 2: find next BOS at index 8
content span = indices 9–11

skip index 9 !!!
apply mask to 10, 11

mask now:    [0 0 0 0 1 1 1 0 0 0 1 1 0 0]
```

<br>

## 3. DPODataset

### __getitem__

1. Convert "chosen" and "rejected" multi-turn dialog lists into strings using the chat template
2. Tokenize both
3. Generate loss masks that select only assistant tokens
4. Create X (input), Y (target), mask for both chosen and rejected sequences
5. Return everything for the DPO loss

```python
item = self.data[index]
chosen = item['chosen']
rejected = item['rejected']

# sample
{
  "chosen": [
    {"role": "user", "content": "How to cook rice?"},
    {"role": "assistant", "content": "Use a rice cooker..."}
  ],
  "rejected": [
    {"role": "user", "content": "How to cook rice?"},
    {"role": "assistant", "content": "Just google it."}
  ]
}

# Apply chat template (convert dialog → text)
chosen_prompt = self.tokenizer.apply_chat_template(
    chosen, tokenize=False, add_generation_prompt=False
)

rejected_prompt = self.tokenizer.apply_chat_template(
    rejected, tokenize=False, add_generation_prompt=False
)

# Tokenize chosen and rejected sequences
chosen_encoding = self.tokenizer(
    chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length'
)

rejected_encoding = self.tokenizer(
    rejected_prompt, truncation=True, max_length=self.max_length, padding='max_length'
)

# Extract input_ids
chosen_input_ids = chosen_encoding['input_ids']
rejected_input_ids = rejected_encoding['input_ids']

# Generate loss masks
chosen_loss_mask = self._generate_loss_mask(chosen_input_ids)
rejected_loss_mask = self._generate_loss_mask(rejected_input_ids)

# Build (X, Y, mask) for chosen sequences
x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)

# Build (X, Y, mask) for rejected sequences
x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

```

## 4. RLAIFDataset

- Prompt: all dialogue turns except last answer
- Answer: the final assistant response (training target)
