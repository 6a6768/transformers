import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len
        self.pad_token_id = tokenizer_tgt.token_to_id("[PAD]")
        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2 # -2 to not include EOS and SOS
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1 

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError(f"Sequence length too long: {src_text} or {tgt_text}")

        encoder_input_tokens = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token_id] * enc_num_padding_tokens, dtype=torch.int64),
            ]
        )
        # add SOS to the decoder input
        decoder_input_tokens = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token_id] * dec_num_padding_tokens, dtype=torch.int64),
            ]
        )

        # add EOS to the label (what we expect as output)
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token_id] * dec_num_padding_tokens, dtype=torch.int64),
            ]
        )

        assert encoder_input_tokens.size(0) == self.seq_len, "Encoder input size is not equal to sequence length"
        assert decoder_input_tokens.size(0) == self.seq_len, "Decoder input size is not equal to sequence length"
        assert label.size(0) == self.seq_len, "Label size is not equal to sequence length"

        return {
            "encoder_input": encoder_input_tokens, # (seq_len)
            "decoder_input": decoder_input_tokens, # (seq_len)
            "encoder_mask": (encoder_input_tokens != self.pad_token_id).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len) to not include padding tokens in the attention
            "decoder_mask": (decoder_input_tokens != self.pad_token_id).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input_tokens.size(0)), # (1, seq_len, seq_len) to not include padding tokens in the attention and the future tokens in the decoder
            "label": label, # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }

def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0 