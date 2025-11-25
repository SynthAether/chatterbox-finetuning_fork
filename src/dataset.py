import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from src.utils import setup_logger


logger = setup_logger(__name__)



class ChatterboxDataset(Dataset):
    
    def __init__(self, config):
        self.cfg = config
        self.data = pd.read_csv(config.csv_path, sep="|", header=None, quoting=3)
        self.sot_token = config.start_text_token 
        self.eot_token = config.stop_text_token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            row = self.data.iloc[idx]
            filename = str(row[0])
            if not filename.endswith(".wav"): filename += ".wav"
            
            pt_path = os.path.join(self.cfg.preprocessed_dir, filename.replace(".wav", ".pt"))
            
            if not os.path.exists(pt_path):
                return None
            
            data = torch.load(pt_path)
            
            
            text_tokens = data["text_tokens"]
            if text_tokens.size(0) > self.cfg.max_text_len - 2:
                text_tokens = text_tokens[:self.cfg.max_text_len - 2]
                
            sot = torch.tensor([self.sot_token], dtype=torch.long)
            eot = torch.tensor([self.eot_token], dtype=torch.long)
            text_tokens = torch.cat([sot, text_tokens, eot])

            # 2. Speech Tokens
            speech_tokens = data["speech_tokens"]
            if speech_tokens.size(0) > self.cfg.max_speech_len:
                speech_tokens = speech_tokens[:self.cfg.max_speech_len]

            return {
                "text_tokens": text_tokens,
                "speech_tokens": speech_tokens,
                "speaker_emb": data["speaker_emb"],
                "prompt_tokens": data["prompt_tokens"]
            }

        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            return None


def data_collator(batch):

    batch = [item for item in batch if item is not None]
    if not batch: 
        return {}

    # Padding
    text_tokens = pad_sequence([x["text_tokens"] for x in batch], batch_first=True, padding_value=0)
    speech_tokens = pad_sequence([x["speech_tokens"] for x in batch], batch_first=True, padding_value=0)
    prompt_tokens = pad_sequence([x["prompt_tokens"] for x in batch], batch_first=True, padding_value=0)

    speaker_embs = torch.stack([x["speaker_emb"] for x in batch])

    # Lengths (Required for masking)
    text_lens = torch.tensor([len(x["text_tokens"]) for x in batch], dtype=torch.long)
    speech_lens = torch.tensor([len(x["speech_tokens"]) for x in batch], dtype=torch.long)


    return {
        "text_tokens": text_tokens,
        "text_token_lens": text_lens,
        "speech_tokens": speech_tokens,
        "speech_token_lens": speech_lens,
        "speaker_emb": speaker_embs,
        "prompt_tokens": prompt_tokens
    }