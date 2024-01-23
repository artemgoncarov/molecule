import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 512
max_pred_len = 200


class Vocabulary:
    def __init__(self, freq_threshold=2, reverse=False):
        self.itos = {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "<unk>"}
        self.stoi = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        self.freq_threshold = freq_threshold
        self.reverse = reverse
        self.tokenizer = self._tokenizer

    def __len__(self):
        return len(self.itos)

    def _tokenizer(self, text):
        return (char for char in text)

    def tokenize(self, text):
        if self.reverse:
            return [token for token in self.tokenizer(text)][::-1]
        else:
            return [token for token in self.tokenizer(text)]

    def build_vocabulary(self, sentence_list):
        """Basically builds a frequency map for all possible characters."""
        frequencies = {}
        idx = len(self.itos)

        for sentence in sentence_list:
            # Preprocess the InChI.
            for char in self.tokenize(sentence):
                if char in frequencies:
                    frequencies[char] += 1
                else:
                    frequencies[char] = 1

                if frequencies[char] == self.freq_threshold:
                    self.stoi[char] = idx
                    self.itos[idx] = char
                    idx += 1

    def numericalize(self, text):
        """Convert characters to numbers."""
        tokenized_text = self.tokenize(text)

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<unk>"]
            for token in tokenized_text
        ]


class BMSDataset(Dataset):
    def __init__(self, df, vocab, photo_dir):
        super().__init__()
        self.df = df
        self.vocab = vocab
        self.photo_dir = photo_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Read the image
        img_id = self.df.iloc[idx]['id']
        label = self.df.iloc[idx]['smiles']
        label_len = len(label) + 2  # (2 for <sos> and <eos>)
        img_path = os.path.join(self.photo_dir, f'{img_id}.png')

        img = self._load_from_file(img_path)

        # Convert label to numbers
        label = self._get_numericalized(label, self.vocab)
        return img, torch.tensor(label), torch.tensor(label_len)

    def _get_numericalized(self, sentence, vocab):
        """Numericalize given text using prebuilt vocab."""
        numericalized = [vocab.stoi["<sos>"]]
        numericalized.extend(vocab.numericalize(sentence))
        numericalized.append(vocab.stoi["<eos>"])
        return numericalized

    def _load_from_file(self, img_path):
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        image /= 255.0  # Normalize
        return image


class BMSDatasetTest(Dataset):
    def __init__(self, df, photo_dir):
        super().__init__()
        self.df = df
        self.photo_dir = photo_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Read the image
        img_id = self.df.iloc[idx]['id']
        img_path = os.path.join(self.photo_dir, f'{img_id}.png')
        img = self._load_from_file(img_path)
        return torch.tensor(img).permute(2, 0, 1)

    def _load_from_file(self, img_path):
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        image /= 255.0  # Normalize
        return image



def save_checkpoint(epoch, encoder, decoder, encoder_optimizer, decoder_optimizer):
    """
    Saves model checkpoint.
    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param encoder: encoder model
    :param decoder: decoder model
    :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
    :param decoder_optimizer: optimizer to update decoder's weights
    :param bleu4: validation BLEU-4 score for this epoch
    :param is_best: is this checkpoint the best so far?
    """
    state = {'epoch': epoch,
             'encoder': encoder,
             'decoder': decoder,
             'encoder_optimizer': encoder_optimizer,
             'decoder_optimizer': decoder_optimizer}
    filename = f'checkpoint_{epoch}.pth.tar'
    torch.save(state, filename)


def inference(encoder, decoder, imgs, vocab):
    imgs = imgs.to(DEVICE)
    batch_size = len(imgs)

    encoder_out = encoder(imgs)  # (1, enc_image_size, enc_image_size, encoder_dim)
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)

    encoder_out = encoder_out.view(batch_size, -1, encoder_dim)

    # Start decoding
    h, c = decoder.init_hidden_state(encoder_out)
    start = torch.full((batch_size, 1), vocab.stoi['<sos>']).to(DEVICE)
    pred = torch.zeros((batch_size, max_pred_len), dtype=torch.long).to(DEVICE)
    pred[:, 0] = start.squeeze()

    idx = 1

    while True:
        embeddings = decoder.embedding(start).squeeze(1)

        awe, _ = decoder.attention(encoder_out, h)

        gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
        awe = gate * awe

        h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))
        scores = decoder.fc(h)  # (s, vocab_size)
        scores = F.log_softmax(scores, dim=1)

        start = scores.argmax(1).reshape(-1, 1).to(DEVICE)

        pred[:, idx] = start.squeeze(1)

        if idx >= max_pred_len - 1:
            break

        idx += 1

    return pred


def batch_stringify(batch, vocab):
    preds = []
    for item in batch:
        pred = np.vectorize(vocab.itos.get)(item)
        # Truncate everything after <eos>
        try:
            pred = pred[1:np.nonzero(pred == '<eos>')[0][0]]
        except IndexError:
            pred = pred[1:]
            pass

        preds.append("".join(pred))
    return preds
