import pandas as pd
import numpy as np
from tqdm import tqdm
import timm
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch.nn.utils.rnn import pad_sequence

from sklearn.model_selection import train_test_split
from Levenshtein import distance as levenshtein_distance
from model import (
    Encoder,
    DecoderWithAttention,
    DEVICE,
    AverageMeter,
    accuracy,
    clip_gradient,
    train_epoch,
    validate_epoch,
)
from utils import Vocabulary, BMSDataset, save_checkpoint, inference, batch_stringify

emb_dim = 512  # dimension of word embeddings
attention_dim = 512  # dimension of attention linear layers
decoder_dim = 512  # dimension of decoder RNN
dropout = 0.4
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

start_epoch = 0
epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
epochs = 3  # number of epochs to train for (if early stopping is not triggered)
workers = 4  # for data-loading; right now, only 1 works with h5py
encoder_lr = 1e-3  # learning rate for encoder if fine-tuning
decoder_lr = 4e-3  # learning rate for decoder
fine_tune_encoder = False  # fine-tune encoder?

label_df = pd.read_csv(
    "/Users/artemgoncarov/Documents/Projects/source/train_data/train.csv"
).iloc[:, 1:]
img_dir = "/Users/artemgoncarov/Documents/Projects/source/train_data/train"

label_df = label_df.iloc[:10000]
train_df, test_df = train_test_split(label_df, test_size=0.1, shuffle=False)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

freq_threshold = 2
vocab = Vocabulary(freq_threshold=freq_threshold, reverse=False)
vocab.build_vocabulary(train_df["smiles"].to_list())


def bms_collate(batch):
    imgs, labels, label_lens = [], [], []

    for data_point in batch:
        imgs.append(torch.from_numpy(data_point[0]).permute(2, 0, 1))
        labels.append(data_point[1])
        label_lens.append(data_point[2])

    labels = pad_sequence(labels, batch_first=True, padding_value=vocab.stoi["<pad>"])

    return torch.stack(imgs), labels, torch.stack(label_lens).reshape(-1, 1)


train_dataset = BMSDataset(train_df, vocab, img_dir)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    pin_memory=True,
    num_workers=workers,
    shuffle=True,
    collate_fn=bms_collate,
)

val_dataset = BMSDataset(test_df, vocab, img_dir)
val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    pin_memory=True,
    num_workers=workers,
    shuffle=False,
    collate_fn=bms_collate,
)

decoder = DecoderWithAttention(
    attention_dim=attention_dim,
    embed_dim=emb_dim,
    decoder_dim=decoder_dim,
    vocab_size=len(vocab),
    dropout=dropout,
)
decoder_optimizer = torch.optim.Adam(
    params=filter(lambda p: p.requires_grad, decoder.parameters()), lr=decoder_lr
)
encoder = Encoder(timm.create_model("resnet101", pretrained=True))
encoder.fine_tune(fine_tune_encoder)
encoder_optimizer = (
    torch.optim.Adam(
        params=filter(lambda p: p.requires_grad, encoder.parameters()), lr=encoder_lr
    )
    if fine_tune_encoder
    else None
)

decoder = decoder.to(DEVICE)
encoder = encoder.to(DEVICE)

criterion = nn.CrossEntropyLoss().to(DEVICE)

best_score = 0
for epoch in range(start_epoch, epochs):
    # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
    if epochs_since_improvement == 20:
        break

    # One epoch's training
    train_loss, train_top5acc, batch_time = train_epoch(
        train_loader=train_loader,
        encoder=encoder,
        decoder=decoder,
        criterion=criterion,
        encoder_optimizer=encoder_optimizer,
        decoder_optimizer=decoder_optimizer,
        epoch=epoch,
    )

    val_loss, val_top5acc, _ = validate_epoch(val_loader, encoder, decoder, criterion)

    if best_score < val_top5acc.avg:
        best_score = val_top5acc.avg
        print(f"Saving checkpoint. Best score: {best_score:.4f}")
        save_checkpoint(
            epoch + 1, encoder, decoder, encoder_optimizer, decoder_optimizer
        )

    print(f"Epoch: {epoch + 1:02} | Time: {batch_time.avg} sec")
    print(f"\t    Train Loss: {train_loss.avg:.4f} | Val. Loss: {val_loss.avg:.4f}")
    print(
        f"\t    Top5 Acc.: {train_top5acc.avg:.3f} | Val. Top5 Acc.: {val_top5acc.avg:.3f} \n"
    )

preds, gts = [], []
for imgs, caps, capslen in tqdm(val_loader):
    preds.extend(
        batch_stringify(
            inference(encoder, decoder, imgs, vocab).cpu().detach().numpy(), vocab
        )
    )
    gts.extend(batch_stringify(caps, vocab))

print(
    f"Levenshtein distance: {np.mean(np.vectorize(levenshtein_distance)(preds, gts))}"
)
