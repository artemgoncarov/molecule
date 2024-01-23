import argparse
import pandas as pd
import torch

from torch.utils.data import DataLoader

from utils import Vocabulary, BMSDatasetTest, batch_stringify, inference

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', default='../data/test', type=str, help='img dir')
    parser.add_argument('--indices_path', default='../data/indices.csv', type=str, help='indices path')
    parser.add_argument('--output_path', default='../predictions.csv', type=str, help='output path')
    args = parser.parse_args()

    checkpoint = torch.load('checkpoint_12.pth.tar', map_location=device)
    encoder = checkpoint['encoder']
    decoder = checkpoint['decoder']

    sample_sub_df = pd.read_csv('train.csv')
    indices_df = pd.read_csv(args.indices_path)

    freq_threshold = 2
    vocab = Vocabulary(freq_threshold=freq_threshold, reverse=False)
    vocab.build_vocabulary(sample_sub_df['smiles'].to_list())

    test_dataset = BMSDatasetTest(indices_df, args.img_dir)
    test_batch_size = 8
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        pin_memory=True,
        num_workers=4,
        shuffle=False,
    )

    preds = []
    for imgs in test_dataloader:

        preds.extend(batch_stringify(inference(encoder, decoder, imgs, vocab).cpu().detach().numpy(), vocab))

    df = pd.DataFrame.from_dict({"id": indices_df.id, "smiles": preds})
    df.to_csv(args.output_path)


if __name__ == "__main__":
    main()
