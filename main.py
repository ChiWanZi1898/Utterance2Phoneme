from dataset import create_dataloader
from model import BiLSTM
from torch import nn
from torch.optim import Adam, lr_scheduler
import phoneme_list
import numpy as np
import editdistance
from ctcdecode import CTCBeamDecoder
from tqdm import tqdm
import torch
import os
from datetime import timedelta, datetime, tzinfo
import pytz
import pandas as pd

if __name__ == "__main__":
    batch_size = 64
    exp_id = "base_1_5_2"
    start_time = datetime.now().astimezone(pytz.timezone("America/Los_Angeles"))
    start_time_string = start_time.strftime("%Y-%m-%d-%H-%M-%S")
    checkpoint = "/home/ubuntu/Utterance2Phoneme/checkpoints/base_1_5_2_2020-03-31-17-02-51/013_9.161844"
    # checkpoint = ""
    print(f"Start Time: {start_time_string}")

    valid_dataloader = create_dataloader(
        "./data/wsj0_dev.npy", "./data/wsj0_dev_merged_labels.npy", batch_size=batch_size,
        shuffle=False)

    train_dataloader = create_dataloader(
        "./data/wsj0_train", "./data/wsj0_train_merged_labels.npy", batch_size=batch_size,
        shuffle=True)

    test_dataloader = create_dataloader(
        "./data/wsj0_test", None, batch_size=batch_size, test=True, shuffle=False
    )
    model = BiLSTM(40, 256, 47, 5, use_gpu=True)
    # model = Model(40, 47, 256)

    if checkpoint:
        model.load_state_dict(torch.load(checkpoint))

    model = model.cuda()
    ctc_loss = nn.CTCLoss()


    def criterion(out, label, data_len, label_len):
        loss = ctc_loss(out, label, data_len, label_len)
        reg_loss = 0
        for param in model.parameters():
            reg_loss += (param ** 2).sum()

        factor = 0.00001
        loss += factor * reg_loss
        return loss


    optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=5e-5)
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30, 40, 50, 60, 70, 80], gamma=0.5)

    decoder = CTCBeamDecoder(["$"] * 47, beam_width=10, num_processes=4, log_probs_input=True)


    def valid(e):
        distances = []
        pbar = tqdm(valid_dataloader)

        s1 = ""
        s2 = ""
        distance = 0

        for data, label, data_len, label_len in pbar:
            pbar.set_description("VALID")
            model.eval()

            out, out_lens = model(data, data_len)
            out_seq, _, _, out_lens = decoder.decode(out.transpose(1, 0), out_lens)

            for i in range(out_lens.size(0)):
                seq1 = out_seq[i, 0, :out_lens[i, 0]]

                s1 = ''.join(phoneme_list.PHONEME_MAP[i - 1] for i in seq1)
                seq2 = label[i, :label_len[i]]

                s2 = ''.join(phoneme_list.PHONEME_MAP[i - 1] for i in seq2)

                distance = editdistance.eval(s1, s2)
                distances.append(distance)

        mean_distances = np.mean(distances)
        print(f"Avg Distance: {mean_distances}")
        print(s1, len(s1))
        print(s2, len(s2))
        print(f"Distance: {distance}")

        dir = os.path.join("checkpoints", "_".join([exp_id, start_time_string]))
        if not os.path.exists(dir):
            os.makedirs(dir)
        path = os.path.join(dir, f"{e:03d}_{mean_distances:.06f}")
        torch.save(model.state_dict(), path)


    def test(e):
        results = []
        pbar = tqdm(test_dataloader)

        for data, data_len in pbar:
            pbar.set_description("TEST")
            model.eval()

            out, out_lens = model(data, data_len)
            out_seq, _, _, out_lens = decoder.decode(out.transpose(1, 0), out_lens)

            for i in range(out_lens.size(0)):
                seq1 = out_seq[i, 0, :out_lens[i, 0]]

                s1 = ''.join(phoneme_list.PHONEME_MAP[i - 1] for i in seq1)
                results.append(s1)

        dir = os.path.join("submissions", "_".join([exp_id, start_time_string]))
        if not os.path.exists(dir):
            os.makedirs(dir)
        path = os.path.join(dir, f"{e:03d}.csv")

        output = {
            'id': list(range(len(results))),
            'Predicted': results
        }

        pd.DataFrame(output, columns=['id', 'Predicted']).to_csv(path, index=False, header=True)


    def train(e):
        losses = []
        pbar = tqdm(train_dataloader)
        for data, label, data_len, label_len in pbar:
            pbar.set_description("TRAIN")
            model.train()

            optimizer.zero_grad()
            out, _ = model(data, data_len)
            loss = criterion(out, label, data_len, label_len)

            loss.backward()
            optimizer.step()
            losses.append(loss.cpu().detach().numpy())

        mean_loss = np.mean(losses)
        print(f"Avg Loss: {mean_loss}")


    test(0)
    valid(0)
    for e in range(1, 201):
        train(e)
        valid(e)
        test(e)
        # scheduler.step()
