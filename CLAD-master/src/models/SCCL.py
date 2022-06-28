import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from transformers import BertPreTrainedModel
import os
import time
import numpy as np
from sklearn import cluster

import os
import pandas as pd
import torch.utils.data as util_data
from torch.utils.data import Dataset
import config


class VirtualAugSamples(Dataset):
    def __init__(self, train_x, train_y):
        assert len(train_x) == len(train_y)
        self.train_x = train_x
        self.train_y = train_y

    def __len__(self):
        return len(self.train_x)

    def __getitem__(self, idx):
        return {'text': self.train_x[idx], 'label': self.train_y[idx]}


def unshuffle_loader(text, label):
    train_text = text
    train_dataset = VirtualAugSamples(train_text, label)
    train_loader = util_data.DataLoader(
        train_dataset,
        batch_size=config.cluster_model_batch_size,
        shuffle=False,
        num_workers=1)
    return train_loader


def virtual_augmentation_loader(text, label):
    train_text = text
    import pdb
    train_dataset = VirtualAugSamples(train_text, label)
    train_loader = util_data.DataLoader(
        train_dataset,
        batch_size=config.cluster_model_batch_size,
        shuffle=True,
        num_workers=4)
    return train_loader


# from transformers import AutoModel, AutoTokenizer


class SCCLBert(nn.Module):
    def __init__(self, bert_model, tokenizer, cluster_centers=None, alpha=1.0):
        super(SCCLBert, self).__init__()

        self.tokenizer = tokenizer
        self.bert = bert_model
        self.emb_size = self.bert.config.hidden_size
        self.alpha = alpha

        # Instance-CL head
        self.contrast_head = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size), nn.ReLU(inplace=True),
            nn.Linear(self.emb_size, 128))

        # Clustering head
        initial_cluster_centers = torch.tensor(cluster_centers,
                                               dtype=torch.float,
                                               requires_grad=True)
        self.cluster_centers = Parameter(initial_cluster_centers)

    def forward(self, input_ids, attention_mask, task_type="virtual"):
        if task_type == "evaluate":
            return self.get_mean_embeddings(input_ids, attention_mask)

        elif task_type == "virtual":
            input_ids_1, input_ids_2 = torch.unbind(input_ids, dim=1)
            attention_mask_1, attention_mask_2 = torch.unbind(attention_mask,
                                                              dim=1)

            mean_output_1 = self.get_mean_embeddings(input_ids_1,
                                                     attention_mask_1)
            mean_output_2 = self.get_mean_embeddings(input_ids_2,
                                                     attention_mask_2)
            return mean_output_1, mean_output_2

    def get_mean_embeddings(self, input_ids, attention_mask):
        bert_output = self.bert.forward(input_ids=input_ids,
                                        attention_mask=attention_mask)
        attention_mask = attention_mask.unsqueeze(-1)
        mean_output = torch.sum(bert_output[0] * attention_mask,
                                dim=1) / torch.sum(attention_mask, dim=1)
        return mean_output

    def get_cluster_prob(self, embeddings):
        norm_squared = torch.sum(
            (embeddings.unsqueeze(1) - self.cluster_centers)**2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator**power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)

    def local_consistency(self, embd0, embd1, embd2, criterion):
        p0 = self.get_cluster_prob(embd0)
        p1 = self.get_cluster_prob(embd1)
        p2 = self.get_cluster_prob(embd2)

        lds1 = criterion(p1, p0)
        lds2 = criterion(p2, p0)
        return lds1 + lds2

    def contrast_logits(self, embd1, embd2=None):
        feat1 = F.normalize(self.contrast_head(embd1), dim=1)
        if embd2 != None:
            feat2 = F.normalize(self.contrast_head(embd2), dim=1)
            return feat1, feat2
        else:
            return


eps = 1e-8


class KLDiv(nn.Module):
    def forward(self, predict, target):
        assert predict.ndimension() == 2, 'Input dimension must be 2'
        target = target.detach()
        p1 = predict + eps
        t1 = target + eps
        logI = p1.log()
        logT = t1.log()
        TlogTdI = target * (logT - logI)
        kld = TlogTdI.sum(1)
        return kld


class KCL(nn.Module):
    def __init__(self):
        super(KCL, self).__init__()
        self.kld = KLDiv()

    def forward(self, prob1, prob2):
        kld = self.kld(prob1, prob2)
        return kld.mean()


def target_distribution(batch: torch.Tensor) -> torch.Tensor:
    weight = (batch**2) / (torch.sum(batch, 0) + 1e-9)
    return (weight.t() / torch.sum(weight, 1)).t()


class PairConLoss(nn.Module):
    def __init__(self, temperature=0.05):
        super(PairConLoss, self).__init__()
        self.temperature = temperature
        self.eps = 1e-08
        print(f"\n Initializing PairConLoss \n")

    def forward(self, features_1, features_2):
        device = features_1.device
        batch_size = features_1.shape[0]
        features = torch.cat([features_1, features_2], dim=0)
        mask = torch.eye(batch_size, dtype=torch.bool).to(device)
        mask = mask.repeat(2, 2)
        mask = ~mask

        pos = torch.exp(
            torch.sum(features_1 * features_2, dim=-1) / self.temperature)
        pos = torch.cat([pos, pos], dim=0)
        neg = torch.exp(
            torch.mm(features,
                     features.t().contiguous()) / self.temperature)
        neg = neg.masked_select(mask).view(2 * batch_size, -1)

        neg_mean = torch.mean(neg)
        pos_n = torch.mean(pos)
        Ng = neg.sum(dim=-1)

        loss_pos = (-torch.log(pos / (Ng + pos))).mean()

        return {
            "loss": loss_pos,
            "pos_mean": pos_n.detach().cpu().numpy(),
            "neg_mean": neg_mean.detach().cpu().numpy(),
            "pos": pos.detach().cpu().numpy(),
            "neg": neg.detach().cpu().numpy()
        }


class SCCLvTrainer(nn.Module):
    def __init__(self, model, tokenizer, optimizer, train_loader, eval_loader):
        super(SCCLvTrainer, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.eta = 1

        self.cluster_loss = nn.KLDivLoss(size_average=False)
        self.contrast_loss = PairConLoss(temperature=0.5)

        self.gstep = 0
        #print(f"*****Intialize SCCLv, temp:{self.args.temperature}, eta:{self.args.eta}\n")

    def get_batch_token(self, text):
        token_feat = self.tokenizer.batch_encode_plus(
            text,
            max_length=config.max_length,
            return_tensors='pt',
            padding='max_length',
            truncation=True)
        return token_feat

    def prepare_transformer_input(self, batch):
        if len(batch) == 4:
            text1, text2, text3 = batch['text'], batch[
                'augmentation_1'], batch['augmentation_2']
            feat1 = self.get_batch_token(text1)
            feat2 = self.get_batch_token(text2)
            feat3 = self.get_batch_token(text3)

            input_ids = torch.cat([
                feat1['input_ids'].unsqueeze(1),
                feat2['input_ids'].unsqueeze(1),
                feat3['input_ids'].unsqueeze(1)
            ],
                                  dim=1)
            attention_mask = torch.cat([
                feat1['attention_mask'].unsqueeze(1),
                feat2['attention_mask'].unsqueeze(1),
                feat3['attention_mask'].unsqueeze(1)
            ],
                                       dim=1)

        elif len(batch) == 2:
            text = batch['text']
            feat1 = self.get_batch_token(text)
            feat2 = self.get_batch_token(text)

            input_ids = torch.cat([
                feat1['input_ids'].unsqueeze(1),
                feat2['input_ids'].unsqueeze(1)
            ],
                                  dim=1)
            attention_mask = torch.cat([
                feat1['attention_mask'].unsqueeze(1),
                feat2['attention_mask'].unsqueeze(1)
            ],
                                       dim=1)

        return input_ids.cuda(), attention_mask.cuda()

    def train_step_virtual(self, input_ids, attention_mask):

        embd1, embd2 = self.model(input_ids,
                                  attention_mask,
                                  task_type="virtual")

        # Instance-CL loss

        feat1, feat2 = self.model.contrast_logits(embd1, embd2)
        losses = self.contrast_loss(feat1, feat2)
        loss = self.eta * losses["loss"]

        # Clustering loss
        #if self.args.objective == "SCCL":
        output = self.model.get_cluster_prob(embd1)
        target = target_distribution(output).detach()

        cluster_loss = self.cluster_loss(
            (output + 1e-08).log(), target) / output.shape[0]
        loss += 0.5 * cluster_loss
        losses["cluster_loss"] = cluster_loss.item()

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return losses

    def train_step_explicit(self, input_ids, attention_mask):

        embd1, embd2, embd3 = self.model(input_ids,
                                         attention_mask,
                                         task_type="explicit")

        # Instance-CL loss
        feat1, feat2 = self.model.contrast_logits(embd2, embd3)
        losses = self.contrast_loss(feat1, feat2)
        loss = self.eta * losses["loss"]

        # Clustering loss
        #if self.args.objective == "SCCL":
        output = self.model.get_cluster_prob(embd1)
        target = target_distribution(output).detach()

        cluster_loss = self.cluster_loss(
            (output + 1e-08).log(), target) / output.shape[0]
        loss += cluster_loss
        losses["cluster_loss"] = cluster_loss.item()

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return losses

    def train(self):
        #print('\n={}/{}=Iterations/Batches'.format(self.args.max_iter, len(self.train_loader)))

        self.model.train()
        for i in np.arange(config.iteration + 1):
            try:
                batch = next(train_loader_iter)
            except:
                train_loader_iter = iter(self.train_loader)
                batch = next(train_loader_iter)

            input_ids, attention_mask = self.prepare_transformer_input(batch)

            losses = self.train_step_virtual(input_ids, attention_mask)
            if i % 100 == 0:
                print(
                    f"iter: {i+1} losses: {losses['loss']}, {losses['cluster_loss']}"
                )

            if (i == config.iteration):
                all_prob = self.evaluate_embedding(i)
                self.model.train()

        return all_prob

    def evaluate_embedding(self, step):

        dataloader = self.eval_loader
        #for test, ood
        #print('---- {} evaluation batches ----'.format(len(dataloader)))

        self.model.eval()
        for i, batch in enumerate(dataloader):
            with torch.no_grad():
                text, label = batch['text'], batch['label']
                feat = self.get_batch_token(text)
                embeddings = self.model(feat['input_ids'].cuda(),
                                        feat['attention_mask'].cuda(),
                                        task_type="evaluate")

                model_prob = self.model.get_cluster_prob(embeddings)
                if i == 0:
                    all_labels = label
                    all_embeddings = embeddings.detach()
                    all_prob = model_prob
                else:
                    all_labels = torch.cat((all_labels, label), dim=0)
                    all_embeddings = torch.cat(
                        (all_embeddings, embeddings.detach()), dim=0)
                    all_prob = torch.cat((all_prob, model_prob), dim=0)

        # Initialize confusion matrices

        all_pred = all_prob.max(1)[1]

        return all_pred, self.model.bert


from transformers import AutoModel, AutoTokenizer, AutoConfig


def get_bert():

    # if args.use_pretrain == "SBERT":
    #     bert_model = get_sbert(args)
    #     tokenizer = bert_model[0].tokenizer
    #     model = bert_model[0].auto_model
    #     print("..... loading Sentence-BERT !!!")
    # else:
    LM_config = AutoConfig.from_pretrained(config.LM)
    model = AutoModel.from_pretrained(config.LM, config=LM_config)
    tokenizer = AutoTokenizer.from_pretrained(config.LM)
    print("..... loading plain BERT !!!")
    return model, tokenizer


def get_optimizer(model, lr_scale=100):

    optimizer = torch.optim.Adam(
        [{
            'params': model.bert.parameters()
        }, {
            'params': model.contrast_head.parameters(),
            'lr': config.cluster_model_train_lr * lr_scale
        }, {
            'params': model.cluster_centers,
            'lr': config.cluster_model_train_lr * lr_scale
        }],
        lr=config.cluster_model_train_lr)

    print(optimizer)
    return optimizer


def get_mean_embeddings(bert, input_ids, attention_mask):
    bert_output = bert.forward(input_ids=input_ids.cuda(),
                               attention_mask=attention_mask.cuda())
    attention_mask = attention_mask.unsqueeze(-1)
    mean_output = torch.sum(bert_output[0] * attention_mask.cuda(),
                            dim=1) / torch.sum(attention_mask.cuda(), dim=1)
    return mean_output


def get_batch_token(tokenizer, text, max_length):
    token_feat = tokenizer.batch_encode_plus(text,
                                             max_length=max_length,
                                             return_tensors='pt',
                                             padding='max_length',
                                             return_token_type_ids=False,
                                             truncation=True)
    return token_feat


from sklearn.cluster import KMeans


def get_kmeans_centers(bert, tokenizer, train_loader, num_classes, max_length):
    bert.to()
    for i, batch in enumerate(train_loader):

        text = batch['text']
        tokenized_features = get_batch_token(tokenizer, text, max_length)
        corpus_embeddings = get_mean_embeddings(bert, **tokenized_features)

        if i == 0:
            all_embeddings = corpus_embeddings.detach().cpu().numpy()
        else:
            all_embeddings = np.concatenate(
                (all_embeddings, corpus_embeddings.detach().cpu().numpy()),
                axis=0)

    clustering_model = KMeans(n_clusters=num_classes)
    clustering_model.fit(all_embeddings)
    return clustering_model.cluster_centers_


def run(train_text, label):

    # dataset loader
    train_loader = virtual_augmentation_loader(train_text,
                                               label.numpy().tolist())
    eval_loader = unshuffle_loader(train_text, label.numpy().tolist())
    # model
    torch.cuda.set_device(config.device)
    bert, tokenizer = get_bert()

    bert.cuda()
    cluster_centers = get_kmeans_centers(bert, tokenizer, train_loader,
                                         config.cluster_num, config.max_length)

    model = SCCLBert(bert,
                     tokenizer,
                     cluster_centers=cluster_centers,
                     alpha=1.0)
    model = model.cuda()

    # optimizer
    optimizer = get_optimizer(model)

    trainer = SCCLvTrainer(model, tokenizer, optimizer, train_loader,
                           eval_loader)
    all_prob = trainer.train()

    return all_prob


if __name__ == '__main__':
    run(train_text, label)
