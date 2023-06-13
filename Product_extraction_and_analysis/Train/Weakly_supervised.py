import pandas as pd
import numpy as np
import pickle
import json
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
import random
import argparse
import logging
from collections import OrderedDict
from ast import literal_eval
from konlpy.tag import Komoran, Okt
from tqdm import tqdm, trange

from sklearn.feature_extraction.text import CountVectorizer
from scipy.special import softmax
from transformers import AutoTokenizer, BertModel, BertConfig
from torch.utils.data import DataLoader, Dataset

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from transformers.data.metrics import acc_and_f1
from transformers import (
    get_cosine_schedule_with_warmup
)
from torch import nn
import torch


# Setup logging
logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

with open("komoran_seed.pkl", "rb") as f:
  komoran_seed = pickle.load(f)

with open("okt_seed.pkl", "rb") as f:
  okt_seed = pickle.load(f)

def get_seed_words(token_seed):
  aspects = token_seed.keys()
  seed_words = []
  for aspect in aspects:
    temp = token_seed[aspect]
    seed_words += list(temp.keys())
  seed_words = list(set(seed_words))
  return seed_words

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

class Teacher:
  def __init__(self, token_seed, tokenizer_name, aspects):
    self.token_name = "{}_token".format(tokenizer_name)
    self.token_seed = token_seed
    self.seed_words = get_seed_words(token_seed)
    self.vectorizer = CountVectorizer(decode_error="ignore")
    self.vectorizer.fit([" ".join(self.seed_words)])
    self.aspects = aspects
    self.bag_of_words = None
    self.weight_mat = None

  def pred(self, data):
    # If a token column is a string, not an array change it to array
    train_col =  data[self.token_name]
    if isinstance(train_col.iloc[train_col.first_valid_index()], str):
      data[self.token_name] = data[self.token_name].apply(lambda x: literal_eval(x))
    self.bag_of_words = self.vectorizer.transform(data[self.token_name].apply(lambda x : " ".join(x))).toarray()
    seed_idx_map = self.vectorizer.vocabulary_
    aspect_idx_map = {}
    bag_of_words_pred = np.empty((data.shape[0],1))
    if self.weight_mat is None:
      self.weight_mat = np.ones((data.shape[0], len(self.aspects)+1))
    for i, aspect in enumerate(self.aspects):
      aspect_seed = self.token_seed[aspect].keys()
      aspect_idx_map[aspect] = [seed_idx_map[seed] for seed in aspect_seed]
      aspect_seed_weight = self.weight_mat[aspect_idx_map[aspect], i].reshape(1,-1)
      aspect_seed_occur = (self.bag_of_words[:, aspect_idx_map[aspect]] * aspect_seed_weight).sum(axis=1)
      bag_of_words_pred = np.concatenate([bag_of_words_pred, aspect_seed_occur.reshape(-1,1)], axis=1)
    bag_of_words_pred = bag_of_words_pred[:, 1:]
    general_asp_score = (bag_of_words_pred.sum(axis=1) == 0).astype("int64")
    bag_of_words_pred = np.concatenate([bag_of_words_pred, general_asp_score.reshape(-1,1)], axis=1)
    bag_of_words_pred = softmax(bag_of_words_pred, axis=1)

    return bag_of_words_pred
  def update(self, student_pred):
    '''
    weight z_j^k (j-th seed word predictive power of k-th aspect)
    z_j has (1,k) dimension -> z has (N, k) dimension where N is the number of seed words
    '''
    occur_mat = (self.bag_of_words > 0).astype("int64")
    temp_weight = np.matmul(occur_mat.T, student_pred) + np.finfo(float).eps
    self.weight_mat = temp_weight/(temp_weight.sum(axis=1).reshape(-1,1))

class CustomDataset(Dataset):
    def __init__(self, tokens, labels):
        self.tokens = tokens
        self.labels = labels

    def __getitem__(self, index):
        input_ids = self.tokens["input_ids"][index]
        token_type_ids = self.tokens["token_type_ids"][index]
        attention_mask = self.tokens["attention_mask"][index]
        label = self.labels[index]

        return input_ids, token_type_ids, attention_mask, label, index
    
    def __len__(self):
        return len(self.labels)

class BertClassifier(nn.Module):
  def __init__(self, model_name, n_labels, dropout=0.0):
    super(BertClassifier, self).__init__()
    config = BertConfig.from_pretrained(model_name)
    self.bert = BertModel.from_pretrained(model_name)
    self.dropout = nn.Dropout(dropout)
    self.linear = nn.Linear(config.hidden_size, n_labels)

  def forward(self, input_id, mask, token_ids):
    _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask, token_type_ids=token_ids, return_dict=False)
    dropout_output = self.dropout(pooled_output)
    output = self.linear(dropout_output)
    return output

'''
Arguments
model_name seed, tokenizer_name, aspects, seed_words, lr, weight_decay,batch_size, bootstrap_epoch, scheduler_gamma, epochs, 
'''
class TrainArgs():
  def __init__(self, args, aspects):
    self.model_name = args.model_name
    self.seed = args.seed
    self.tokenizer_name = args.tokenizer_name
    self.lr = args.lr
    self.weight_decay = args.weight_decay
    self.dropout = args.dropout
    self.batch_size = args.batch_size
    self.bootstrap_epoch = args.bootstrap_epoch
    self.warmup_ratio = args.warmup_ratio
    self.adam_epsilon = args.adam_epsilon
    self.epochs = args.epochs #default 25
    self.model_path = args.model_path
    self.aspects = aspects
    
def concat_col_aspects(df):
  return df["aspect1"] + " " +  df["aspect2"] + " " + df["aspect3"]

def prepare_evaluation_data(args, gpu, n_gpus, val_data, test_data, batch_size, num_workers, pin_memory, tokenizer):
  val_token = tokenizer(val_data["review_text"].values.tolist(), 
                          padding=True, truncation=True, return_tensors="pt")
  test_token = tokenizer(test_data["review_text"].values.tolist(), 
                            padding=True, truncation=True, return_tensors="pt")
  val_data[["aspect1", "aspect2", "aspect3"]] = val_data[["aspect1", "aspect2", "aspect3"]].fillna("")
  test_data[["aspect1", "aspect2", "aspect3"]] = test_data[["aspect1", "aspect2", "aspect3"]].fillna("")
  label_vectorizer = CountVectorizer()
  eval_aspect = args.aspects + ['none']
  label_vectorizer.fit(eval_aspect)
  val_label_df = val_data.apply(lambda x: concat_col_aspects(x), axis=1)
  test_label_df = test_data.apply(lambda x: concat_col_aspects(x), axis=1)
  val_label = label_vectorizer.transform(val_label_df).toarray()
  test_label = label_vectorizer.transform(test_label_df).toarray()

  val_dataset = CustomDataset(val_token, val_label)
  val_sampler = DistributedSampler(val_dataset, num_replicas=n_gpus, rank=gpu, shuffle=False, drop_last=False)
  val_loader = DataLoader(val_dataset, batch_size=batch_size,num_workers=num_workers, 
                          pin_memory=pin_memory, sampler=val_sampler)
  test_dataset = CustomDataset(test_token, test_label)
  test_sampler = DistributedSampler(test_dataset, num_replicas=n_gpus, rank=gpu, shuffle=False, drop_last=False)
  test_loader = DataLoader(test_dataset, batch_size=batch_size,num_workers=num_workers, 
                          pin_memory=pin_memory, sampler=test_sampler)
  return val_loader, test_loader



def evaluate(gpu, args, model, val_loader, test_loader, n_gpus):
  logger.info("***** Running evaluation *****")

  criterion = nn.CrossEntropyLoss()
  val_iterator = tqdm(val_loader,desc="Validation Iteration", mininterval=10, ncols=100)
  test_iterator = tqdm(test_loader,desc="Test Iteration", mininterval=10, ncols=100)
  val_loss = 0
  test_loss = 0
  with torch.no_grad():
    for i, batch in enumerate(val_iterator):
      model.eval()
      batch = tuple(t.to(gpu) for t in batch)
      inputs = {"input_ids": batch[0], "attention_mask": batch[1],"token_type_ids": batch[2]}
      label = batch[3]
      outputs = model(inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids'])
      loss = criterion(outputs, label.float())
      print(gpu, loss, n_gpus)
      dist.all_reduce(loss, op=dist.ReduceOp.SUM)
      loss = loss / n_gpus
      val_loss += loss.item()
    val_loss = val_loss/len(val_loader)
  with torch.no_grad():
    for i, batch in enumerate(test_iterator):
      model.eval()
      batch = tuple(t.to(gpu) for t in batch)
      inputs = {"input_ids": batch[0], "attention_mask": batch[1],"token_type_ids": batch[2]}
      label = batch[3]
      outputs = model(inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids'])
      loss = criterion(outputs, label.float())
      dist.all_reduce(loss, op=dist.ReduceOp.SUM) 
      loss = loss / n_gpus
      test_loss += loss.item()
    test_loss = test_loss/len(test_loader)
  return val_loss, test_loss

def train_bert(gpu, n_gpus, args, train_data, val_data, test_data, seed_words_list):
  # gpu device
  print("Use GPU: {} for training".format(gpu))
  dist.init_process_group(backend='nccl', 
                    init_method='tcp://127.0.0.1:2345',
                    world_size=n_gpus, 
                    rank=gpu)
  PATH = args.model_path
  student_aspects = args.aspects + ["none"]
  # multi gpu configures -> to have same overall batch size regardless of gpu numbers
  batch_size = int(args.batch_size / n_gpus)
  pin_memory = False
  num_workers = 0
  # sets seeds for numpy, torch and python.random.
  seed_everything(args.seed)
  # Initialize teacher classifier
  picked_seed_word =  seed_words_list[args.tokenizer_name]
  teacher_classifier = Teacher(picked_seed_word, args.tokenizer_name, args.aspects)

  # Get pretrained tokenzier and model(for student classification)
  tokenizer = AutoTokenizer.from_pretrained(args.model_name)
  
  student_classifier = BertClassifier(args.model_name, n_labels = len(student_aspects), dropout=args.dropout).to(gpu)
  logger.info("***** Tokenizing training data *****")
  train_token = tokenizer(train_data["review_text"].values.tolist(), 
                            padding=True, truncation=True, return_tensors="pt")

  # Evaluation data
  val_loader, test_loader = prepare_evaluation_data(args, gpu, n_gpus, val_data, test_data, batch_size, num_workers, pin_memory, tokenizer)
  # Initialize optimizer, scheduler
  total_training_steps = (len(train_data)//args.batch_size + 1) * args.epochs
  warmup_steps = total_training_steps * args.warmup_ratio

  no_decay = ["bias", "LayerNorm.weight"]
  optimizer_grouped_parameters = [
      {"params": [p for n, p in student_classifier.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": args.weight_decay,
        },
      {"params": [p for n, p in student_classifier.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0
        },
  ]

  optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
  scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_training_steps)
  # loss function : Cross-entropy
  criterion = nn.CrossEntropyLoss()
  # Load saved training model
  start_epoch = 0
  best_val = np.inf
  if os.path.exists(os.path.join(PATH, "best_valid_model.pt")):
    print("Load saved model")
    # Load Teacher model
    with open(os.path.join(PATH,"teacher_best_valid_model.pkl"), "rb") as f:
      teacher_classifier = pickle.load(f)
    # Load Student model
    check_point = torch.load(os.path.join(PATH, "best_valid_model.pt"))
    state_dict = check_point["model_state_dict"]
    model_dict = OrderedDict()
    pattern = re.compile('module.')
    for k,v in state_dict.items():
        if re.search("module", k):
            model_dict[re.sub(pattern, '', k)] = v
        else:
            model_dict = state_dict
    student_classifier.load_state_dict(model_dict)
    optimizer.load_state_dict(check_point["optimizer_state_dict"])
    scheduler.load_state_dict(check_point["scheduler_state_dict"])
    start_epoch = check_point["epoch"] + 1
    best_val = check_point["loss"]
  
  # Multi gpu model to sychonizer gradients
  student_classifier = DDP(student_classifier, device_ids=[gpu], output_device=gpu)
  # Progress bar for epoch progress
  train_iterator = tqdm(range(start_epoch, args.epochs), desc="Epoch",mininterval=10, ncols=100)

  # Variables for total logging
  train_loss_list = []
  val_loss_list = []
  test_loss_list = []
  n_batch_step_list = []
  n_batch_step = 0
 
  logger.info("***** Running training *****")
  print("Start Training at Epoch : {}".format(start_epoch))
  bootstrap_epoch = 0
  print("number of epochs : ", args.epochs)
  for epoch, _ in enumerate(train_iterator):
    # Set sample every epoch
    # Variables for logging every epoch
    print("Current bootstrap epoch {}, Argument's bootstrap epoch {}".format(bootstrap_epoch, args.bootstrap_epoch))
    train_loss = 0
    teacher_pred = teacher_classifier.pred(train_data)
    train_dataset = CustomDataset(train_token, teacher_pred)
    train_sampler = DistributedSampler(train_dataset, num_replicas=n_gpus, rank=gpu, shuffle=True, drop_last=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              num_workers=num_workers, pin_memory=pin_memory, sampler=train_sampler)
    epoch_iterator = tqdm(train_loader,desc=f"GPU:{gpu} Iteration", mininterval=10, ncols=100)
    # Check number of train data for each gpu
    if epoch == 0 :
      logger.info(f"Number of traing data GPU {gpu} : {len(train_loader)*batch_size}")
    train_iterator.set_description(f"GPU: {gpu}, train_epoch: {epoch} train_loss: {train_loss:.4f}")
    train_loader.sampler.set_epoch(epoch)
    for i,batch in enumerate(epoch_iterator):
      student_classifier.train()
      batch = tuple(t.to(gpu) for t in batch)
      inputs = {"input_ids": batch[0], "attention_mask": batch[1],"token_type_ids": batch[2]}
      label = batch[3]
      outputs = student_classifier(inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids'])
      loss = criterion(outputs, label)
      loss.backward()
      optimizer.step()
      scheduler.step()
      student_classifier.zero_grad()
      train_loss += loss.item()
      n_batch_step += 1
      train_loss_list.append(loss.item())
      n_batch_step_list.append(n_batch_step)
    train_loss /= len(train_loader)
    print("GPU : {} / Epoch : {} / Train loss : {:.4f}".format(gpu, epoch, train_loss))
  # Save student predictions to update teacher classifier
    student_train_pred = torch.zeros((len(train_dataset),len(student_aspects))).to(gpu)
    pred_sampler = DistributedSampler(train_dataset, shuffle=False)
    pred_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, sampler=pred_sampler)
    pred_iterator = tqdm(pred_loader,desc="Iteration", mininterval=10, ncols=100)
    if bootstrap_epoch == args.bootstrap_epoch:
      print("Preparing Student's prediction to update Teacher")
      with torch.no_grad():
        for i,batch in enumerate(pred_iterator):
          student_classifier.eval()
          batch = tuple(t.to(gpu) for t in batch)
          inputs = {"input_ids": batch[0], "attention_mask": batch[1],"token_type_ids": batch[2]}
          idx = batch[4]
          outputs = student_classifier(inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids'])
          student_train_pred[idx] = outputs
      dist.all_reduce(student_train_pred, op=dist.ReduceOp.SUM)
      student_train_pred = student_train_pred.detach().cpu().numpy()
      student_train_pred = softmax(student_train_pred, axis=1)
      teacher_classifier.update(student_train_pred)
      print(f"GPU:{gpu}, teacher weight matrix sum : {teacher_classifier.weight_mat.sum()}")
      bootstrap_epoch = 0
    else:
      bootstrap_epoch += 1
    # Evalution

    val_loss, test_loss = evaluate(gpu, args, student_classifier, val_loader, test_loader, n_gpus)
    val_loss_list.append(val_loss)
    test_loss_list.append(test_loss)
    print("Epoch : {} / Val loss : {:.4f}".format(epoch, val_loss))
    print("Epoch : {} / Test loss : {:.4f}".format(epoch, test_loss))
    # save logs
    log_data = {"train_loss_log": train_loss_list, "val_loss_log": val_loss_list, 
                "test_loss_log": test_loss_list, "n_batch_step_log":n_batch_step_list }
    if gpu == 0:
      try:
        with open(os.path.join(PATH,'logs.pkl'), 'wb') as f:
          pickle.dump(log_data, f)
      except: 
        print("Error occured while saving logs")
      # save model if it show best validation performance
      if best_val > val_loss:
        best_val = val_loss
        try:
          torch.save({
                'epoch': epoch,
                'model_state_dict': student_classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict' : scheduler.state_dict(),
                'loss': val_loss,
                }, os.path.join(PATH, "best_valid_model.pt"))
          with open(os.path.join(PATH,'teacher_best_valid_model.pkl'), 'wb') as f:
            pickle.dump(teacher_classifier, f)
          print("Model saved")
        except:
          print("Error occured while saving models")
          best_model_configs = {'best_epoch' : epoch, 'best_model_state' : student_classifier.state_dict(),
                        'best_optimizer_state_dict' : optimizer.state_dict(), 'best_loss': val_loss,
                        'teach_model' : teacher_classifier}

def main():
    # Pass Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--tokenizer_name", type=str)
    parser.add_argument("--model_name", type=str, default="klue/bert-base")
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--warmup_ratio", type=float, default=0.01)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--bootstrap_epoch", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=8)

    args = parser.parse_args()

    # Prepare data
    logger.info("***** Preparing data *****")
    develop = pd.read_csv("develop.csv", encoding='utf-8-sig')
    test = pd.read_csv("test.csv", encoding='utf-8-sig')
    seed_words = {'okt':okt_seed, 'komoran':komoran_seed}
    review_df = pd.read_csv("train_review.csv", encoding='utf-8-sig')
    develop_test_index = np.concatenate([develop["idx"].values, test["idx"].values])
    train = review_df.drop(develop_test_index, axis=0)
    train = train.dropna(axis=0).reset_index(drop=True)
    develop["okt_token"] = develop["okt_token"].apply(lambda x: literal_eval(x))
    aspects = list(develop["aspect1"].unique())
    aspects.remove("none")
    # Train model
    targs = TrainArgs(args, aspects)
    logger.info("***** Arguments Prepared *****")
    print(f"cuda: {torch.cuda.is_available()}")
    n_gpus = torch.cuda.device_count()
    print("Total number of gpus :", n_gpus)
    print("Number of training data :", len(train))
    if not os.path.exists(args.model_path):
      os.mkdir(args.model_path)
    mp.spawn(train_bert, nprocs=n_gpus, args=(n_gpus, targs, train, develop, test, seed_words))
if __name__ == "__main__":
    main()