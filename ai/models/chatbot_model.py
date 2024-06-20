# def chatbot(sentence: str):

#     result = {
#         "product": "오뚜기 컵누들",
#         "function": 1
#     }

#     return result
# BERT 모델에 넣을 dataset
from KoBERT.kobert import get_tokenizer
from KoBERT.kobert import get_pytorch_kobert_model
# basic dependencies
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# import gluonnlp as nlp
import numpy as np
from tqdm import tqdm

# data processing dependencies
import pandas as pd

# transformers
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

import gluonnlp as nlp
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
bertmodel, vocab = get_pytorch_kobert_model()
# tokenizer = get_tokenizer()
# tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower = False)

max_len = 64
batch_size = 32
warmup_ratio = 0.1
num_epochs = 20
max_grad_norm = 1
log_interval = 200
learning_rate = 5e-05

class BERTDataset(Dataset):
  def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len, pad, pair):
    transform = nlp.data.BERTSentenceTransform(bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)
    self.sentences = [transform([i[sent_idx]]) for i in dataset]
    self.labels = [np.int32(i[label_idx]) for i in dataset]
    # print(self.sentences)
    # print(self.labels)

  def __getitem__(self, i):
    return (self.sentences[i] + (self.labels[i], ))

  def __len__(self):
    return (len(self.labels))

# # 데이터 토큰화
tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower = False)

# data_train = BERTDataset(dataset_train, 0, 1, tok , max_len, True, False)
# data_test = BERTDataset(dataset_test, 0, 1, tok, max_len, True, False)

# train_dataloader = torch.utils.data.DataLoader(data_train, batch_size = batch_size, num_workers = 5)
# test_dataloader = torch.utils.data.DataLoader(data_test, batch_size = batch_size, num_workers = 5)

# define classifier
# num_classes : 분류하고자 하는 class 수
class BERTClassifier(nn.Module):
  def __init__(self, bert, hidden_size = 768, num_classes = 5, dr_rate=None, params=None):
    super(BERTClassifier, self).__init__()
    self.bert = bert
    self.dr_rate = dr_rate

    self.classifier = nn.Linear(hidden_size, num_classes)
    if dr_rate:
      self.dropout = nn.Dropout(p=dr_rate)

  def gen_attention_mask(self, token_ids, valid_length):
    attention_mask = torch.zeros_like(token_ids)
    for i, v in enumerate(valid_length):
      attention_mask[i][:v] = 1
    return attention_mask.float()

  def forward(self, token_ids, valid_length, segment_ids):
    attention_mask = self.gen_attention_mask(token_ids, valid_length)

    _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
    if self.dr_rate:
      out = self.dropout(pooler)
    return self.classifier(out)
  
model = BERTClassifier(bertmodel, dr_rate = 0.5).to(device)
model.load_state_dict(torch.load('../weights/chat_model.pt'))

category = {'슈링크플레이션': 0 , '스킴플레이션': 1, '원재료유사품': 2, '즐겨찾기': 3, '2가지상품비교': 4}

def test(predict_sentence):
  data = [predict_sentence, '0']
  dataset_another = [data]
  another_test = BERTDataset(dataset_another, 0, 1, tok, max_len, True, False)
  test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=batch_size, num_workers = 5)

  model.eval()
  for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
    token_ids = token_ids.long().to(device)
    segment_ids = segment_ids.long().to(device) # Move segment_ids to device
    valid_length = valid_length
    label = label.long().to(device)
    # token_ids = token_ids.long()
    # segment_ids = segment_ids.long() # Move segment_ids to device
    # valid_length = valid_length
    # label = label.long()

    out = model(token_ids, valid_length, segment_ids)
    test_eval = []
    for i in out:
      logits = i
      logits = logits.detach().cpu().numpy()
      test_eval.append(list(category.keys())[(np.argmax(logits))])

    result = dict()
    result['function'] = test_eval[0]
    print('Inference 결과:', test_eval[0])
    return(test_eval[0])

if __name__ == '__main__':
  test('이거 두 개를 비교해줘.')