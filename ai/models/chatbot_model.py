import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# import gluonnlp as nlp
import numpy as np
from rapidfuzz import fuzz
import numpy as np
# from transformers import AdamW
from konlpy.tag import Komoran

max_len = 64
batch_size = 32
warmup_ratio = 0.1
num_epochs = 20
max_grad_norm = 1
log_interval = 200
learning_rate = 5e-05

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len, pad, pair):
            self.sentences = [([i[sent_idx]]) for i in dataset]
            self.labels = [np.int32(i[label_idx]) for i in dataset]
    # print(self.sentences)
    # print(self.labels)

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))
  
def chatbot(sentence: str):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
    tok = 1
    data = [sentence, 1.0, 1.0, 0.0]
    dataset_another = [data]
    # another_test = Dataset()
    test_dataloader = DataLoader(dataset_another, batch_size=batch_size, num_workers = 5)

    for i in range(len(data)):
        token_ids = data[1]
        token_ids = token_ids
        segment_ids=data[2]
        segment_ids = segment_ids # Move segment_ids to device
        # valid_length = valid_length
        label=data[3]
        label = label

    # token_ids = token_ids.long()
    # segment_ids = segment_ids.long() # Move segment_ids to device
    # valid_length = valid_length
    # label = label.long()

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

    category = {'슈링크플레이션': 0 , '스킴플레이션': 1, '원재료유사품': 2, '즐겨찾기': 3, '2가지상품비교': 4}

    out = np.asarray([0,1,2,3,4])
    test_eval = []
    for i in out:
      logits = i
      test_eval.append(list(category.keys())[(np.argmax(logits))])

    idxi = -2
    max_func = 0.0
    func_list = ['스킴플레이션', '슈링크플레이션', '원재료가 비슷한 상품 찾기', '즐겨찾기', '두 가지 상품 비교']
    for i in range(len(func_list)):
        prob = fuzz.ratio(func_list[i], sentence)
        if np.float16(prob) > np.float16(max_func):
            max_func = prob
            idxi = i

    product=''
    komoran = Komoran()
    morphs = komoran.morphs(sentence)
    # for m in morphs:
    if ('와' in morphs):
        bef = morphs.index('와')
        product = morphs[bef-1]

    elif ('과' in morphs):
        bef = morphs.index('과')
        product = morphs[bef-1]

    result = dict()
    result['product'] = product
    result['function'] = int(idxi + 1)

    return result