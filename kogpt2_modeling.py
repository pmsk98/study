# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 14:59:50 2020

@author: user
"""


from kogpt2.pytorch_kogpt2 import get_pytorch_kogpt2_model
from gluonnlp.data import SentencepieceTokenizer
from kogpt2.utils import get_tokenizer
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch





  
PU='cpu' 


class Data_Set(Dataset):

  def __init__ (self, file_path,vocab,tokenizer):
    self.data = []
    self.vocab = vocab
    self.tokenizer = tokenizer

    f = open(file_path,'r',encoding='utf-8')

    file = f.read()
    file = file.split('\n')

    dataset = []
    now = ''

    for i, line in enumerate(file):
      if i % 30 == 0 and i != 0:
        dataset.append(now)
        now = ''

      now = now + '\n' + line

    for line in dataset:
      tokenized_line = tokenizer(line[:-1])

      indexing_word = [vocab[vocab.bos_token], ]+ vocab[tokenized_line] + [vocab[vocab.eos_token]]
      self.data.append(indexing_word)

    f.close()

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    return self.data[index]



model, vocab = get_pytorch_kogpt2_model()

model.to(torch.device(PU)) #모델 연산 유닛 설정
model.train() #모델 학습모드로 변경

save_path = 'C:/Users/user/KoGPT2/'

from transformers import GPT2Config, GPT2LMHeadModel

kogpt2_config = {
      "initializer_range": 0.02,
      "layer_norm_epsilon": 0.000001,
      "n_ctx": 1024,
      "n_embd": 768,
      "n_head": 12,
      "n_layer": 12,
      "n_positions": 1024,
      "vocab_size": 50000,
      "activation_function": "gelu"
}


torch.save(model.state_dict,'model_state_dict.pth') #모델의 가중치 값을 저장하는 코드입니다.
model.load_state_dict(torch.load(save_path+'KoGPT2_checkpoint.tar')) #모델의 가중치 값을 불러오는 코드입니다.

torch.save(model, PATH) #모델 전체를 저장하는 코드입니다.
model = torch.load(PATH) #모델 전체를 불러오는 코드입니다.




kogpt2model = GPT2LMHeadModel(config=GPT2Config.from_dict(kogpt2_config))



kogpt2model.train()

kogpt2model.to(torch.device(PU))

model = kogpt2model




file_path = 'C:/Users/user/KoGPT2/dataset.txt'
tokenizer = SentencepieceTokenizer(get_tokenizer(), num_best=0, alpha=0)

data = Data_Set(file_path, vocab, tokenizer)

dataset = DataLoader(data, batch_size=2, shuffle=True, pin_memory=True)

learning_rate = 0.00005
epochs = 1
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


for epoch in range(0, epochs+1):
  cnt = 0

  for data in dataset:
    optimizer.zero_grad()

    data = torch.stack(data)
    data = data.transpose(1,0)
    data = data.to(PU)

    output = model(data,labels=data)
    loss, logits = output[:2]
    loss.backward()
    optimizer.step()

    if cnt % 20 == 0:
      print("[+] epoch : {}, cnt : {}, loss : {} [+]".format(epoch, cnt+1, str(loss)[7:12]))


    cnt += 1
    
def dataset (file_path):
  data = []
  tokenizer = SentencepieceTokenizer(get_tokenizer())
  f = open(file_path,'r',encoding='utf-8')

  while True:
    file = f.readline()

    if not file:
      break
    line = tokenizer(file[:-1])
    indexing_word = [vocab[vocab.bos_token]]+ vocab[line] + [vocab[vocab.eos_token]]
    data.append(indexing_word)

  f.close()

  return data    

model, vocab = get_pytorch_kogpt2_model()



model.to(torch.device(PU)) #모델 연산 유닛 설정


model.eval()

del model

kogpt2_config = {
      "initializer_range": 0.02,
      "layer_norm_epsilon": 0.000001,
      "n_ctx": 1024,
      "n_embd": 768,
      "n_head": 12,
      "n_layer": 12,
      "n_positions": 1024,
      "vocab_size": 50000,
      "activation_function": "gelu"
}

kogpt2model = GPT2LMHeadModel(config=GPT2Config.from_dict(kogpt2_config))


kogpt2model.eval()

kogpt2model.to(torch.device(PU))

model = kogpt2model



Tokenizer = SentencepieceTokenizer(get_tokenizer(), num_best=0, alpha=0)

sentence = 'gs'
toked = Tokenizer(sentence)
temp = []
cnt = 0
while True:
  input_ids = torch.tensor([vocab[vocab.bos_token],] + vocab[toked]).unsqueeze(0)
  pred = model(input_ids)[0]

  gen = vocab.to_tokens(torch.argmax(pred, axis=-1).squeeze().tolist())
  print(gen)
  print(gen[-1])
  gen = gen[-1]
  cnt += 1

  if cnt == 50:
    break

  if '</s>' == gen:
    break
  sentence += gen.replace('▁', ' ')
  toked = Tokenizer(sentence)

print(sentence)