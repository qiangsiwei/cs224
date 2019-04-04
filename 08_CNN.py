# -*- coding: utf-8 -*-

import re, torch, random, gensim, numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

USE_CUDA = torch.cuda.is_available()
if USE_CUDA: torch.cuda.set_device(0)
FTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
BTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor

class Model(nn.Module):
	def __init__(self, v_size, e_dim, o_size, ker_dim=100, ker_sizes=(3,4,5), dropout=0.5):
		super(Model,self).__init__()
		self.embed = nn.Embedding(v_size,e_dim)
		self.convs = nn.ModuleList([nn.Conv2d(1,ker_dim,(K,e_dim)) for K in ker_sizes])
		self.dropout = nn.Dropout(dropout)
		self.fc = nn.Linear(len(ker_sizes)*ker_dim,o_size)
		init = (2.0/e_dim)**0.5
		self.embed.weight.data.uniform_(-init,init)
	def init_weights(self, pretrained, is_static=False):
		self.embed.weight = nn.Parameter(torch.from_numpy(pretrained).float())
		if is_static: self.embed.weight.requires_grad = False
	def forward(self, inputs, is_training=False):
		inputs = self.embed(inputs).unsqueeze(1)
		inputs = [F.relu(conv(inputs)).squeeze(3) for conv in self.convs]
		inputs = [F.max_pool1d(inp,inp.size(2)).squeeze(2) for inp in inputs]
		concated = torch.cat(inputs,1)
		if is_training: concated = self.dropout(concated)
		return F.log_softmax(self.fc(concated),1)

class CNN(object):
	def __init__(self):
		self.b_size = 50
		self.lr = 0.001
	def train(self):
		data = open('data/08/train_5500.label.txt').read().strip().decode('utf-8').split(u'\n')
		X,Y = zip(*[ln.split(':',1)[::-1] for ln in data]); X = list(X)
		for i,x in enumerate(X): X[i] = re.sub(r'\d','#',x).split()
		w2i = {w:i+2 for i,w in enumerate(set([w for x in X for w in x]))}
		w2i['<pad>'] = 0; w2i['<unk>'] = 1
		t2i = {t:i for i,t in enumerate(set(Y))}
		i2w,i2t = {i:w for w,i in w2i.items()},{i:t for t,i in t2i.items()}
		X_tr,Y_tr = [],[]
		for x,y in zip(X,Y):
			X_tr.append(Variable(LTensor([w2i[w] if w in w2i else w2i['<unk>'] for w in x])).view(1,-1))
			Y_tr.append(Variable(LTensor([t2i[y]])).view(1,-1))
		data_tr = zip(X_tr,Y_tr); random.shuffle(data_tr)
		data_tr,data_te = data_tr[:int(len(data_tr)*.9)],data_tr[int(len(data_tr)*.9):]
		# wv = gensim.models.KeyedVectors.load_word2vec_format('data/08/GoogleNews-vectors-negative300.bin',binary=True)
		# pretrained = np.vstack([wv[w] if w in wv else np.random.randn(300) for w in w2i])
		model = Model(len(w2i),300,len(t2i),100,(3,4,5))
		# model.init_weights(pretrained)
		if USE_CUDA: model = model.cuda()
		loss_func = nn.CrossEntropyLoss()
		opt = optim.Adam(model.parameters(),lr=self.lr)
		# training
		for epoch in xrange(5):
			random.shuffle(data_tr); losses = []
			for i in xrange(0,len(data_tr),self.b_size):
				x,y = zip(*data_tr[i:i+self.b_size])
				x_max = max([s.size(1) for s in x])
				x = torch.cat([torch.cat([s,Variable(LTensor([w2i['<pad>']]*(x_max-s.size(1)))).view(1,-1)],1)\
					if s.size(1)<x_max else s for s in x])
				y = torch.cat(y).view(-1)
				model.zero_grad()
				preds = model(x,is_training=True)
				loss = loss_func(preds,y)
				loss.backward(); opt.step()
				losses.append(loss.data.tolist())
			print np.mean(losses); losses = []
		# testing
		accu = 0
		for x,y in data_te:
			p = model(x).max(1)[1].data.tolist()[0]
			y = y.data.tolist()[0][0]
			if p == y: accu += 1
		print 100.*accu/len(data_te)

if __name__ == '__main__':
	CNN().train()
