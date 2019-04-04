# -*- coding: utf-8 -*-

import nltk, torch, random, numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from operator import itemgetter as oi

USE_CUDA = torch.cuda.is_available()
if USE_CUDA: torch.cuda.set_device(0)
FTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
BTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor

class Model(nn.Module): 
	def __init__(self, w, v_size, e_dim, h_dim, o_size):
		super(Model,self).__init__()
		self.embed = nn.Embedding(v_size,e_dim)
		self.h_layer1 = nn.Linear(e_dim*(w*2+1),h_dim)
		self.h_layer2 = nn.Linear(h_dim,h_dim)
		self.o_layer = nn.Linear(h_dim,o_size)
		self.relu = nn.ReLU()
		self.softmax = nn.LogSoftmax(dim=1)
		self.dropout = nn.Dropout(0.3)
	def forward(self, inputs, is_training=False): 
		embeds = self.embed(inputs)
		concated = embeds.view(-1,embeds.size(1)*embeds.size(2))
		h0 = self.relu(self.h_layer1(concated))
		if is_training: h0 = self.dropout(h0)
		h1 = self.relu(self.h_layer2(h0))
		if is_training: h1 = self.dropout(h1)
		return self.softmax(self.o_layer(h1))

class NER(object):
	def __init__(self):
		self.w = 2
		self.b_size = 128
		self.lr = 0.001
	def train(self):
		def x2i(l, start=0):
			s = set([i for x in l for i in x])
			return {w:i+start for i,w in enumerate(s)}
		corpus = nltk.corpus.conll2002.iob_sents()
		data = [(map(oi(0),x),map(oi(2),x)) for x in corpus]; ss,ts = zip(*data)
		w2i,t2i = x2i(ss,2),x2i(ts); w2i.update({'<UNK>':0,'<DUMMY>':1})
		i2w,i2t = {i:w for w,i in w2i.items()},{i:t for t,i in t2i.items()}
		wins = [(win,sample[1][i]) for sample in data for i,win in \
			enumerate(nltk.ngrams(['<DUMMY>']*self.w+list(sample[0])+['<DUMMY>']*self.w,self.w*2+1))]
		random.shuffle(wins)
		data_tr,data_te = wins[:int(len(wins)*.9)],wins[int(len(wins)*.9):]
		model = Model(self.w,len(w2i),50,300,len(t2i))
		if USE_CUDA: model = model.cuda()
		loss_func = nn.CrossEntropyLoss()
		opt = optim.Adam(model.parameters(),lr=self.lr)
		# training
		for epoch in xrange(3):
			losses = []
			for i in xrange(0,len(data_tr),self.b_size):
				model.zero_grad()
				bx,by = zip(*data_tr[i:i+self.b_size])
				bx = torch.cat([Variable(LTensor([w2i[w] if w in w2i else w2i['<UNK>'] for w in x])).view(1,-1) for x in bx])
				by = torch.cat([Variable(LTensor([t2i[y]])) for y in by])
				preds = model(bx,is_training=True)
				loss = loss_func(preds,by)
				loss.backward(); opt.step()
				losses.append(loss.data.tolist())
			print np.mean(losses); losses = []
		# testing
		accu = 0
		for x,y in data_te:
			x = Variable(LTensor([w2i[w] if w in w2i else w2i['<UNK>'] for w in x])).view(1,-1) 
			p = i2t[model(x).max(1)[1].data.tolist()[0]]
			if p == y: accu += 1
		print 100.*accu/len(data_te)
		
if __name__ == '__main__':
	NER().train()
