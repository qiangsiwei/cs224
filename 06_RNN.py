# -*- coding: utf-8 -*-

import torch, numpy as np
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
	def __init__(self, v_size, e_dim, h_dim, n_layers=1, dropout=0.5):
		super(Model,self).__init__()
		self.n_layers = n_layers
		self.h_dim = h_dim
		self.embed = nn.Embedding(v_size,e_dim)
		self.rnn = nn.LSTM(e_dim,h_dim,n_layers,batch_first=True)
		self.linear = nn.Linear(h_dim,v_size)
		self.dropout = nn.Dropout(dropout)
		self.embed.weight = nn.init.xavier_uniform_(self.embed.weight)
		self.linear.weight = nn.init.xavier_uniform_(self.linear.weight)
		self.linear.bias.data.fill_(0)
	def init_hidden(self, b_size):
		h = Variable(torch.zeros(self.n_layers,b_size,self.h_dim))
		c = Variable(torch.zeros(self.n_layers,b_size,self.h_dim))
		return (h.cuda(),c.cuda()) if USE_CUDA else (h,c)
	def detach_hidden(self, hiddens):
		return tuple([h.detach() for h in hiddens])
	def forward(self, inputs, hidden, is_training=False): 
		embeds = self.embed(inputs)
		if is_training: embeds = self.dropout(embeds)
		o,h = self.rnn(embeds,hidden)
		return self.linear(o.contiguous().view(o.size(0)*o.size(1),-1)),h

class LM(object):
	def __init__(self):
		self.lr = 0.01
		self.b_size = 16
		self.seq_len = 20
	def train(self):
		def proc(fn, w2i=None):
			flatten = lambda a:[c for b in a for c in b]
			corpus = open(fn).read().strip().decode('utf-8').split(u'\n')
			corpus = flatten([ln.strip().split()+['</s>'] for ln in corpus])
			if w2i is None: w2i = {w:i+1 for i,w in enumerate(set(corpus))}; w2i['<unk>'] = 0
			return LTensor([w2i[w] if w in w2i else w2i['<unk>'] for w in corpus]),w2i
		def batch(data):
			data = data.narrow(0,0,data.size(0)//self.b_size*self.b_size)
			data = data.view(self.b_size,-1).contiguous()
			if USE_CUDA: data = data.cuda()
			return data
		data_tr,w2i = proc('data/06/ptb.train.txt')
		# data_dv,_ = proc('data/06/ptb.valid.txt',w2i)
		# data_te,_ = proc('data/06/ptb.test.txt',w2i)
		data_tr,data_dv,data_te = batch(data_tr),batch(data_dv),batch(data_te)
		i2w = {i:w for w,i in w2i.iteritems()}
		model = Model(len(w2i)+1,128,256,1,0.5) 
		if USE_CUDA: model = model.cuda()
		loss_func = nn.CrossEntropyLoss()
		opt = optim.Adam(model.parameters(),lr=self.lr)
		for epoch in xrange(30):
			losses = []; h = model.init_hidden(self.b_size)
			for i in xrange(0,data_tr.size(1)-self.seq_len,self.seq_len):
				h = model.detach_hidden(h)
				x = Variable(data_tr[:,i:i+self.seq_len])
				y = Variable(data_tr[:,(i+1):(i+1)+self.seq_len].contiguous())
				model.zero_grad()
				p,h = model(x,h,is_training=True)
				loss = loss_func(p,y.view(-1))
				loss.backward(); opt.step()
				losses.append(loss.data.tolist())
			print np.exp(np.mean(losses)); losses = []

if __name__ == '__main__':
	LM().train()
