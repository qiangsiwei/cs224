# -*- coding: utf-8 -*-

import nltk, torch, numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from collections import Counter

USE_CUDA = torch.cuda.is_available()
if USE_CUDA: torch.cuda.set_device(0)
FTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
BTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor

class Model(nn.Module):
	def __init__(self, v_size, proj_dim):
		super(Model,self).__init__()
		self.emb_u = nn.Embedding(v_size,proj_dim)
		self.emb_v = nn.Embedding(v_size,proj_dim)
		self.bias_u = nn.Embedding(v_size,1)
		self.bias_v = nn.Embedding(v_size,1)
		init = (2.0/(v_size+proj_dim))**0.5
		self.emb_u.weight.data.uniform_(-init,init)
		self.emb_v.weight.data.uniform_(-init,init)
		self.bias_u.weight.data.uniform_(-init,init)
		self.bias_v.weight.data.uniform_(-init,init)
	def forward(self, c_words, t_words, c, w):
		c_embs = self.emb_v(c_words)
		t_embs = self.emb_u(t_words)
		c_bias = self.bias_v(c_words).squeeze(1)
		t_bias = self.bias_u(t_words).squeeze(1)
		loss = w*torch.pow(t_embs.bmm(c_embs.transpose(1,2)).squeeze(2)+c_bias+t_bias-c,2)
		return torch.sum(loss)
	def prediction(self, inputs):
		return self.emb_v(inputs)+self.emb_u(inputs)

class Glove(object):
	def __init__(self):
		self.w = 3
		self.e_dim = 30
		self.b_size = 256
		self.lr = 0.01
	def train(self):
		def weighting(c, c_max=100, alpha=.75):
		    return (c/c_max)**alpha if c<c_max else 1
		flatten = lambda a: [c for b in a for c in b]
		corpus = list(nltk.corpus.gutenberg.sents('melville-moby_dick.txt'))[:500]
		corpus = [[word.lower() for word in sent] for sent in corpus]
		wd_cnt = Counter(flatten(corpus))
		border = int(len(wd_cnt)*0.01)
		stopwords = wd_cnt.most_common()[:border]+wd_cnt.most_common()[-border:]
		vocab = list(set(flatten(corpus))-set(stopwords)); vocab.append('<UNK>')
		w2i = {w:i+1 for i,w in enumerate(vocab)}; w2i['<UNK>'] = 0
		i2w = {i:w for w,i in w2i.iteritems()}
		wins = flatten([list(nltk.ngrams(['<DUMMY>']*self.w+c+['<DUMMY>']*self.w,self.w*2+1)) for c in corpus])
		data = [(win[self.w],win[i]) for win in wins for i in range(self.w*2+1) if i!=self.w and win[i]!='<DUMMY>']
		bigram = Counter(data)
		X_ij = {}; weis = {}
		for (wi,wj),c in bigram.iteritems():
			X_ij[(wi,wj)] = X_ij[(wj,wi)] = bigram.get((wi,wj),0)+1
			weis[(wi,wj)] = weis[(wj,wi)] = weighting(bigram.get((wi,wj),0)+1)
		x = []; y = []; c = []; w = []
		for wi,wj in data:
			x.append(Variable(LTensor([w2i[wi] if wi in w2i else w2i['<UNK>']])).view(1,-1))
			y.append(Variable(LTensor([w2i[wj] if wj in w2i else w2i['<UNK>']])).view(1,-1))
			c.append(torch.log(Variable(FTensor([X_ij.get((wi,wj),1)]))).view(1,-1))
			w.append(Variable(FTensor([weis.get((wi,wj),0)])).view(1,-1))
		model = Model(len(w2i),self.e_dim)
		if USE_CUDA: model = model.cuda()
		opt = optim.Adam(model.parameters(),lr=self.lr)
		losses = []
		for epoch in xrange(100):
			for i in xrange(0,len(x),self.b_size):
				model.zero_grad()
				bx,by,bc,bw = x[i:i+self.b_size],y[i:i+self.b_size],c[i:i+self.b_size],w[i:i+self.b_size]
				loss = model(torch.cat(bx),torch.cat(by),torch.cat(bc),torch.cat(bw))
				loss.backward(); opt.step()
				losses.append(loss.data.tolist())
			print np.mean(losses); losses = []

if __name__ == '__main__':
	Glove().train()
