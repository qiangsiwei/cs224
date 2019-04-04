# -*- coding: utf-8 -*-

import nltk, torch, random, numpy as np
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
		self.logsm = nn.LogSigmoid()
		init = (2.0/(v_size+proj_dim))**0.5
		self.emb_u.weight.data.uniform_(0,0)
		self.emb_v.weight.data.uniform_(-init,init)
	def forward(self, c_words, t_words, n_words):
		c_embs = self.emb_v(c_words)
		t_embs = self.emb_u(t_words)
		n_embs = self.emb_u(n_words)
		loss = self.logsm(t_embs.bmm(c_embs.transpose(1,2)).squeeze(2))+\
			self.logsm(torch.sum(n_embs.bmm(c_embs.transpose(1,2)).squeeze(2),1).view(n_words.size(0),-1))
		return -torch.mean(loss)
	def prediction(self, inputs):
		return self.emb_v(inputs) 

class SkipGram(object):
	def __init__(self):
		self.w = 3
		self.z = 0.001
		self.n_neg = 10
		self.e_dim = 30
		self.b_size = 256
		self.lr = 0.01
	def train(self):
		def neg_sampling(t_words, k):
			neg_samples = []
			for i in xrange(len(t_words)):
				t_ind = t_words[i].data.tolist()[0]; nsample = []
				while len(nsample) < k:
					neg = random.choice(wd_tab)
					if w2i[neg] != t_ind: nsample.append(neg)
				neg_samples.append(Variable(LTensor([w2i[wd] if wd in w2i else w2i['<UNK>'] for wd in nsample])).view(1,-1))
			return torch.cat(neg_samples)
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
		x,y = zip(*[(win[self.w],win[i]) for win in wins for i in range(self.w*2+1) if i!=self.w and win[i]!='<DUMMY>'])
		x = [Variable(LTensor([w2i[wd] if wd in w2i else w2i['<UNK>']])).view(1,-1) for wd in x]
		y = [Variable(LTensor([w2i[wd] if wd in w2i else w2i['<UNK>']])).view(1,-1) for wd in y]
		model = Model(len(w2i),self.e_dim)
		if USE_CUDA: model = model.cuda()
		opt = optim.Adam(model.parameters(),lr=self.lr)
		wd_sum = sum([cnt for wd,cnt in wd_cnt.iteritems() if wd in vocab])
		wd_tab = [v for v in vocab for _ in xrange(int(((1.*wd_cnt[v]/wd_sum)**0.75)/self.z))]
		losses = []
		for epoch in xrange(100):
			for i in xrange(0,len(x),self.b_size):
				model.zero_grad()
				bx,by = x[i:i+self.b_size],y[i:i+self.b_size]
				negs = neg_sampling(by,self.n_neg)
				loss = model(torch.cat(bx),torch.cat(by),negs)
				loss.backward(); opt.step()
				losses.append(loss.data.tolist())
			print np.mean(losses); losses = []

if __name__ == '__main__':
	SkipGram().train()
