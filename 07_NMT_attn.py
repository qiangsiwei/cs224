# -*- coding: utf-8 -*-

import re, torch, random, unicodedata, numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import *
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

USE_CUDA = torch.cuda.is_available()
if USE_CUDA: torch.cuda.set_device(0)
FTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
BTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor

class Enc(nn.Module):
	def __init__(self, v_size, e_dim, h_dim, n_layer=1, bidirec=True):
		super(Enc, self).__init__()
		self.h_dim = h_dim
		self.n_layer = n_layer
		self.embed = nn.Embedding(v_size,e_dim)
		self.n_direc = 2 if bidirec else 1
		self.gru = nn.GRU(e_dim,h_dim,n_layer,batch_first=True,bidirectional=bidirec)
	def init_hidden(self, inputs):
		hidden = Variable(torch.zeros(self.n_layer*self.n_direc,inputs.size(0),self.h_dim))
		return hidden.cuda() if USE_CUDA else hidden
	def init_weight(self):
		self.embed.weight = nn.init.xavier_uniform_(self.embed.weight)
		self.gru.weight_hh_l0 = nn.init.xavier_uniform_(self.gru.weight_hh_l0)
		self.gru.weight_ih_l0 = nn.init.xavier_uniform_(self.gru.weight_ih_l0)
	def forward(self, inputs, input_lens):
		embed = self.embed(inputs)
		hidden = self.init_hidden(inputs)
		packed = pack_padded_sequence(embed,input_lens,batch_first=True)
		outputs,hidden = self.gru(packed,hidden)
		outputs,_ = pad_packed_sequence(outputs,batch_first=True)
		if self.n_layer > 1: hidden = hidden[-2:] if self.n_direc==2 else hidden[-1]
		return outputs, torch.cat([h for h in hidden],1).unsqueeze(1)

class Dec(nn.Module):
	def __init__(self, v_size, e_dim, h_dim, n_layer=1, dropout=0.1):
		super(Dec,self).__init__()
		self.h_dim = h_dim
		self.n_layer = n_layer
		self.embed = nn.Embedding(v_size,e_dim)
		self.dropout = nn.Dropout(dropout)
		self.gru = nn.GRU(e_dim+h_dim,h_dim,n_layer,batch_first=True)
		self.linear = nn.Linear(h_dim*2,v_size)
		self.attn = nn.Linear(self.h_dim,self.h_dim)
	def init_hidden(self, inputs):
		hidden = Variable(torch.zeros(self.n_layer,inputs.size(0),self.h_dim))
		return hidden.cuda() if USE_CUDA else hidden
	def init_weight(self):
		self.embed.weight = nn.init.xavier_uniform_(self.embed.weight)
		self.gru.weight_hh_l0 = nn.init.xavier_uniform_(self.gru.weight_hh_l0)
		self.gru.weight_ih_l0 = nn.init.xavier_uniform_(self.gru.weight_ih_l0)
		self.linear.weight = nn.init.xavier_uniform_(self.linear.weight)
		self.attn.weight = nn.init.xavier_uniform_(self.attn.weight)
	def attention(self, hidden, enc_outs, enc_masks):
		b_size = enc_outs.size(0)
		maxlen = enc_outs.size(1)
		hidden = hidden[0].unsqueeze(2)
		energies = self.attn(enc_outs.contiguous().view(b_size,maxlen,-1))
		attns = energies.bmm(hidden).squeeze(2)
		alpha = F.softmax(attns,1).unsqueeze(1)
		return alpha.bmm(enc_outs), alpha
	def forward(self, inputs, context, max_len, enc_outs, enc_masks=None, is_training=False):
		embed = self.embed(inputs)
		hidden = self.init_hidden(inputs)
		if is_training: embed = self.dropout(embed)
		decodes = []
		for i in xrange(max_len):
			_,hidden = self.gru(torch.cat((embed,context),2),hidden)
			concated = torch.cat((hidden,context.transpose(0,1)),2)
			softmax = F.log_softmax(self.linear(concated.squeeze(0)),1)
			decodes.append(softmax)
			embed = self.embed(softmax.max(1)[1]).unsqueeze(1)
			if is_training: embed = self.dropout(embed)
			context,alpha = self.attention(hidden,enc_outs,enc_masks)
		return torch.cat(decodes,1).view(inputs.size(0)*max_len,-1)
	def decode(self, context, enc_outs, t2i):
		start = Variable(LTensor([[t2i['<s>']]*1])).transpose(0,1)
		embed = self.embed(start)
		hidden = self.init_hidden(start)
		decodes = []; attns = []; decoded = embed
		while decoded.data.tolist()[0] != t2i['</s>']:
			_,hidden = self.gru(torch.cat((embed,context),2),hidden)
			concated = torch.cat((hidden,context.transpose(0,1)),2)
			softmax = F.log_softmax(self.linear(concated.squeeze(0)),1)
			decodes.append(softmax)
			decoded = softmax.max(1)[1]
			embed = self.embed(decoded).unsqueeze(1)
			context,alpha = self.attention(hidden,enc_outs,None)
			attns.append(alpha.squeeze(1))
		return torch.cat(decodes).max(1)[1], torch.cat(attns)

class NMT(object):
	def __init__(self):
		self.min = 3
		self.max = 25
		self.b_size = 64
		self.lr = 0.001
	def train(self):
		def norm_str(string):
			def to_ascii(string):
				return ''.join(c for c in unicodedata.normalize('NFD',string)\
					if unicodedata.category(c)!='Mn')
			return re.sub(r'\s+',' ',re.sub(r'[^a-zA-Z,.!?]+',' ',\
				re.sub(r'([,.!?])',r' \1 ',to_ascii(string.lower().strip())))).strip()
		corpus = open('data/07/eng-fra.txt').read().strip().decode('utf-8').split('\n')[:10000]
		data = []
		for line in corpus:
			s1,s2 = line.split('\t')
			if not s1.strip() or not s2.strip(): continue
			s1,s2 = norm_str(s1).split(),norm_str(s2).split()
			if self.min<=len(s1)<=self.max and self.min<=len(s2)<=self.max: data.append((s1,s2))
		s2i = {w:i+4 for i,w in enumerate(set([w for s in zip(*data)[0] for w in s]))}
		s2i.update({'<PAD>':0,'<UNK>':1,'<s>':2,'</s>':3})
		i2s = {i:w for w,i in s2i.iteritems()}
		t2i = {w:i+4 for i,w in enumerate(set([w for s in zip(*data)[1] for w in s]))}
		t2i.update({'<PAD>':0,'<UNK>':1,'<s>':2,'</s>':3})
		i2t = {i:w for w,i in t2i.iteritems()}
		data_tr = [(\
			Variable(LTensor([s2i[w] if w in s2i else s2i['<UNK>'] for w in s1+['</s>']])).view(1,-1),\
			Variable(LTensor([t2i[w] if w in t2i else t2i['<UNK>'] for w in s2+['</s>']])).view(1,-1))\
				for s1,s2 in data]
		def pad(batch):
			x,y = zip(*sorted(batch,key=lambda b:b[0].size(1),reverse=True))
			x_max,y_max = max([s.size(1) for s in x]),max([s.size(1) for s in y])
			bx,by = [],[]
			for i in xrange(len(batch)):
				bx.append(torch.cat([x[i],Variable(LTensor([s2i['<PAD>']]*(x_max-x[i].size(1)))).view(1,-1)],1) if x[i].size(1)<x_max else x[i])
				by.append(torch.cat([y[i],Variable(LTensor([t2i['<PAD>']]*(y_max-y[i].size(1)))).view(1,-1)],1) if y[i].size(1)<y_max else y[i])
			bx,by = torch.cat(bx),torch.cat(by)
			xl = [list(map(lambda s:s==0,s.data)).count(False) for s in bx]
			yl = [list(map(lambda s:s==0,s.data)).count(False) for s in by]
			return bx, by, xl, yl
		# training
		enc = Enc(len(s2i),300,512,3); enc.init_weight()
		dec = Dec(len(t2i),300,512*2); dec.init_weight()
		if USE_CUDA: enc,dec = enc.cuda(),dec.cuda()
		loss_func = nn.CrossEntropyLoss(ignore_index=0)
		enc_opt = optim.Adam(enc.parameters(),lr=self.lr)
		dec_opt = optim.Adam(dec.parameters(),lr=self.lr*5.0)
		for epoch in xrange(3):
			random.shuffle(data_tr); losses = []
			for i in xrange(0,len(data_tr),self.b_size):
				bx,by,xl,yl = pad(data_tr[i:i+self.b_size])
				enc.zero_grad()
				dec.zero_grad()
				output,hidden = enc(bx,xl)
				x_masks = torch.cat([Variable(BTensor(tuple(map(lambda s:s==0,t.data)))) for t in bx]).view(bx.size(0),-1)
				start = Variable(LTensor([[t2i['<s>']]*bx.size(0)])).transpose(0,1)
				preds = dec(start,hidden,by.size(1),output,x_masks,True)
				loss = loss_func(preds,by.view(-1)); loss.backward()
				torch.nn.utils.clip_grad_norm_(enc.parameters(),50.0); enc_opt.step()
				torch.nn.utils.clip_grad_norm_(dec.parameters(),50.0); dec_opt.step()
				losses.append(loss.data.tolist())
			print np.mean(losses); losses = []
		# show attn
		x,y = random.choice(data_tr)
		output,hidden = enc(x,[x.size(1)])
		p,attn = dec.decode(hidden,output,t2i)
		x = [i2s[i] for i in x.data.tolist()[0]]
		p = [i2t[i] for i in p.data.tolist()]
		if USE_CUDA: attn = attn.cpu()
		fig = plt.figure()
		ax = fig.add_subplot(111)
		cax = ax.matshow(attn.data.numpy(),cmap='bone')
		fig.colorbar(cax)
		ax.set_xticklabels(['']+x,rotation=90)
		ax.set_yticklabels(['']+p)
		ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
		ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
		plt.savefig('attn.png')

if __name__ == '__main__':
	NMT().train()
