# -*- coding: utf-8 -*-

import torch, random, numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from copy import deepcopy
from tqdm import tqdm

USE_CUDA = torch.cuda.is_available()
if USE_CUDA: torch.cuda.set_device(0)
FTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
BTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor

class Model(nn.Module):
	def __init__(self, v_size, h_dim, o_size, w2i, dropout=0.1):
		super(Model,self).__init__()
		self.h_dim = h_dim
		self.embed = nn.Embedding(v_size,h_dim,padding_idx=0)
		self.F_gru = nn.GRU(h_dim,h_dim,batch_first=True)
		self.Q_gru = nn.GRU(h_dim,h_dim,batch_first=True)
		self.gate = nn.Sequential(\
			nn.Linear(h_dim*4,h_dim),nn.Tanh(),nn.Linear(h_dim,1),nn.Sigmoid())
		self.attn_gcell = nn.GRUCell(h_dim,h_dim)
		self.mem_gcell = nn.GRUCell(h_dim,h_dim)
		self.ans_gcell = nn.GRUCell(h_dim*2,h_dim)
		self.ans_fc = nn.Linear(h_dim,o_size)
		self.dropout = nn.Dropout(dropout)
		self.w2i = w2i
	def init_hidden(self, inputs):
		hidden = Variable(torch.zeros(1,inputs.size(0),self.h_dim))
		return hidden.cuda() if USE_CUDA else hidden
	def init_weight(self):
		nn.init.xavier_uniform_(self.embed.state_dict()['weight'])
		for name,param in self.F_gru.state_dict().items():
			if 'weight' in name: nn.init.xavier_normal_(param)
		for name,param in self.Q_gru.state_dict().items():
			if 'weight' in name: nn.init.xavier_normal_(param)
		for name,param in self.gate.state_dict().items():
			if 'weight' in name: nn.init.xavier_normal_(param)
		for name,param in self.attn_gcell.state_dict().items():
			if 'weight' in name: nn.init.xavier_normal_(param)
		for name,param in self.mem_gcell.state_dict().items():
			if 'weight' in name: nn.init.xavier_normal_(param)
		for name,param in self.ans_gcell.state_dict().items():
			if 'weight' in name: nn.init.xavier_normal_(param)
		nn.init.xavier_normal_(self.ans_fc.state_dict()['weight'])
		self.ans_fc.bias.data.fill_(0)
	def forward(self, facts, fms, Qs, qms, n_dec, epi=3, is_training=False):
		# Facts
		enc_facts = []
		for fact,fm in zip(facts,fms):
			embed = self.embed(fact)
			if is_training: embed = self.dropout(embed)
			hidden = self.init_hidden(fact)
			outputs,_ = self.F_gru(embed,hidden)
			hidden = [o[fm[i].data.tolist().count(0)-1] for i,o in enumerate(outputs)]
			enc_facts.append(torch.cat(hidden).view(fact.size(0),-1).unsqueeze(0))
		enc_facts = torch.cat(enc_facts)
		# Qs
		embed = self.embed(Qs)
		if is_training: embed = self.dropout(embed)
		hidden = self.init_hidden(Qs)
		outputs,hidden = self.Q_gru(embed,hidden)
		enc_Q = torch.cat([o[qms[i].data.tolist().count(0)-1] for i,o in enumerate(outputs)]).view(Qs.size(0),-1)
		# Mem
		mem = enc_Q
		B = enc_facts.size(0)
		T = enc_facts.size(1)
		for i in xrange(epi):
			hidden = self.init_hidden(enc_facts.transpose(0,1)[0]).squeeze(0)
			for t in xrange(T):
				z = torch.cat([enc_facts.transpose(0,1)[t]*enc_Q,\
							   enc_facts.transpose(0,1)[t]*mem,\
							   torch.abs(enc_facts.transpose(0,1)[t]-enc_Q),\
							   torch.abs(enc_facts.transpose(0,1)[t]-mem)],1)
				g_t = self.gate(z)
				hidden = g_t*self.attn_gcell(enc_facts.transpose(0,1)[t],hidden)+(1-g_t)*hidden
			mem = self.mem_gcell(hidden,mem)
		# Ans
		ans_hidden = mem
		start = Variable(LTensor([[self.w2i['<s>']]*mem.size(0)])).transpose(0,1)
		y_t_1 = self.embed(start).squeeze(1); decodes = []
		for t in xrange(n_dec):
			ans_hidden = self.ans_gcell(torch.cat([y_t_1,enc_Q],1),ans_hidden)
			decodes.append(F.log_softmax(self.ans_fc(ans_hidden),1))
		return torch.cat(decodes,1).view(B*n_dec,-1)

class DMN(object):
	def __init__(self):
		self.b_size = 64
		self.lr = 0.001
	def train(self):
		def load(fn):
			data = []; fact = []
			for line in open(fn).read().strip().split('\n'):
				ind = line.split(' ')[0]
				if ind == '1': fact = []
				if '?' in line:
					q,a,_ = line.split('\t')
					q = q.strip().replace('?','').split(' ')[1:]+['?']
					a = a.split()+['</s>'] 
					data.append([deepcopy(fact),q,a])
				else:
					fact.append(line.replace('.','').split(' ')[1:]+['</s>'])
			return data
		flat = lambda a:[c for b in a for c in b]
		data_tr = load('data/10/qa5_three-arg-relations_train.txt')
		fact,q,a = zip(*data_tr)
		w2i = {w:i+4 for i,w in enumerate(set(flat(flat(fact))+flat(q)+flat(a)))}
		w2i.update({'<PAD>':0,'<UNK>':1,'<s>':2,'</s>':3})
		idx = lambda x:Variable(LTensor([w2i[w] if w in w2i else w2i['<UNK>'] for w in x])).view(1,-1)
		for i in xrange(len(data_tr)):
			for j,fact in enumerate(data_tr[i][0]):
				data_tr[i][0][j] = idx(fact)
			data_tr[i][1] = idx(data_tr[i][1])
			data_tr[i][2] = idx(data_tr[i][2])
		model = Model(len(w2i)+1,80,len(w2i)+1,w2i)
		model.init_weight()
		if USE_CUDA: model = model.cuda()
		loss_func = nn.CrossEntropyLoss(ignore_index=0)
		opt = optim.Adam(model.parameters(),lr=self.lr)
		# training
		def pad(batch):
			fact,q,a = zip(*batch)
			max_f = max([len(f) for f in fact])
			max_l = max([x.size(1) for x in flat(fact)])
			max_q = max([x.size(1) for x in q])
			max_a = max([x.size(1) for x in a])
			facts,fm,Qs,As = [],[],[],[]
			for i in xrange(len(batch)):
				facts.append(torch.cat([torch.cat(\
				[fact[i][j],Variable(LTensor([w2i['<PAD>']]*(max_l-fact[i][j].size(1)))).view(1,-1)],1)\
					if fact[i][j].size(1)<max_l else fact[i][j] for j in xrange(len(fact[i]))]+\
				[Variable(LTensor([w2i['<PAD>']]*max_l)).view(1,-1) for _ in xrange(max_f-len(fact[i]))]))
				fm.append(torch.cat([Variable(BTensor(tuple(map(lambda s:s==0,t.data))),volatile=False) for t in facts[-1]]).view(facts[-1].size(0),-1))
				Qs.append(torch.cat([q[i],Variable(LTensor([w2i['<PAD>']]*(max_q-q[i].size(1)))).view(1,-1)],1) if q[i].size(1)<max_q else q[i])
				As.append(torch.cat([a[i],Variable(LTensor([w2i['<PAD>']]*(max_a-a[i].size(1)))).view(1,-1)],1) if a[i].size(1)<max_a else a[i])
			Qs,As = torch.cat(Qs),torch.cat(As)
			qm = torch.cat([Variable(BTensor(tuple(map(lambda s:s==0,t.data))),volatile=False) for t in Qs]).view(Qs.size(0),-1)
			return facts, fm, Qs, qm, As
		for epoch in xrange(50):
			random.shuffle(data_tr); losses = []
			for i in tqdm(xrange(0,len(data_tr),self.b_size)):
				facts,fm,Qs,Qm,As = pad(data_tr[i:i+self.b_size])
				model.zero_grad()
				pred = model(facts,fm,Qs,Qm,As.size(1),3,True)
				loss = loss_func(pred,As.view(-1))
				loss.backward(); opt.step()
				losses.append(loss.data.tolist())
			if np.mean(losses) < 0.01: break
			print np.mean(losses); losses = []
		# testing
		def pad_fact(fact):
			x_max = max([s.size(1) for s in fact])
			fact = torch.cat([torch.cat([fact[i],Variable(LTensor([w2i['<PAD>']]*(x_max-fact[i].size(1)))).view(1,-1)],1)\
				if fact[i].size(1)<x_max else fact[i] for i in xrange(len(fact))])
			fm = torch.cat([Variable(BTensor(tuple(map(lambda s:s==0,t.data))),volatile=False) for t in fact]).view(fact.size(0),-1)
			return fact, fm
		data_te = load('data/10/qa5_three-arg-relations_test.txt')
		for i in xrange(len(data_te)):
			for j,fact in enumerate(data_te[i][0]):
				data_te[i][0][j] = idx(fact)
			data_te[i][1] = idx(data_te[i][1])
			data_te[i][2] = idx(data_te[i][2])
		accu = 0
		for fact,Q,A in data_te:
			fact,fm = pad_fact(fact)
			qm = Variable(BTensor([0]*Q.size(1)),volatile=False).unsqueeze(0)
			A = A.squeeze(0)
			p = model([fact],[fm],Q,qm,A.size(0),3)
			if p.max(1)[1].data.tolist() == A.data.tolist(): accu += 1
		print 100.*accu/len(data_te)

if __name__ == '__main__':
	DMN().train()
