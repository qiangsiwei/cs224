# -*- coding: utf-8 -*-

import torch, random, numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from tqdm import tqdm

USE_CUDA = torch.cuda.is_available()
if USE_CUDA: torch.cuda.set_device(0)
FTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
BTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor

class Node(object):
	def __init__(self, label, word=None):
		self.label,self.word = label,word
		self.p = self.l = self.r = None
		self.is_leaf = False
	def __str__(self):
		return '[{0}:{1}]'.format(self.word,self.label) if self.is_leaf else\
			'({0} <- [{1}:{2}] -> {3})'.format(self.l,self.word,self.label,self.r)

class Tree(object):
	def __init__(self, tree_str, B_char='(', E_char=')'):
		self.B,self.E = '(',')'
		self.root = self.parse([b for a in tree_str.strip().split() for b in a])
		self.labels = self.get_labels(self.root)
	def parse(self, tokens, parent=None):
		assert (tokens[0],tokens[-1]) == (self.B,self.E)
		split = 2; cnt_B = cnt_E = 0
		if tokens[split] == self.B: cnt_B += 1; split += 1
		while cnt_B != cnt_E:
			if tokens[split] == self.B: cnt_B += 1
			if tokens[split] == self.E: cnt_E += 1
			split += 1
		node = Node(int(tokens[1])); node.parent = parent
		if cnt_B == 0:
			node.word,node.is_leaf = ''.join(tokens[2:-1]).lower(),True; return node
		node.l = self.parse(tokens[2:split],parent=node)
		node.r = self.parse(tokens[split:-1],parent=node)
		return node
	def get_words(self):
		return [node.word for node in self.get_leaves(self.root)]
	def get_labels(self, node):
		if node is None: return []
		return self.get_labels(node.l)+self.get_labels(node.r)+[node.label]
	def get_leaves(self, node):
		if node is None: return []
		return [node] if node.is_leaf else self.get_leaves(node.l)+self.get_leaves(node.r)
    
def load_trees(data='train'):
    return [Tree(ln) for ln in open('data/09/{}.txt'.format(data)).read().strip().split('\n')]

class RNTN(nn.Module):
	def __init__(self, w2i, h_dim, o_size):
		super(RNTN,self).__init__()
		self.w2i = w2i
		self.embed = nn.Embedding(len(w2i),h_dim)
		self.V = nn.ParameterList([nn.Parameter(torch.randn(h_dim*2,h_dim*2)) for _ in xrange(h_dim)])
		self.W = nn.Parameter(torch.randn(h_dim*2,h_dim))
		self.b = nn.Parameter(torch.randn(1,h_dim))
		self.W_out = nn.Linear(h_dim,o_size)
	def init_weight(self):
		nn.init.xavier_uniform_(self.embed.state_dict()['weight'])
		nn.init.xavier_uniform_(self.W_out.state_dict()['weight'])
		for param in self.V.parameters(): nn.init.xavier_uniform_(param)
		nn.init.xavier_uniform_(self.W)
		self.b.data.fill_(0)
	def tree_propagation(self, node):
		recu_T = OrderedDict(); curr = None
		if node.is_leaf:
			tensor = Variable(LTensor([self.w2i[node.word]])) if node.word in self.w2i.keys() else\
				Variable(LTensor([self.w2i['<unk>']]))
			curr = self.embed(tensor)
		else:
			recu_T.update(self.tree_propagation(node.l))
			recu_T.update(self.tree_propagation(node.r))
			cc = torch.cat([recu_T[node.l],recu_T[node.r]],1)
			xVx = torch.cat([torch.matmul(torch.matmul(cc,v),cc.transpose(0,1)) for i,v in enumerate(self.V)],1)
			Wx = torch.matmul(cc,self.W)
			curr = torch.tanh(xVx+Wx+self.b)
		recu_T[node] = curr
		return recu_T
	def forward(self, trees, root_only=False):
		if not isinstance(trees,list): trees = [trees]
		propagated = []
		for tree in trees:
			recu_T = self.tree_propagation(tree.root)
			recu_T = [recu_T[tree.root]] if root_only else\
				[tensor for node,tensor in recu_T.iteritems()]
			propagated.extend(recu_T)
		propagated = torch.cat(propagated)
		return F.log_softmax(self.W_out(propagated),1)

class Recu(object):
	def __init__(self):
		self.b_size = 20
		self.lr = 0.01
		self.lamb = 1e-5
		self.root_only = True
	def train(self):
		data_tr = load_trees('train')
		w2i = {w:i+1 for i,w in enumerate(set([w for t in data_tr for w in t.get_words()]))}; w2i['<unk>'] = 0
		i2w = {i:w for w,i in w2i.iteritems()}
		model = RNTN(w2i,30,5)
		model.init_weight()
		if USE_CUDA: model = model.cuda()
		loss_func = nn.CrossEntropyLoss()
		opt = optim.Adam(model.parameters(),lr=self.lr)
		# training
		for epoch in xrange(1):
			random.shuffle(data_tr); losses = []
			for i in tqdm(xrange(0,len(data_tr),self.b_size)):
				btr = data_tr[i:i+self.b_size]
				labels = Variable(LTensor([tr.labels[-1] for tr in btr])) if self.root_only else\
					Variable(LTensor([lable for tr in btr for lable in tr.labels]))
				model.zero_grad()
				preds = model(btr,self.root_only)
				loss = loss_func(preds,labels)
				loss.backward(); opt.step()
				losses.append(loss.data.tolist())
				if i>200: break
			print np.mean(losses); losses = []
		# testing
		data_te = load_trees('test'); accu = 0; n_node = 0
		for test in tqdm(data_te):
			preds = model(test,self.root_only)
			lables = test.labels[-1] if self.root_only else test.labels
			for p,l in zip(preds.max(1)[1].data.tolist(),labels):
				n_node += 1; accu += int(p==l)
		print 100.*accu/n_node

if __name__ == '__main__':
	Recu().train()
