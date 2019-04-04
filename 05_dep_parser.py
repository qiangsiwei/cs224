# -*- coding: utf-8 -*-

import nltk, torch, random
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm

USE_CUDA = torch.cuda.is_available()
if USE_CUDA: torch.cuda.set_device(0)
FTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
BTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor

class TranState(object):
	def __init__(self, sent):
		self.root = ('<root>','<root>',-1)
		self.stack = [self.root]
		self.buff = [(s[0],s[1],i) for i,s in enumerate(sent)]
		self.addr = [s[0] for s in sent]+[self.root[0]]
		self.arcs = []
		self.terminal = False
	def __str__(self):
		return 'stack:{0}\nbuff:{1}'.format([str(s[0]) for s in self.stack],[str(s[0]) for s in self.buff])
	def shift(self):
		if len(self.buff) >= 1:
			self.stack.append(self.buff.pop(0))
		else: print 'empty buff'
	def L_arc(self, rel=None):
		if len(self.stack) >= 2:
			s2,s1 = self.stack[-2:]
			arc = {'gid':len(self.arcs),'head':s2[2],'addr':s1[2],'form':s1[0],'pos':s1[1]}
			if rel: arc['rel'] = rel
			self.arcs.append(arc); self.stack.pop(-2)
		elif self.stack == [self.root]: print 'element lacking'
	def R_arc(self, rel=None):
		if len(self.stack) >= 2:
			s2,s1 = self.stack[-2:]
			arc = {'gid':len(self.arcs),'head':s1[2],'addr':s2[2],'form':s2[0],'pos':s2[1]}
			if rel: arc['rel'] = rel
			self.arcs.append(arc); self.stack.pop(-1)
		elif self.stack == [self.root]: print 'element lacking'
	def get_Lmost(self, index):
		l = ['<NULL>','<NULL>',None]
		if index == None: return l
		for arc in self.arcs:
			if arc['head'] == index: l = [arc['form'],arc['pos'],arc['addr']]; break
		return l
	def get_Rmost(self, index):
		r = ['<NULL>','<NULL>',None]
		if index == None: return r
		for arc in reversed(self.arcs):
			if arc['head'] == index: r = [arc['form'],arc['pos'],arc['addr']]; break
		return r
	def is_done(self):
		return len(self.buff) == 0 and self.stack == [self.root]
	def to_tree_string(self):
		if not self.is_done(): return None
		ingredient = [[arc['form'],self.addr[arc['head']]] for arc in self.arcs]
		ingredient = ingredient[-1:]+ingredient[:-1]
		return self._make_tree(ingredient,0)
	def _make_tree(self, ingredient, i, new=True):
		treestr = '('+ingredient[i][0]+' ' if new else ''
		ingredient[i][0] = 'CHECK'
		parents,_ = list(zip(*ingredient))
		if ingredient[i][1] not in parents:
			return treestr+ingredient[i][1]
		treestr += '('+ingredient[i][1]+' '
		for head,node in enumerate(parents):
			if node == ingredient[i][1]:
				treestr += self._make_tree(ingredient,head,False)+' '
		treestr = treestr.strip()+')'
		if new: treestr += ')'
		return treestr

class Model(nn.Module):
	def __init__(self, v_size, ve_dim, t_size, te_dim, h_dim, a_dim):
		super(Model,self).__init__()
		self.v_emb = nn.Embedding(v_size,ve_dim)
		self.t_emb = nn.Embedding(t_size,te_dim)
		self.h_dim = h_dim
		self.a_dim = a_dim
		self.linear = nn.Linear((ve_dim+te_dim)*10,self.h_dim)
		self.out = nn.Linear(self.h_dim,self.a_dim)
		self.v_emb.weight.data.uniform_(-0.01,0.01)
		self.t_emb.weight.data.uniform_(-0.01,0.01)
	def forward(self, vs, ts):
		vm = self.v_emb(vs).view(vs.size(0),-1)
		tm = self.t_emb(ts).view(ts.size(0),-1)
		inputs = torch.cat([vm,tm],1)
		h1 = torch.pow(self.linear(inputs),3)
		preds = -self.out(h1)
		return F.log_softmax(preds,1)

class DepParser(object):
	def __init__(self):
		self.v_emb = 50
		self.t_emb = 10
		self.h_dim = 512
		self.b_size = 256
		self.lr = 0.001
	def train(self, fn_train='data/05/train.txt', fn_vocab='data/05/vocab.txt', fn_dev='data/05/dev.txt'):
		def proc1(line):
			sent,trans = line.split('|||')
			return nltk.pos_tag(sent.split()),trans.split()
		def proc2(line):
			return line.split('\t')[0]
		def procd(data):
			def get_feat(state):
				def seq_id(seq, x2i):
					return Variable(LTensor(map(lambda x:x2i[x] if x in x2i else x2i['<unk>'],seq)))
				wf,tf = [],[]
				for k in (-1,-2,-3):
					wf.append(state.stack[k][0]) if len(state.stack)>=abs(k) and \
					state.stack[k][0] in w2i.keys() else wf.append('<NULL>')
				for k in (0,1,2):
					wf.append(state.buff[k][0]) if len(state.buff)>=k+1 and \
					state.buff[k][0] in w2i.keys() else wf.append('<NULL>')
				for k in (-1,-2,-3):
					tf.append(state.stack[k][1]) if len(state.stack)>=abs(k) and \
					state.stack[k][1] in t2i.keys() else tf.append('<NULL>')
				for k in (0,1,2):
					tf.append(state.buff[k][1]) if len(state.buff)>=k+1 and \
					state.buff[k][1] in t2i.keys() else tf.append('<NULL>')
				lc_s1 = state.get_Lmost(state.stack[-1][2]) if len(state.stack)>=1 else state.get_Lmost(None)
				rc_s1 = state.get_Rmost(state.stack[-1][2]) if len(state.stack)>=1 else state.get_Rmost(None)
				lc_s2 = state.get_Lmost(state.stack[-2][2]) if len(state.stack)>=2 else state.get_Lmost(None)
				rc_s2 = state.get_Rmost(state.stack[-2][2]) if len(state.stack)>=2 else state.get_Rmost(None)
				ws,ts,_ = zip(*[lc_s1,rc_s1,lc_s2,rc_s2]); wf.extend(ws); tf.extend(ts)
				return seq_id(wf,w2i).view(1,-1),seq_id(tf,t2i).view(1,-1)
			for tx,ty in tqdm(data):
				state = TranState(tx)
				trans = ty+['REDUCE_R']
				while len(trans):
					feat,act = get_feat(state),trans.pop(0)
					actT = Variable(LTensor([a2i[act]])).view(1,-1)
					yield [feat,actT]
					if act == 'SHIFT': state.shift()
					if act == 'REDUCE_L': state.L_arc()
					if act == 'REDUCE_R': state.R_arc()
		train = map(proc1,open(fn_train,'r').read().strip().split('\n'))
		vocab = map(proc2,open(fn_vocab,'r').read().strip().split('\n'))
		x_tr,y_tr = zip(*train); _,poss = list(zip(*[i for s in x_tr for i in s]))
		w2i = {v:i for i,v in enumerate(vocab)}
		w2i['<root>'] = len(w2i); w2i['<NULL>'] = len(w2i)
		t2i = {t:i for i,t in enumerate(set(poss))}
		t2i['<root>'] = len(t2i); t2i['<NULL>'] = len(t2i)
		a2i = {a:i for i,a in enumerate(['SHIFT','REDUCE_L','REDUCE_R'])}
		# training
		data_tr = list(procd(train))
		model = Model(len(w2i),self.v_emb,len(t2i),self.t_emb,self.h_dim,len(a2i))
		if USE_CUDA: model = model.cuda()
		loss_func = nn.NLLLoss()
		opt = optim.Adam(model.parameters(),lr=self.lr)
		random.shuffle(data_tr); losses = []
		for i,ind in enumerate(xrange(0,len(data_tr),self.b_size)):
			model.zero_grad()
			x,y = zip(*data_tr[ind:ind+self.b_size])
			ws,ts = zip(*x)
			preds = model(torch.cat(ws),torch.cat(ts))
			loss = loss_func(preds,torch.cat(y).view(-1))
			loss.backward(); opt.step()
			losses.append(loss.data.tolist())
			if (i+1)%100 == 0: print np.mean(losses); losses = []
		# testing
		test = map(proc1,open(fn_dev,'r').read().strip().split('\n'))
		test = list(procd(test)); accu = 0
		for data in test:
			(w,t),y = data[0],data[1]
			p = model(w,t).max(1)[1]
			p = p.data.tolist()[0]
			if p == y: accu += 1
		print 100.*accu/len(test)

if __name__ == '__main__':
	parser = DepParser()
	parser.train()
