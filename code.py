# DE RE PUBLICA


##### Step 1: Read in and clean up text ####
from cltk.corpus.latin import latinlibrary
from string import digits

def cleanup_sent(sent, lower=True, brackets=2, remove_oc_parent=True, convert_hyphens = True):

	def filterer(snt):
		return list(filter(lambda x: x != "", snt))

	s = filterer(sent)

	if lower:
		s = filterer(list(map(lambda x: x.lower(), s)))

	def remove_numbers_helper(wd):
		remove_digits = str.maketrans('', '', digits)
		t = wd.translate(remove_digits)
		t = t.replace("\t  ","")
		return t
	s=filterer(list(map(remove_numbers_helper, s)))

	if brackets == 1 or brackets == 2:
		for i in range(len(s)-1):
			if s[i] == "&" and s[i+1] == "lt" and s[i+2] == ";":
				s[i] = "<"
				s[i+1] = ""
				s[i+2] = ""
			elif "&lt" in s[i] and s[i+1] == ";":
				s[i] = s[i].replace("&lt", "<")
				s[i+1] = ""
			elif "&gt" in s[i] and s[i+1] == ";":
				s[i] = s[i].replace("&gt", ">")
				s[i+1] = ""
		s = filterer(s)

	if brackets == 2:
		for i in range(len(s)):
			if "<" in s[i]:
				s[i]=s[i].replace("<", "")
			elif ">" in s[i]:
				s[i]=s[i].replace(">", "")
		s = filterer(s)

	if remove_oc_parent:
		for i in range(len(s)-1):
			if s[i] == "(" and s[i+1] == ")" or s[i] == "[" and s[i+1] == "]":
				s[i] = ""
				s[i+1] = ""
		s = filterer(s)

	if convert_hyphens:
		for i in range(len(s)-1):
			if s[i][-2:] == "&#" and s[i+1] == ";":
				s[i] = s[i].replace("&#", "")
				s[i+1] = "#"
		s= filterer(s)

	return s

def sentence_joiner(sent):
	punc = ["!", ",", ".", ";", "?", ">", ")", ":", "'", '"', "]"]
	backwards_punc = ["<", "(", "["]

	def backwards_punc_tester(wd):
		for punct in backwards_punc:
			if punct in wd:
				return True
		return False

	s = ""
	for i in range(len(sent)):
		if i != 0 and sent[i] not in punc and not backwards_punc_tester(sent[i-1]):
			s = s + " "
		s = s + sent[i]

	s = s.replace(" -", "")

	return s

def sentence_stringer(sentences):
	s = " ".join(sentences)
	return s

files = latinlibrary.fileids()

def get_text_by_author(author):
	cicero_files = [f for f in files if f[:len(author)] == author]

	cicero_text = {}
	for f in cicero_files:
		print(f)

		def are_english_wds(arr):
			eng = ['the', 'latin', 'library', 'the', 'classics', 'page']
			for wd in eng:
				if wd in arr:
					return True
			return False

		sentences = list(map(lambda x: cleanup_sent(x), latinlibrary.sents([f])))
		sentences = list(filter(lambda x: x != [] and not are_english_wds(x), sentences))
		cicero_text[f] = sentences

	cicero_unique_files = set([])
	for f in cicero_files:
		t = f[len(author):][:-4]
		s = ""
		for i in t:
			if i in "0123456789.":
				break
			else:
				s += i
		cicero_unique_files.add(s)

	cicero_text_flat = {}
	for f in cicero_unique_files:
		cicero_text_flat[f] = []
		for f2 in cicero_text: # Ordering of parts of texts is ok on the most part
			if f in f2:
				cicero_text_flat[f] += cicero_text[f2]

	return(cicero_text_flat)

cicero_texts = get_text_by_author('cicero/')
livy_texts = get_text_by_author('livy/liv.')
tacitus_texts = get_text_by_author('tacitus/tac.')

text = sentence_stringer(list(map(sentence_joiner, cicero_texts['repub'])))
with open("repub.txt", "w") as buf:
	buf.write(text)


##### Step 2: Do some exploratory analysis #####
from cltk.tag.pos import POSTag
import csv

PUNC = ["!", ",", ".", ";", "?", ">", ")", ":", "'", '"', "]", "&", "#", "<", "(", "[", " ", "*","..."]

parse_map = {"n": "noun",
				"v": "verb",
				"t": "particple",
				"a": "adjective",
				"d": "adverb",
				"c": "conjunction",
				"r": "preposition",
				"p": "pronoun",
				"m": "numeral",
				"i": "interjection",
				"e": "exclamation",
				"u": "punctuation"}
pos_names = list(parse_map.values()) + ["unknown", "-", "untagged"]

def explore_text(sentences):

	res = {"num_sentences": len(sentences), 
			"num_words" : None, 
			"num_chars" : None}

	def interpret_word(wd):
		
		if wd in PUNC:
			return (0,0)
		else:
			num_chars = 0
			for char in wd:
				if char not in PUNC:
					num_chars += 1
			return (1, num_chars)

	wd_data = list(map(lambda x: list(map(interpret_word, x)), sentences)) # [[(wd, chars), ...], ...]
	sent_data = list(map(lambda x: tuple(map(lambda y: sum(y), list(zip(*x)))), wd_data)) # [(wds, chars), ...]
	text_data = tuple(map(lambda x: sum(x), list(zip(*sent_data)))) # (wds, chars)

	res["num_words" ] = text_data[0]
	res["num_chars"] = text_data[1]

	tagger = POSTag('latin')

	def tag_sent(sent):
		r = list(tagger.tag_crf(sentence_joiner(sent)))
		words = list(filter(lambda x: interpret_word(x)[0] == 1, sent))
		#parses = list(zip(*r))[1]

		def convert_parse(parse):
			if parse == "Unk":
				return "unknown"
			elif len(parse) == 9:
				if parse[0] == "-":
					return "-"
				else:
					return parse_map[parse[0].lower()]
			
		bool_arr = [False] * len(words)
		poses = ["untagged"] * len(words)

		for p in r:
			wd = p[0]
			st = p[1]
			for i in range(len(words)):
				if words[i].lower() == wd.lower() and not bool_arr[i]:
					bool_arr[i] = True
					poses[i] = convert_parse(st)
					break

		counts = [None] * len(pos_names)
		for name in pos_names:
			counts[pos_names.index(name)] = poses.count(name)

		return tuple(counts)

	sent_counts = list(map(tag_sent, sentences)) # [(5,4,1, ...), ...]
	for pos in pos_names:
		num = sum(list(zip(*sent_counts))[pos_names.index(pos)])
		res["num_" + pos] = num

	return res

def get_text_data(texts):
	res = {}
	for text in texts:
		res[text] = explore_text(texts[text])
		print(text, res[text], '\n')
	return res

cicero_data = get_text_data(cicero_texts)
tacitus_data = get_text_data(tacitus_texts)
livy_data = get_text_data(livy_texts)


def convert_dict_to_list_of_dicts_for_csv(d):
	lst = []
	for key in d:
		v = {}
		v["text_name"]= key
		for k2 in d[key]:
			v[k2]=d[key][k2]
		lst.append(v)
	return lst

cicero_data_dcts = convert_dict_to_list_of_dicts_for_csv(cicero_data)
tacitus_data_dcts = convert_dict_to_list_of_dicts_for_csv(tacitus_data)
livy_data_dcts = convert_dict_to_list_of_dicts_for_csv(livy_data)

def dictlist_to_csv(dictlist, filename):
	with open (filename, 'w') as csvfile:
		writer = csv.DictWriter(csvfile, fieldnames = list(dictlist[0].keys()))
		writer.writeheader()
		for d in dictlist:
			writer.writerow(d)

dictlist_to_csv(cicero_data_dcts, "cicero_data.csv")
dictlist_to_csv(tacitus_data_dcts, "tacitus_data.csv")
dictlist_to_csv(livy_data_dcts, "livy_data.csv")



##### Step 3. TextRank #####
import imp
keywords = imp.load_module("keywords", *imp.find_module("textrank/summa/keywords"))
from cltk.stem.lemma import LemmaReplacer
lemmatizer = LemmaReplacer('latin')
from cltk.stop.latin.stops import STOPS_LIST
import re
SyntacticUnit = imp.load_module("syntactic_unit", *imp.find_module("textrank/summa/syntactic_unit")).SyntacticUnit

stop = STOPS_LIST + ['-que', '-ve', '-ne', 'edo', 'video', 'omnis', 'vel', 'quasi', 'unde', 'nunc', 'noster', 'dico', 'volo', 'jam']

def get_tokens(sentence, remove_stop = True):

	def tag_sent(sent):
		r = list(tagger.tag_crf(sentence_joiner(sent)))
		words = sent
		#parses = list(zip(*r))[1]

		def convert_parse(parse):
			if parse == "Unk":
				return "unknown"
			elif len(parse) == 9:
				if parse[0] == "-":
					return "-"
				else:
					return parse_map[parse[0].lower()]
			
		bool_arr = [False] * len(words)
		poses = ["untagged"] * len(words)

		for p in r:
			wd = p[0]
			st = p[1]
			for i in range(len(words)):
				if words[i].lower() == wd.lower() and not bool_arr[i]:
					bool_arr[i] = True
					poses[i] = convert_parse(st)
					break

		return poses

	def lemmatize_word(word):
		w = word.lower()
		#w = jv_replacer.replace(w)
		if w == "re":
			return ["res"]
		if w == '-que' or w == '-ve' or w == '-ne':
			return [w]
		w = w.replace("'","")
		w = w.replace('"','')
		w = w.replace(".","")
		l = lemmatizer.lemmatize(w)
		if len(l) == 1:
			if (l[0] == "publica" or l[0] == "publicum" or l[0] == "publico"):
				return ["publicus"]
			if l[0] == "aliqua":
				return ["aliquis"]
			if l[0] == "sua":
				return ["suus"]
			if l[0] == "populo":
				return ["populus"]
			if l[0] == 'omne':
				return ['omnis']
		return l

	res = []
	prev_res = False
	poses = tag_sent(sentence)
	for i in range(len(sentence)):
		wd = sentence[i]
		if wd in PUNC + [""]:
			continue
		else:
			token = lemmatize_word(wd)[0]
			if remove_stop and re.sub(r'[0-9]+', '',token) in stop:
				continue
			res.append((wd, token, poses[i]))

	

	return res

def split_sent(sentence):
	wds = list(map(lambda x: x.lower(), sentence))
	wds = list(filter(lambda x: x not in PUNC, wds))
	return wds

republica_text = cicero_texts['repub']

repub_tokens = list(map(get_tokens,republica_text))
repub_tokens_sus = {}
for sent in repub_tokens:
	for wd in sent:
		if wd[0] in repub_tokens_sus:
			continue
		else:
			repub_tokens_sus[wd[0]] = SyntacticUnit(wd[0], wd[1], wd[2])

split_text = [item for sublist in list(map(split_sent, republica_text)) for item in sublist]
text_split_punc = [item.lower() for sublist in republica_text for item in sublist]

with open ("text_split_punc.txt", "w") as buf:
	buf.write(str(text_split_punc))

ranking = keywords.keywords({"tokens": repub_tokens_sus, "split_text": split_text, "text_split_punc": text_split_punc}, scores = True)


##### Step 4. Pure Frequencies #####
allowed = ['noun', 'adjective']
counts = {}
lemmas_to_word = keywords._lemmas_to_words(repub_tokens_sus)
rev_lemmas_to_word = {}
for wd in lemmas_to_word:
	for wd2 in lemmas_to_word[wd]:
		rev_lemmas_to_word[wd2] = wd
for wd in split_text:
	if wd in rev_lemmas_to_word and repub_tokens_sus[wd].tag in allowed:
		if rev_lemmas_to_word[wd] not in counts:
			counts[rev_lemmas_to_word[wd]] = 1
		else:
			counts[rev_lemmas_to_word[wd]] += 1
final_counts = list(counts.items())
final_counts.sort(key = lambda x: x[1], reverse = True)
final_counts

for i in (list(zip(ranking[0:20], final_counts[0:20]))):
	print(i[0], '\t', i[1])



##### Step 5. Word2Vec #####
import multiprocessing

params={"size": 70,
        "alpha":0.025, 
        "window":5, 
        "min_count":3, 
        "max_vocab_size":None, 
        "sample":0.001, 
        "seed":1, 
        "workers":multiprocessing.cpu_count(), 
        "min_alpha":0.0001, 
        "sg":0, 
        "hs":0, 
        "negative":5, 
        "cbow_mean":1, 
        "hashfxn":hash, 
        "iter":10, 
        "null_word":0, 
        "trim_rule":None, 
        "sorted_vocab":1, 
        "batch_words":10000, 
        "compute_loss":False, 
        "callbacks":()}

import gensim.models.word2vec as w2v

drp2vec = w2v.Word2Vec(
    **params
)

to_use = [[x[1] for x in get_tokens(sent, remove_stop = True)] for sent in republica_text]

drp2vec.build_vocab(to_use)
drp2vec.train(sentences,total_examples=drp2vec.corpus_count,epochs=100)

from sklearn import manifold
tsne = manifold.TSNE(n_components=2, random_state=0)
all_word_vectors_matrix = drp2vec.wv.syn0
all_word_vectors_matrix_2d = tsne.fit_transform(all_word_vectors_matrix)

import pandas as pd

points = pd.DataFrame(
    [
        (word, coords[0], coords[1])
        for word, coords in [
            (word, all_word_vectors_matrix_2d[drp2vec.wv.vocab[word].index])
            for word in drp2vec.wv.vocab
        ]
    ],
    columns=["word", "x", "y"]
)
points.head(40)

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_context("poster")
points.plot.scatter("x", "y", s=10, figsize=(20, 12))
plt.show()



def plot_region(x_bounds, y_bounds):
    slice = points[
        (x_bounds[0] <= points.x) &
        (points.x <= x_bounds[1]) & 
        (y_bounds[0] <= points.y) &
        (points.y <= y_bounds[1])
    ]
    
    ax = slice.plot.scatter("x", "y", s=35, figsize=(10, 8))
    for i, point in slice.iterrows():
        ax.text(point.x + 0.005, point.y + 0.005, point.word, fontsize=11)

    plt.show()

def plot_area_of_word(word):
    x = None
    y= None
    for i in range(points.shape[0]):
        if word == points['word'][i]:
            x = points.iloc[i]['x']
            y = points.iloc[i]['y']
            break
    if x is None:
        print (word + " not found.")
        return
    print ("Bounds:",(x-5,x+5),(y-5,y+5))
    plot_region(x_bounds = (x-5,x+5),y_bounds=(y-5,y+5))

plot_region((-8,-3),(7,13))
plot_region((-13,-7),(7,13))


def reg_sample(regions):
	res = [None] * len(regions)
	for i in range(len(regions)):
		reg = regions[i]
		tmp = []
		for j in range(points.shape[0]):
			x = points.iloc[j]['x']
			y = points.iloc[j]['y']
			if x > reg[0][0] and x < reg[0][1] and y > reg[1][0] and y < reg[1][1]:
				tmp.append(points.iloc[j]['word'])
		res[i] = tmp
	return res


plot_area_of_word("manilius")
plot_area_of_word("societas")
plot_area_of_word("civitas")
plot_area_of_word("quaero")
drp2vec.most_similar("socrate")
drp2vec.most_similar("tubero")
drp2vec.most_similar("civitas")

words_by_region = reg_sample([((-20,0),(0,20)), ((-20,40),(-40,-10))])

def analyze_region(region_wds):
	poses = list(map(lambda x:repub_tokens_sus[lemmas_to_word[x][0]].tag , region_wds))
	counts = [None] * len(pos_names)
	for name in pos_names:
		counts[pos_names.index(name)] = poses.count(name)
	print(counts)
	res = {}
	for pos in pos_names:
		num = counts[pos_names.index(pos)]
		res["num_" + pos] = num

	return res




##### Part 6. Text Generation
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping

with open("repub.txt", "r") as buf:
	raw_text = buf.read()


# create mapping of unique chars to integers
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))

n_chars = len(raw_text)
n_vocab = len(chars)

# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
	seq_in = raw_text[i:i + seq_length]
	seq_out = raw_text[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])

n_patterns = len(dataX)

# reshape X to be [samples, time steps, features]
X = np.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_vocab)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)

# define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# define the checkpoint
filepath="new_weights2/weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
modelCheckpoint = ModelCheckpoint(filepath, monitor = 'loss')
earlyStopping = EarlyStopping(monitor = 'loss')
callbacks_list = [modelCheckpoint, earlyStopping]

model.fit(X, y, epochs=50, batch_size=86, callbacks=callbacks_list)

# load the network weights
filename1 = "old_weightgs/weights-improvement-19-2.2728.hdf5"
filename2 = "old_weightgs/weights-improvement-19-1.6940-bigger.hdf5"
filename3 = "new_weights/weights-improvement-09-1.9432-bigger.hdf5"
model.load_weights(filename3)
model.compile(loss='categorical_crossentropy', optimizer='adam')

int_to_char = dict((i, c) for i, c in enumerate(chars))

start = 5399
pattern = dataX[start]
print ("Seed:")
seq = ''.join([int_to_char[value] for value in pattern])
print (seq)
# generate characters
res = []

class Node(object):
	def __init__(self, letter, prob, level, cur_word, cum_prob, cum_seq):
		self.prob = prob
		self.letter = letter
		self.level = level
		self.children = []
		self.cur_word = cur_word
		self.cum_prob = cum_prob
		self.cum_seq = cum_seq

result = Node(pattern[-1], 1, 0, "", 0, seq)

def is_word(wd):
	s = ""
	for c in wd:
		if c in "abcdefghijklmnopqrstuvwxyz":
			s.append(c)
	return s in text_split_punc

def start_word(wd):
	for w in text_split_punc:
		if len(w) >= len(wd):
			if w[:len(wd)] == wd:
				return True
	return False

def predict(node):
	if node.level == 1:
		return
	new_seq = node.cur_seq[-100:]
	pattern = [char_to_int[c] for c in new_seq]
	x = np.reshape(pattern, (1, len(pattern), 1))
	x = x / float(n_vocab)
	prediction = model.predict(x, verbose=0)[0]
	top_10 = prediction.argsort()[-10:][::-1]
	for i in range(len(top_10)):
		char = int_to_char[top_10[i]]
		pred = prediction[top_10[i]]
		if char not in "abcdefghijklmnopqrstuvwxyz":
			if is_word(node.cur_word):
				node.children.append(Node(char, pred, node.level +1, "", node.cum_prob + math.log(prob), node.cum_seq + char))
		else:
			if start_word(node.cur_word + char):
				node.children.append(Node(char, pred, node.level + 1, node.cur_word + char, node.cum_prob + math.log(prob), node.cum_seq + char))
	for child in node.children:
		predict(child)


for i in range(300):
	x = np.reshape(pattern, (1, len(pattern), 1))
	x = x / float(n_vocab)
	prediction = model.predict(x, verbose=0)
	index = np.argmax(prediction[0])
	result = int_to_char[index]
	seq_in = [int_to_char[value] for value in pattern]
	res.append(result)
	pattern.append(index)
	pattern = pattern[1:len(pattern)]

print("".join(res))



