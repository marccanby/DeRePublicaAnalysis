##### Part I Helper functions
from cltk.corpus.latin import latinlibrary
from string import digits

def get_text_by_author(author, verbose = True, flatten = True):
	"""
	Read in all texts by the author.
	"""

	# Get files for author
	files = latinlibrary.fileids()
	author_files = [f for f in files if f[:len(author)] == author]

	# Read in sentences for each file
	author_text = {}
	if verbose:
		print("Raw file names:")
	for f in author_files:
		if verbose:
			to_print = f[len(author):][:-4]
			print(to_print, end = ' ')

		author_text[f] = list(latinlibrary.sents([f]))

	if not flatten:
		return author_text

	# Get unique file names
	author_unique_files = set([])
	for f in author_files:
		t = f[len(author):][:-4]
		s = ""
		for i in t:
			if i in "0123456789.":
				break
			else:
				s += i
		author_unique_files.add(s)

	# Flatten text
	author_text_flat = {}
	if verbose:
		print ("\n\nFlattened file names:")
	for f in author_unique_files:
		if verbose:
			print(f, end = ' ')
		author_text_flat[f] = []
		for f2 in author_text: # Ordering of parts of texts is ok on the most part
			if f in f2:
				author_text_flat[f] += author_text[f2]

	return author_text_flat


def cleanup_sent(sent, lower=True, brackets=2, remove_oc_parent=True, convert_hyphens = True):
	"""
	Cleanup raw sentence.
	"""

	# Get rid of empty strings in sentence
	def filterer(snt):
		return list(filter(lambda x: x != "", snt))

	s = filterer(sent)

	# Lowercase all words
	if lower:
		s = filterer(list(map(lambda x: x.lower(), s)))

	# Remove numbers in words
	def remove_numbers_helper(wd):
		remove_digits = str.maketrans('', '', digits)
		t = wd.translate(remove_digits)
		t = t.replace("\t  ","")
		return t
	s=filterer(list(map(remove_numbers_helper, s)))

	# Collapse angle brackets
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

	# Get rid of angle brackets if desired
	if brackets == 2:
		for i in range(len(s)):
			if "<" in s[i]:
				s[i]=s[i].replace("<", "")
			elif ">" in s[i]:
				s[i]=s[i].replace(">", "")
		s = filterer(s)

	# Remove brackets with nothing in them
	if remove_oc_parent:
		for i in range(len(s)-1):
			if s[i] == "(" and s[i+1] == ")" or s[i] == "[" and s[i+1] == "]":
				s[i] = ""
				s[i+1] = ""
		s = filterer(s)

	# Convert &# hyphens to #
	if convert_hyphens:
		for i in range(len(s)-1):
			if s[i][-2:] == "&#" and s[i+1] == ";":
				s[i] = s[i].replace("&#", "")
				s[i+1] = "#"
		s= filterer(s)

	return s


def word_joiner(sent):
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


##### Part II Helper functions
from cltk.stem.lemma import LemmaReplacer
from cltk.stop.latin.stops import STOPS_LIST
from cltk.tag.pos import POSTag
import re
STOPS = STOPS_LIST + ['-que', '-ve', '-ne', 'edo', 'video', 'omnis', 'vel', 'quasi', 'unde', 'nunc', 'noster', 'dico', 'volo', 'jam', 'iste']
PUNC = ["!", ",", ".", ";", "?", ">", ")", ":", "'", '"', "]", "&", "#", "<", "(", "[", " ", "*", "..."]
PARSE_MAP = {"n": "noun",
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

def get_tokens(sentence, remove_stop = True):
	"""
	Tokenize sentence. Outputs [(original, lemmatized, pos), ...]
	"""

	tagger = POSTag('latin')
	lemmatizer = LemmaReplacer('latin')

	def tag_sent(sent):
		r = list(tagger.tag_crf(word_joiner(sent)))
		words = sent
		#parses = list(zip(*r))[1]

		def convert_parse(parse):
			if parse == "Unk":
				return "unknown"
			elif len(parse) == 9:
				if parse[0] == "-":
					return "-"
				else:
					return PARSE_MAP[parse[0].lower()]
			
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
	poses = tag_sent(sentence)
	for i in range(len(sentence)):
		wd = sentence[i]
		if wd in PUNC + [""]:
			continue
		else:
			token = lemmatize_word(wd)[0]
			if remove_stop and re.sub(r'[0-9]+', '',token) in STOPS:
				continue
			res.append((wd, token, poses[i]))

	return res


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_points(pts_2d, vocab):

	points = pd.DataFrame(
	    [
	        (word, coords[0], coords[1])
	        for word, coords in [
	            (word, pts_2d[vocab[word].index])
	            for word in vocab
	        ]
	    ],
	    columns=["word", "x", "y"]
	)
	
	sns.set_context("poster")
	points.plot.scatter("x", "y", s=10, figsize=(17, 9))
	ax = points.plot.scatter("x", "y", s=35, figsize=(17, 9))
	for i, point in points.iterrows():
		ax.text(point.x + 0.005, point.y + 0.005, point.word, fontsize=11)

	plt.show()

	return points

def plot_region(x_bounds, y_bounds, points):
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

def plot_area_of_word(word, points):
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
    plot_region(x_bounds = (x-5,x+5),y_bounds=(y-5,y+5),points=points)



##### Part III Helper functions
import random

def plot_clusters(pos, clusters):
	xs, ys = pos[:, 0], pos[:, 1]
	cluster_names = {i:str(i) for i in range(len(set(clusters)))}
	cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e', 5: '#000000'}
	df = pd.DataFrame(dict(x=xs, y=ys, label=clusters))

	#group by cluster
	groups = df.groupby('label')


	# set up plot
	fig, ax = plt.subplots(figsize=(17, 9)) # set size
	ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

	#iterate through groups to layer the plot
	#note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
	for name, group in groups:
	    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, 
	            label=cluster_names[name], color=cluster_colors[name], 
	            mec='none')
	    ax.set_aspect('auto')
	    ax.tick_params(\
	        axis= 'x',          # changes apply to the x-axis
	        which='both',      # both major and minor ticks are affected
	        bottom='on',      # ticks along the bottom edge are off
	        top='off',         # ticks along the top edge are off
	        labelbottom='on')
	    ax.tick_params(\
	        axis= 'y',         # changes apply to the y-axis
	        which='both',      # both major and minor ticks are affected
	        left='on',      # ticks along the bottom edge are off
	        top='off',         # ticks along the top edge are off
	        labelleft='on')
	    
	ax.legend(numpoints=1)  #show legend with only 1 point

	#add label in x,y position with the label as the film title
	for i in range(len(df)):
	    ax.text(df.ix[i]['x'], df.ix[i]['y'], i, size=8)  
	plt.show()
    

def sample_by_cluster(cluster_no, n_sample, clusters, original_sentences, good_idxes):
	idxes= [i for i in range(len(clusters)) if clusters[i] == cluster_no]
	random.shuffle(idxes)
	chosen = idxes[:n_sample]
	to_return = [original_sentences[good_idxes[c]] for c in chosen]
	for sent in to_return:
		print(sent)

def sample_by_region(xlim, ylim, n_sample):
	in_range = []
	for i in range(len(xs)):
		x = xs[i]
		y = ys[i]
		if x >= xlim[0] and x <= xlim[1] and y >= ylim[0] and y <= ylim[1]:
			in_range.append(i)
	random.shuffle(in_range)
	chosen = idxes[:n_sample]
	to_return = [original_sentences[good_idxes[c]] for c in chosen]
	for sent in to_return:
		print(sent)

def show_by_id(ids, n_sample=None):
	if n_sample is None:
		n_sample = len(ids)
	ids2 = [x for x in ids]
	random.shuffle(ids2)
	chosen = ids2[:n_sample]
	to_return = [original_sentences[good_idxes[c]] for c in chosen]
	for sent in to_return:
		print(sent)

