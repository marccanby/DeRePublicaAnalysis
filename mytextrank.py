
class Graph:

	def __init__(nodes):
		self.graph = {}
		for node in nodes:
			self.graph[node] = set([])

	def insert_edge(tok1, tok2):
		self.graph[tok1].add(tok2)
		self.graph[tok2].add(tok1)

# graph
tokens = set([wdset[1] for sent in repub_tokens for wdset in sent])
graph = Graph(nodes)

# put edges in graph
token_mapping = {}
for sent in repub_tokens:
	for wd in sent:
		if wd[1] in token_mapping:
			continue
		else:
			token_mapping[wd[0]] = SyntacticUnit(wd[0], wd[1])