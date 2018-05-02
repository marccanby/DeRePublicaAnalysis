from utils import get_text_by_author
from utils import cleanup_sent

cicero_texts_dirty = get_text_by_author('cicero/', verbose = True, flatten = True)
cicero_texts = {x: list(map(cleanup_sent, cicero_texts_dirty[x])) for x in cicero_texts_dirty}

