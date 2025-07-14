# %%
%pip install numpy scipy scikit-learn pandas
%pip install --no-cache-dir --force-reinstall https://dm.cs.tu-dortmund.de/nats/nats25_02_01_tfidf-0.1-py3-none-any.whl
import nats25_02_01_tfidf

# %% [markdown]
# # TF-IDF
# ## Setup our working context and load the data
# 
# In this assignment, we will work with a database of inaugural speeches of US presidents.

# %%
import numpy as np, pandas as pd, scipy
import gzip, json, urllib
file_path, _ = urllib.request.urlretrieve("https://dm.cs.tu-dortmund.de/nats/data/inaugural.json.gz")
inaugural = json.load(gzip.open(file_path,"rt"))
labels = [t[0] for t in inaugural]
speeches = [t[1] for t in inaugural]
print(labels)
print(speeches)

# %% [markdown]
# ## Build a Sparse Document-Term Matrix
# 
# Build a document-term matrix for the inaugural speeches.
# 
# Use sparse data structures, a minimum document frequency of 5, remove english stopwords.

# %%
from sklearn.feature_extraction.text import CountVectorizer # Please use this
cv = CountVectorizer(stop_words="english",lowercase=True,min_df=5)
dtm = cv.fit_transform(speeches)
vocab = cv.get_feature_names_out()
print("Document term matrix has shape", dtm.shape)
print(dtm)

# %%
nats25_02_01_tfidf.hidden_tests_4_0(vocab, dtm, speeches)

# %%
# Pretty display the data with pandas:
pd.DataFrame.sparse.from_spmatrix(dtm,index=labels,columns=vocab).head()

# %% [markdown]
# ## Most Frequent Words for Each Speech
# 
# Compute the most frequent word (except for the stopwords already removed) for each speech.

# %%
# Build a dictionary speech label to most frequent word
most_frequent = dict()
data_frame = pd.DataFrame.sparse.from_spmatrix(dtm,index=labels,columns=vocab)
for index, row in data_frame.iterrows():
    top_5_words = row.nlargest(5)
    most_frequent[index] = top_5_words
for sp, w in sorted(most_frequent.items()): print(sp, w, sep="\t")

# %%
nats25_02_01_tfidf.hidden_tests_8_0(labels, most_frequent)

# %% [markdown]
# ## TF-IDF
# 
# From the document-term matrix, compute the TF-IDF matrix. Implement the standard version of TF-IDF (`ltc`).
# 
# Be careful with 0 values, ensure that your matrix remains *sparse*. Do *not* rely on Wikipedia, it has errors.
# 
# Perform the transformation in three steps, named `tf`, `idf`, `tfidf`. First implement term frequency.

# %%

def tf(dtm: scipy.sparse.csr_matrix) -> scipy.sparse.csr_matrix:
    dtm_log = dtm.astype(np.float32)
    dtm_log.data = 1 + np.log(dtm_log.data)
    return dtm_log

# %%
# Inspect your matrix
pd.DataFrame.sparse.from_spmatrix(tf(dtm),index=labels,columns=vocab).head()

# %%
nats25_02_01_tfidf.hidden_tests_12_0(dtm, tf)

# %% [markdown]
# Implement the `idf` function.

# %%
def idf(dtm):
    dtm_ones = dtm.copy()
    dtm_ones.data[:] = 1 
    df = dtm_ones.sum(axis=0)
    num_docs = dtm.shape[0]
    idf_values = np.log(num_docs / df)
    
    return idf_values

# %%
b=(np.ones((dtm.shape[0],)) @ dtm)
print(np.log(dtm.shape[0] / b))
print(idf(dtm))

# %%
nats25_02_01_tfidf.hidden_tests_16_0(dtm, idf)

# %% [markdown]
# Now implement the full `tfidf` function, using above implementations of `df` and `idf`.
# 
# Hint: you may find `scipy.sparse.spdiags` useful to keep the computations *sparse*.
# 
# You are **not allowed** to use sklearns `TfidfVectorizer`!

# %%
def tfidf(dtm):
    """Finish the computation of standard TF-IDF with the c step"""
    _tf, _idf = tf(dtm), idf(dtm) # Must use above functions.
    pass # Your solution here

# %%
# Inspect your matrix
pd.DataFrame.sparse.from_spmatrix(tfidf(dtm),index=labels,columns=vocab).head()

# %%
nats25_02_01_tfidf.hidden_tests_20_0(dtm, tfidf)

# %% [markdown]
# ## Compare to sklearn
# 
# Now you are allowed to use `TfidfVectorizer`!
# 
# Use sklearns `TfidfVectorizer` (make sure to choose parameters appropriately). Compare the results.

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
tvect = TfidfVectorizer() # set appropriate parameters!
sktfidf = None # Store the TF-IDF result obtained via sklearn
skvocab = None # The vocabulary
pass # Your solution here

# %%
# Pretty display the data with pandas:
pd.DataFrame.sparse.from_spmatrix(sktfidf,index=labels,columns=skvocab).head()

# %%
nats25_02_01_tfidf.hidden_tests_24_0(vocab, skvocab, dtm, sktfidf)

# %% [markdown]
# ## Understand the difference
# 
# By visual inspection of the two matrixes, you will notice that they do *not* agree.
# 
# Check the [bug reports of scikit-learn](https://github.com/scikit-learn/scikit-learn/issues?q=is%3Aissue+tf-idf+is%3Aopen) for related bug reports, and check the scikit-learn documentation *carefully* to figure out the difference.
# 
# Is it better or worse? We don't know. But scikit-learn does not implement the standard approach!
# 
# But: we can easily "hack" sklearn to produce the desired result.
# 
# Hint: Use `fit`, adjust the vectorizer, and `tranform` separately.

# %%
# Work around this issue in scikit-learn
tvect2 = TfidfVectorizer() # set appropriate parameters!
sktfidf2 = None # Store the TF-IDF result obtained via sklearn
skvocab2 = None # The vocabulary
# Use fit(), adjust as necessary, transform() to get the desired result!
pass # Your solution here

# %%
# Pretty display the data with pandas:
pd.DataFrame.sparse.from_spmatrix(sktfidf2,index=labels,columns=skvocab2).head()

# %%
nats25_02_01_tfidf.hidden_tests_28_0(sktfidf, skvocab2, vocab, dtm, sktfidf2, tfidf)

# %% [markdown]
# ## Compute the Cosine Similarity Matrix
# 
# Compute the cosine similarity matrix of the speeches above.
# 
# You are not allowed to use sklearn for this.

# %%
X = tfidf(dtm) # use your own tfidf results
sim = None # Compute cosine similarities
pass # Your solution here
del X # free memory again.
print("Matrix of shape %d x %d" % sim.shape)

# %%
nats25_02_01_tfidf.hidden_tests_31_0(sim, dtm)

# %% [markdown]
# ## Find the two most similar speeches
# 
# Given the similarity matrix, find the two most similar (different) speeches.

# %%
most_similar = (None, None, None) # Store a pair of document *labels* and their similarity
pass # Your solution here
print("%s\t%s\t%g" % most_similar)

# %%
nats25_02_01_tfidf.hidden_tests_34_0(sim, labels, most_similar)


