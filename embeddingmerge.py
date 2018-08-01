import collections
import gensim
import inspect
import numpy as np
import pandas as pd
from types import FunctionType
from typing import Iterable

def mergeEmbeddings(embeddings: Iterable[Iterable[float]],
                    weights: Iterable[float]=None,
                    method: FunctionType=np.sum,
                    **kwargs) -> np.array:
    """
    Takes a list of embeddings and merges them into a single embeddings.
    used is to create a sentence embedding from word embeddings.

    Method should have signature (embedding, weights, **kwargs).
    """
    if len(embeddings) == 0:
        raise ValueError("embeddings input empty!")
    if weights is not None:
        if type(weights) != np.array:
            weights = np.array(weights)
        if len(weights) != len(embeddings):
            raise ValueError(
                "Incorrect # of weights! {0} weights, {1} embeddings".format(
                    len(weights), len(embeddings)))
    if method is None:
        method = np.sum
    # sane default behavior is "sum"
    if not kwargs and method == np.sum:
        if weights is not None:
            embeddings = embeddings * weights[:, np.newaxis]
            # This should not delete the actual value, 
            # only the local name (unit tested)
            weights = None
        kwargs['axis'] = 0
    try:
        if weights is None:
            return method(embeddings, **kwargs)
        return method(embeddings, weights, **kwargs)
    except TypeError as te:
        print(("\n\nError calling defined method.\n "
               + "method called: {0}\n").format(method),
            "\n\nNOTE: This can happen if you are passing weights "
            "in a function that doesn't take them as the second argument!\n"
            "Function signature was:\n\t {0}".format(inspect.signature(method)),
            ("\nArgs passed were:"
              + "\n\tembeddings: {0}"
              + "\n\tweights: {1}"
              + "\n\tkwargs: {2}").format(
                  embeddings, weights, kwargs))
        raise(te)


def sentenceEmbedding(documentCol: Iterable[Iterable[str]],
                      model: gensim.models.keyedvectors,
                      weights: Iterable[Iterable[float]]=None,
                      mergefn: FunctionType=np.sum,
                      mergefnKwargs: dict={}
                      )-> np.array:
    """
    Merge a vector of sentences (which are already split into words)
        into a sentence embedding. 
    Methods for each embedding merging methods are input as params. 
    See mergefn param for reference
    NOTE: This method expects the document column to be already split 
        and preprocessed (so in series of ['word', 'another', 'word'] format)
    """
    documentCol = pd.Series(documentCol)
    if weights is not None:
        weights = pd.Series(weights)
    # pre-allocate result for speedup
    SentenceEmbeddings = np.zeros((len(documentCol), model.vector_size))
    for row in range(len(documentCol)):
        document = documentCol.iloc[row]
        wordWeights = np.array(weights.iloc[row]) if weights is not None else None
        if wordWeights is not None and len(wordWeights) != len(document):
            raise ValueError(
                ("Incorrect # of weights on row {0} Weights:\n{1}\nWords:\n{2}"
                ).format(row, wordWeights, document))
        # pre allocate word embedding matrix for sentence 
        sentenceWords = np.zeros((len(document), model.vector_size))
        oovIndeces = []
        for word in range(len(document)):
            try:
                sentenceWords[word] = model[document[word]]
            except KeyError:
                oovIndeces.append(word)
        if oovIndeces: # if oov words, filter result
            allIndices = np.indices((len(document),))
            notOOVindices = np.setxor1d(allIndices, oovIndeces)
            sentenceWords = sentenceWords[[notOOVindices]]
            if weights is not None:
                try:
                    wordWeights = wordWeights[[notOOVindices]]
                except IndexError:
                    print(("Index error on: \n\t{0}\n with weights: \n\t {1}"
                        "\nDropped Indices: {2}"
                        ).format(document, weights[row], oovIndeces))
                    raise IndexError
        # edge cases (0 or 1 word sentence)
        if len(sentenceWords) == 0:
            continue
        elif len(sentenceWords) == 1:
            SentenceEmbeddings[row] = sentenceWords[0]
        else:
            SentenceEmbeddings[row] = mergeEmbeddings(
                    sentenceWords,
                    weights=wordWeights,
                    method=mergefn,
                    **mergefnKwargs)
    return SentenceEmbeddings


def groupedEmbedding(documentCol: Iterable[Iterable[str]],
                     groupByCol: Iterable[int],
                     model: gensim.models.keyedvectors,
                     weights: Iterable[Iterable[float]]=None,
                     sentenceMerge: FunctionType=np.sum,
                     sentenceKwarg: dict={},
                     groupMerge: FunctionType=np.sum,
                     groupKwargs: dict={},
                     verbose=True
                    ) -> dict:
    """
    Creates embeddings for Groups of documents
    Done by embedding the documents then embedding each document embedding 
    	into a single embedding per group
    For example, you can create paragraph embeddings by passing a list of split 
    	sentences with groupByCol being the paragraph number on each sentence.
    Methods for each embedding merging methods are input in sentenceMerge 
    	and groupMerge
    """
    documentCol = pd.Series(documentCol)
    groupByCol = pd.Series(groupByCol)
    if weights is not None:
        weights = pd.Series(weights)
    groupEmbeddings = {}
    for group in groupByCol.unique():
        # find documents for the group
        indices = documentCol.index[groupByCol == group].tolist()
        SentenceEmbeddings = sentenceEmbedding(
            documentCol.iloc[indices],
            model,
            weights.iloc[indices] if weights is not None else None,
            mergefn=sentenceMerge,
            mergefnKwargs=sentenceKwarg)
        try:
            groupEmbeddings[group] = mergeEmbeddings(
                SentenceEmbeddings,
                weights=None,
                method=groupMerge,
                **groupKwargs)
        except ValueError as ve:
            if verbose:
                print("Error in group Embeddings: {1}".format(SentenceEmbeddings))
                print(ve)
            pass
    return groupEmbeddings