import nltk

#import sample text
from nltk.corpus import brown

#find most common suffix

#find frequncy of words
suffix_fdist = nltk.FreqDist()
for word in brown.words():
    #find frequency of suffixes
    word = word.lower()
    suffix_fdist[word[-1:]] +=1
    suffix_fdist[word[-2:]] +=1
    suffix_fdist[word[-3:]] +=1

#100 most common suffixes
suffixes = suffix_fdist.most_common(100)

#function to check common suffix and proceeding word in sentence and tags for proceeding words
def pos_features(sentence, i, history):
    features = {"suffix(1)": sentence[i][-1:],
                "suffix(2)": sentence[i][-2:],
                "suffix(3)": sentence[i][-3:]}
    if i == 0:
        features["prev-word"] = "<START>"
        features["prev-tag"] = "<START>"
    else:
        features["prev-word"] = sentence[i-1]
        features["prev-tag"] = history[i-1]
    return features

#sequence classifier, uses tags from classifier
class ConsecutivePosTagger(nltk.TaggerI):
    def __init__(self, train_sentences):
        train_set = []
        for tagged_sentence in train_sentences:
            untagged_sentence = nltk.tag.untag(tagged_sentence)
            history = []
            for i, (word,tag) in enumerate(tagged_sentence):
                featureset = pos_features(untagged_sentence, i, history)
                train_set.append( (featureset, tag) )
                history.append(tag)
        self.classifier = nltk.NaiveBayesClassifier.train(train_set)
    
    def tag(self, sentence):
        history = []
        for i, word in enumerate(sentence):
            featureset = pos_features(sentence, i, history)
            tag = self.classifier.classify(featureset)
            history.append(tag)
        return zip(sentence, history)

#import training data for decision tree classifier
tagged_sentences = brown.tagged_sents(categories='news')

#split train and test data
training_size = int(len(tagged_sentences)*0.2)
train_set, test_set = tagged_sentences[training_size:], tagged_sentences[:training_size]

#tag classify
tagging = ConsecutivePosTagger(train_set)

#test accuracy
print(tagging.evaluate(test_set))


