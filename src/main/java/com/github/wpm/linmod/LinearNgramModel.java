package com.github.wpm.linmod;

import com.github.wpm.NgramTfIdf;
import com.github.wpm.TfIdf;

import java.util.*;

/**
 * Linear n-gram model
 * This model's vocabulary provides integer indexes into tables of weights and biases that can be linearly combined to
 * calculate class likelihoods. The features are tf-idf scores for n-grams in the text.
 */
@SuppressWarnings("UnusedDeclaration")
public class LinearNgramModel {
    private Map<String, Integer> vocabulary;
    private List<Integer> ngrams;
    private List<Double> idf;
    private List<List<Double>> weights;
    private List<Double> biases;

    /**
     * Cached representation of the idf table as a map because the Tf-Idf module requires a map
     */
    private Map<Integer, Double> idfTable;

    @Override
    public String toString() {
        return String.format("LinearNgramModel<%d features, %d class, ngrams %s>", features(), classes(), ngrams);
    }

    public int classes() {
        return biases.size();
    }

    public int features() {
        return weights.get(0).size();
    }

    /**
     * Classify a document
     *
     * @param document document to score
     * @return array of log likelihoods for the document to belong to each of the model's classes
     */
    public double[] classLogLikelihoods(String document) {
        Collection<String> terms = documentTerms(document);
        Map<String, Double> tf = TfIdf.tf(terms);
        Map<Integer, Double> itf = new HashMap<>();
        for (Map.Entry<String, Double> e : tf.entrySet()) {
            itf.put(vocabulary.get(e.getKey()), e.getValue());
        }
        Map<Integer, Double> tfIdf = TfIdf.tfIdf(itf, idfTable(), TfIdf.Normalization.COSINE);
        int m = classes();
        double[] classScores = new double[m];
        for (int i = 0; i < m; i++) {
            classScores[i] = biases.get(i);
            for (Map.Entry<Integer, Double> e : tfIdf.entrySet()) {
                int index = e.getKey();
                double tfidf = e.getValue();
                classScores[i] += weights.get(i).get(index) * tfidf;
            }
        }
        return classScores;
    }

    /**
     * Extract n-gram terms from the document of the order used by the model, discarding those that do not appear in
     * the model's vocabulary
     *
     * @param document text to extract terms from
     * @return set of terms extracted from the document
     */
    private Collection<String> documentTerms(String document) {
        Collection<String> terms = NgramTfIdf.ngramDocumentTerms(ngrams, Arrays.asList(document)).iterator().next();
        Collection<String> recognizedTerms = new ArrayList<>();
        for (String term : terms) {
            if (vocabulary.containsKey(term)) {
                recognizedTerms.add(term);
            }
        }
        return recognizedTerms;
    }

    private Map<Integer, Double> idfTable() {
        if (null == idfTable) {
            idfTable = new HashMap<>();
            for (int i = 0; i < idf.size(); i++) {
                idfTable.put(i, idf.get(i));
            }
        }
        return idfTable;
    }

    public Map<String, Integer> getVocabulary() {
        return vocabulary;
    }

    public void setVocabulary(Map<String, Integer> vocabulary) {
        this.vocabulary = vocabulary;
    }

    public List<Integer> getNgrams() {
        return ngrams;
    }

    public void setNgrams(List<Integer> ngrams) {
        this.ngrams = ngrams;
    }

    public List<Double> getIdf() {
        return idf;
    }

    public void setIdf(List<Double> idf) {
        idfTable = null;
        this.idf = idf;
    }

    public List<List<Double>> getWeights() {
        return weights;
    }

    public void setWeights(List<List<Double>> weights) {
        this.weights = weights;
    }

    public List<Double> getBiases() {
        return biases;
    }

    public void setBiases(List<Double> biases) {
        this.biases = biases;
    }
}
