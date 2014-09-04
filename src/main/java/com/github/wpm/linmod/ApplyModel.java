package com.github.wpm.linmod;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.common.base.Charsets;
import com.google.common.io.Files;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.List;
import java.util.zip.GZIPInputStream;

/**
 * Load a gzipped JSON serialization of a model and use it to score a text file where each line is treated as a
 * separate document.
 */
public class ApplyModel {
    static private int largest(double[] ns) {
        int maxIndex = 0;
        Double max = null;
        for (int i = 0; i < ns.length; i++) {
            if (null == max || ns[i] > max) {
                maxIndex = i;
                max = ns[i];
            }
        }
        return maxIndex;
    }

    static private String formatDoubleArray(String fmt, double[] ns) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < ns.length - 1; i++) {
            sb.append(String.format(fmt, ns[i])).append(" ");
        }
        sb.append(String.format(fmt, ns[ns.length - 1]));
        return sb.toString();
    }

    public static void main(String[] args) throws IOException {
        File modelFile = new File(args[0]);
        File documentsFile = new File(args[1]);

        try (GZIPInputStream gzip = new GZIPInputStream(new FileInputStream(modelFile))) {
            ObjectMapper mapper = new ObjectMapper();
            LinearNgramModel model = mapper.readValue(gzip, LinearNgramModel.class);

            System.out.println(model);
            List<String> documents = Files.readLines(documentsFile, Charsets.UTF_8);
            for (String document : documents) {
                double[] scores = model.classLogLikelihoods(document);
                System.out.format("%d\t%s\t%s\n", largest(scores), formatDoubleArray("%.4f", scores), document);
            }
        }
    }
}
