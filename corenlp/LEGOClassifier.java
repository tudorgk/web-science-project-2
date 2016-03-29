
import java.io.*;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Properties;

import edu.stanford.nlp.ling.*;
import edu.stanford.nlp.process.DocumentPreprocessor;
import edu.stanford.nlp.sentiment.SentimentCoreAnnotations;
import edu.stanford.nlp.sentiment.SentimentUtils;
import org.ejml.simple.SimpleMatrix;

import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.neural.rnn.RNNCoreAnnotations;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.trees.MemoryTreebank;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.Generics;

/**
 * A wrapper class which creates a suitable pipeline for the sentiment
 * model and processes raw text.
 *<br>
 * The main program has the following options: <br>
 * <code>-parserModel</code> Which parser model to use, defaults to englishPCFG.ser.gz <br>
 * <code>-sentimentModel</code> Which sentiment model to use, defaults to sentiment.ser.gz <br>
 * <code>-file</code> Which file to process. <br>
 * <code>-stdin</code> Read one line at a time from stdin. <br>
 * <code>-output</code> pennTrees: Output trees with scores at each binarized node.  vectors: Number tree nodes and print out the vectors.  probabilities: Output the scores for different labels for each node. Defaults to printing just the root. <br>
 * <code>-filterUnknown</code> remove unknown trees from the input.  Only applies to TREES input, in which case the trees must be binarized with sentiment labels <br>
 * <code>-help</code> Print out help <br>
 *
 * @author John Bauer
 */
public class LEGOClassifier {

    private static final NumberFormat NF = new DecimalFormat("0.0000");

    static enum Output {
        PENNTREES, VECTORS, ROOT, PROBABILITIES
    }

    static enum Input {
        TEXT, TREES
    }

    private LEGOClassifier() {} // static methods

    /**
     * Sets the labels on the tree (except the leaves) to be the integer
     * value of the sentiment prediction.  Makes it easy to print out
     * with Tree.toString()
     */
    static void setSentimentLabels(Tree tree) {
        if (tree.isLeaf()) {
            return;
        }

        for (Tree child : tree.children()) {
            setSentimentLabels(child);
        }

        Label label = tree.label();
        if (!(label instanceof CoreLabel)) {
            throw new IllegalArgumentException("Required a tree with CoreLabels");
        }
        CoreLabel cl = (CoreLabel) label;
        cl.setValue(Integer.toString(RNNCoreAnnotations.getPredictedClass(tree)));
    }

    /**
     * Sets the labels on the tree to be the indices of the nodes.
     * Starts counting at the root and does a postorder traversal.
     */
    static int setIndexLabels(Tree tree, int index) {
        if (tree.isLeaf()) {
            return index;
        }

        tree.label().setValue(Integer.toString(index));
        index++;
        for (Tree child : tree.children()) {
            index = setIndexLabels(child, index);
        }
        return index;
    }

    /**
     * Outputs the vectors from the tree.  Counts the tree nodes the
     * same as setIndexLabels.
     */
    static int outputTreeVectors(PrintStream out, Tree tree, int index) {
        if (tree.isLeaf()) {
            return index;
        }

        out.print("  " + index + ":");
        SimpleMatrix vector = RNNCoreAnnotations.getNodeVector(tree);
        for (int i = 0; i < vector.getNumElements(); ++i) {
            out.print("  " + NF.format(vector.get(i)));
        }
        out.println();
        index++;
        for (Tree child : tree.children()) {
            index = outputTreeVectors(out, child, index);
        }
        return index;
    }

    /**
     * Outputs the scores from the tree.  Counts the tree nodes the
     * same as setIndexLabels.
     */
    static int outputTreeScores(PrintStream out, Tree tree, int index) {
        if (tree.isLeaf()) {
            return index;
        }

        out.print("  " + index + ":");
        SimpleMatrix vector = RNNCoreAnnotations.getPredictions(tree);
        for (int i = 0; i < vector.getNumElements(); ++i) {
            out.print("  " + NF.format(vector.get(i)));
        }
        out.println();
        index++;
        for (Tree child : tree.children()) {
            index = outputTreeScores(out, child, index);
        }
        return index;
    }

    /**
     * Outputs a tree using the output style requested
     */
    static void outputTree(PrintStream out, CoreMap sentence, List<Output> outputFormats) {
        Tree tree = sentence.get(SentimentCoreAnnotations.SentimentAnnotatedTree.class);
        for (Output output : outputFormats) {
            switch (output) {
                case PENNTREES: {
                    Tree copy = tree.deepCopy();
                    setSentimentLabels(copy);
                    out.println(copy);
                    break;
                }
                case VECTORS: {
                    Tree copy = tree.deepCopy();
                    setIndexLabels(copy, 0);
                    out.println(copy);
                    outputTreeVectors(out, tree, 0);
                    break;
                }
                case ROOT: {
                    out.println("  " + sentence.get(SentimentCoreAnnotations.SentimentClass.class));
                    break;
                }
                case PROBABILITIES: {
                    Tree copy = tree.deepCopy();
                    setIndexLabels(copy, 0);
                    out.println(copy);
                    outputTreeScores(out, tree, 0);
                    break;
                }
                default:
                    throw new IllegalArgumentException("Unknown output format " + output);
            }
        }
    }

    static final String DEFAULT_TLPP_CLASS = "edu.stanford.nlp.parser.lexparser.EnglishTreebankParserParams";

    public static void help() {
        System.err.println("Known command line arguments:");
        System.err.println("  -sentimentModel <model>: Which model to use");
        System.err.println("  -parserModel <model>: Which parser to use");
        System.err.println("  -file <filename>: Which file to process");
        System.err.println("  -stdin: Process stdin instead of a file");
        System.err.println("  -input <format>: Which format to input, TEXT or TREES.  Will not process stdin as trees.  If trees are not already binarized, they will be binarized with -tlppClass's headfinder, which means they must have labels in that treebank's tagset.");
        System.err.println("  -output <format>: Which format to output, PENNTREES, VECTORS, PROBABILITIES, or ROOT.  Multiple formats can be specified as a comma separated list.");
        System.err.println("  -filterUnknown: remove unknown trees from the input.  Only applies to TREES input, in which case the trees must be binarized with sentiment labels");
        System.err.println("  -tlppClass: a class to use for building the binarizer if using non-binarized TREES as input.  Defaults to " + DEFAULT_TLPP_CLASS);
    }

    /**
     * Reads an annotation from the given filename using the requested input.
     */
    public static List<Annotation> getAnnotations(StanfordCoreNLP tokenizer, Input inputFormat, String filename, boolean filterUnknown) {
        switch (inputFormat) {
            case TEXT: {
                String text = IOUtils.slurpFileNoExceptions(filename);
                Annotation annotation = new Annotation(text);
                tokenizer.annotate(annotation);
                List<Annotation> annotations = Generics.newArrayList();
                for (CoreMap sentence : annotation.get(CoreAnnotations.SentencesAnnotation.class)) {
                    Annotation nextAnnotation = new Annotation(sentence.get(CoreAnnotations.TextAnnotation.class));
                    nextAnnotation.set(CoreAnnotations.SentencesAnnotation.class, Collections.singletonList(sentence));
                    annotations.add(nextAnnotation);
                }
                return annotations;
            }
            case TREES: {
                List<Tree> trees;
                if (filterUnknown) {
                    trees = SentimentUtils.readTreesWithGoldLabels(filename);
                    trees = SentimentUtils.filterUnknownRoots(trees);
                } else {
                    trees = Generics.newArrayList();
                    MemoryTreebank treebank = new MemoryTreebank("utf-8");
                    treebank.loadPath(filename, null);
                    for (Tree tree : treebank) {
                        trees.add(tree);
                    }
                }

                List<Annotation> annotations = Generics.newArrayList();
                for (Tree tree : trees) {
                    CoreMap sentence = new Annotation(Sentence.listToString(tree.yield()));
                    sentence.set(TreeCoreAnnotations.TreeAnnotation.class, tree);
                    List<CoreMap> sentences = Collections.singletonList(sentence);
                    Annotation annotation = new Annotation("");
                    annotation.set(CoreAnnotations.SentencesAnnotation.class, sentences);
                    annotations.add(annotation);
                }
                return annotations;
            }
            default:
                throw new IllegalArgumentException("Unknown format " + inputFormat);
        }
    }

    public static void main(String[] args) throws IOException {
        String parserModel = null;
        String sentimentModel = null;

        String filename = null;
        boolean stdin = false;

        float successfulHits = 0;
        float totalReviews = 0;

        boolean filterUnknown = false;

        List<Output> outputFormats = Collections.singletonList(Output.ROOT);
        Input inputFormat = Input.TEXT;

        String tlppClass = DEFAULT_TLPP_CLASS;

        for (int argIndex = 0; argIndex < args.length; ) {
            if (args[argIndex].equalsIgnoreCase("-sentimentModel")) {
                sentimentModel = args[argIndex + 1];
                argIndex += 2;
            } else if (args[argIndex].equalsIgnoreCase("-parserModel")) {
                parserModel = args[argIndex + 1];
                argIndex += 2;
            } else if (args[argIndex].equalsIgnoreCase("-file")) {
                filename = args[argIndex + 1];
                argIndex += 2;
            } else if (args[argIndex].equalsIgnoreCase("-stdin")) {
                stdin = true;
                argIndex++;
            } else if (args[argIndex].equalsIgnoreCase("-input")) {
                inputFormat = Input.valueOf(args[argIndex + 1].toUpperCase());
                argIndex += 2;
            } else if (args[argIndex].equalsIgnoreCase("-output")) {
                String[] formats = args[argIndex + 1].split(",");
                outputFormats = new ArrayList<>();
                for (String format : formats) {
                    outputFormats.add(Output.valueOf(format.toUpperCase()));
                }
                argIndex += 2;
            } else if (args[argIndex].equalsIgnoreCase("-filterUnknown")) {
                filterUnknown = true;
                argIndex++;
            } else if (args[argIndex].equalsIgnoreCase("-tlppClass")) {
                tlppClass = args[argIndex + 1];
                argIndex += 2;
            } else if (args[argIndex].equalsIgnoreCase("-help")) {
                help();
                System.exit(0);
            } else {
                System.err.println("Unknown argument " + args[argIndex + 1]);
                help();
                throw new IllegalArgumentException("Unknown argument " + args[argIndex + 1]);
            }
        }

        // We construct two pipelines.  One handles tokenization, if
        // necessary.  The other takes tokenized sentences and converts
        // them to sentiment trees.
        Properties pipelineProps = new Properties();
        Properties tokenizerProps = null;
        if (sentimentModel != null) {
            pipelineProps.setProperty("sentiment.model", sentimentModel);
        }
        if (parserModel != null) {
            pipelineProps.setProperty("parse.model", parserModel);
        }
        if (inputFormat == Input.TREES) {
            pipelineProps.setProperty("annotators", "binarizer, sentiment");
            pipelineProps.setProperty("customAnnotatorClass.binarizer", "edu.stanford.nlp.pipeline.BinarizerAnnotator");
            pipelineProps.setProperty("binarizer.tlppClass", tlppClass);
            pipelineProps.setProperty("enforceRequirements", "false");
        } else {
            pipelineProps.setProperty("annotators", "parse, sentiment");
            pipelineProps.setProperty("enforceRequirements", "false");
            tokenizerProps = new Properties();
            tokenizerProps.setProperty("annotators", "tokenize, ssplit");
        }

        if (stdin && tokenizerProps != null) {
            tokenizerProps.setProperty("ssplit.eolonly", "true");
        }

        int count = 0;
        if (filename != null) count++;
        if (stdin) count++;
        if (count > 1) {
            throw new IllegalArgumentException("Please only specify one of -file, or -stdin");
        }
        if (count == 0) {
            throw new IllegalArgumentException("Please specify either -file, or -stdin");
        }

        StanfordCoreNLP tokenizer = (tokenizerProps == null) ? null : new StanfordCoreNLP(tokenizerProps);

        // Add in sentiment
        Properties props = new Properties();
        props.put("annotators", "tokenize, ssplit, pos, lemma, ner, parse, dcoref, sentiment");
        if (sentimentModel != null) {
            props.setProperty("sentiment.model", sentimentModel);
        }

        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);

        if (filename != null) {
            // Process a file.  The pipeline will do tokenization, which
            // means it will split it into sentences as best as possible
            // with the tokenizer.

            // we'll do it like we did when building the binary data set by having chunks of texts and getting the
            // review score and the review (with multiple scentences)
            String text = IOUtils.slurpFileNoExceptions(filename);
            String[] chunks = text.split("\\n\\s*\\n+"); // need blank line to make a new chunk

            for (String chunk : chunks) {
                if (chunk.trim().isEmpty()) {
                    continue;
                }

                // The expected format is that line 0 will be the text of the
                // sentence, and each subsequence line, if any, will be a value
                // followed by the sequence of tokens that get that value.

                // Here we take the first line and tokenize it as one sentence.
                String[] lines = chunk.trim().split("\\n");

                int reviewScore = Integer.parseInt(lines[0]);
                try {
                    String reviewText = lines[1];

                    Annotation reviewAnnotation = new Annotation(reviewText);
                    // run all the selected Annotators on this text
                    pipeline.annotate(reviewAnnotation);
                    int computedReviewScore = 0;
                    int[] scoreVector = {0,0,0,0,0};
                    float sentenceCount = 0;

                    List<CoreMap> sentences = reviewAnnotation.get(CoreAnnotations.SentencesAnnotation.class);
                    if (sentences != null && ! sentences.isEmpty()) {
                        for (int i = 0; i<sentences.size(); i++){
                            Tree tree = sentences.get(i).get(SentimentCoreAnnotations.SentimentAnnotatedTree.class);
                            int score = RNNCoreAnnotations.getPredictedClass(tree);
                            // increase review score vector
                            scoreVector[score]++;
                            sentenceCount++;
                        }
                    }

                    // calculate the overall review score
                    float avgReviewScore = (scoreVector[0] * (-2) + scoreVector[1] * (-1) +
                            scoreVector[2] * (0) +
                            scoreVector[3] * (1)+ scoreVector[4] * (2)) / sentenceCount;

                    if (avgReviewScore <= -0.5f){
                        computedReviewScore = 1;
                    } else if (avgReviewScore > -0.5f && avgReviewScore < 0.5f){
                        computedReviewScore = 2;
                    } else {
                        computedReviewScore = 3;
                    }

                    if (computedReviewScore == reviewScore){
                        successfulHits += 1;
                    }
                    System.out.println("predicition: " + computedReviewScore + " - real score: " + reviewScore);
                    //increase the totalReview Counter
                    totalReviews += 1;
                    System.out.println("Review nr: " + totalReviews);
                    System.out.println(reviewText);
                }catch (Exception e){
                    System.out.println(e.toString());
                }

                System.out.println((float) successfulHits / totalReviews);
                System.out.println();
            }
        } else {
            // Process stdin.  Each line will be treated as a single sentence.
            System.err.println("Reading in text from stdin.");
            System.err.println("Please enter one sentence per line.");
            System.err.println("Processing will end when EOF is reached.");
            BufferedReader reader = IOUtils.readerFromStdin("utf-8");

            for (String line; (line = reader.readLine()) != null; ) {
                line = line.trim();
                if (line.length() > 0) {
                    Annotation annotation = tokenizer.process(line);
                    pipeline.annotate(annotation);
                    for (CoreMap sentence : annotation.get(CoreAnnotations.SentencesAnnotation.class)) {
                        outputTree(System.out, sentence, outputFormats);
                    }
                } else {
                    // Output blank lines for blank lines so the tool can be
                    // used for line-by-line text processing
                    System.out.println("");
                }
            }

        }
    }

}
