# web-science-project-2
WS 2016 Project 2

Usage:
~~~~~~

## Part I [Python (v2.7)]:

- Check `Sentiment.py` script

Basic usage:
```bash
Sentiment.py data.csv
```

## Part II [JAVA (v1.8) + CoreNLP]:

### NOTE: CoreNLP not included in archive (~4.7 GB)

1. Build binarized dataset:

```bash
java -Xms4000m -Xmx4000m -Xmn1536m -XX:+UseConcMarkSweepGC -XX:+UseParNewGC -XX:MaxTenuringThreshold=1 -XX:SurvivorRatio=90 -XX:TargetSurvivorRatio=90 -XX:+UseCompressedOops "*" BuildTrainingSet -input train_set0.txt
```

2. Sentiment Training:

```bash
java -mx10g "*" edu.stanford.nlp.sentiment.SentimentTraining -epochs 10 -numHid 25 -trainPath binary_train_0 -devPath dev.txt -train -nthreads 8 -model sentiment_model_0.ser.gz
```

3. Run Classifier:

```bash
java -mx10g "*" LEGOClassifier -sentimentModel models/sentiment_model_0.ser.gz -file test_sets/test_set0.txt
```
