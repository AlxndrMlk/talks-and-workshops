# Business Loves Uplift: Causal Impact Evaluation With Python

This is a repository for an ODSC AI West 2026 workshop "Business Loves Uplift: Causal Impact Evaluation With Python"

## Abstract

Modern uplift modeling practice is riddled with harmful myths and misconceptions that might give businesses false confidence in ungrounded decisions.

This workshop is designed to teach you how to avoid these misconceptions and make reliable analytical and modeling choices under imperfect information.

It starts with the question that determines how reliable our analysis can be. Given the data we have and the assumptions we make, is the effect identifiable?

In the session, we work through a real-world-inspired case end to end - impact evaluation of a recommender system rolled out to field sales reps across a retail store network.

We go through three identification strategies: from a cluster-randomized A/B test, through a modern graph-based double machine learning approach with sensitivity analysis, to a quasi-experimental design.

Along the way, we discuss the advantages, disadvantages, and failure modes of each approach in real-world contexts, providing practical Python code for each of them, including critical diagnostics.

During the session you will learn:

- How to translate causal identification concepts into practical analytical decisions
- How to use Python to design and analyze three impact evaluation approaches: a cluster-randomized A/B test, a graph-based double machine learning strategy with sensitivity analysis, and a synthetic control analysis
- How to run critical diagnostics that can help us understand the reliability of our study

We'll work through an example using a hosted notebook (Colab) and open-source packages: DoWhy, EconML, cluster-experiments, and mlsynth.

## Outcomes

1 - Causal Identification Module
Objectives: Learn the critical role of causal identification in uplift / causal impact analyses, learn what the estimand is

2 - A/B Testing Module
Objectives: Learn how to design a cluster-randomized A/B test, how to estimate test power using the cluster-experiments package and how to analyze and interpret the results

3 - Double Machine Learning Module
Objectives: Learn the basics of graph-based causal identification strategies, how to estimate causal effects using Double Machine Learning, and how to stress test it against violated assumptions with DoWhy and EconML

4 - Synthetic Control Module
Objectives: Learn how to design and analyze a synthetic control quasi-experiment using mlsynth and how to diagnose it 

5 - Triangulation
Objectives: Learn how to triangulate results from different methods, including how to interpret result mismatch between different estimands

## Code

To be added