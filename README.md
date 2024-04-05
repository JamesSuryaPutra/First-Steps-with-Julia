# First Steps with Julia

# Description
This competition is designed to help you get started with Julia. If you are looking for a good programming language for data science, or if you are already accustomed to one language, we encourage you to also try Julia. Julia is a relatively new language for technical computing that attempts to combine the strengths of other popular programming languages. 

Here we introduce two tutorials to highlight some of Julia's features. The first is focused on the basics of the language. In the second, a complete implementation of the K Nearest Neighbor algorithm is presented, highlighting features such as parallelization and speed.

Both tutorials show that it is easy to write code in Julia, due to its intuitive syntax and design. The tutorials also describe some basics of image processing and some concepts of machine learning such as cross validation. After reviewing them, we hope you will be motivated to write your own machine learning algorithms in Julia.

This tutorial focuses on the task of identifying characters from Google Street View images. It differs from traditional character recognition because the data set contains different character fonts and the background is not the same for all images.

# Evaluation
Your model should identify the character in each image in the test set. The possible characters are 'A-Z', 'a-z', and '0-9'. 

The predictions will be evaluated using Classification Accuracy:
Accuracy = ∑Ni=1truei=predictioni / N
