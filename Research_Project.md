An Exploratory Project Using GloVe Word Embeddings
================
Zayaan Syed
5/3/2022

## R Markdown

**Overview** In this project supervised by Dr. Michael Luvalle, I used
the text2vec implementation of the GloVe Algorithm to understand how
certain word relationships change over time. I will use this approach on
two of Shakespeare’s Plays (*The Tragedy of King Lear* and *The Tragedy
of Romeo and Juliet*) to examine how words relating to events in each
play changes with respect to the main characters. We are interested in
main character pairs because their relationships influence how the plot
progresses. Intuitively, I suspect that some character pairs carry a
larger amount of influence (e.g., Romeo+Juliet) I will do this by seeing
how the vector sum of two characters changes in cosine similarity with
some target word per each act in each respective play. Then, I can
create a time series to visualize that change.

**GloVe Algorithm** It is an unsupervised learning algorithm for
creating vector representation for words. It takes data from
co-occurrence statistics amongst words from a corpus and outputs linear
structures of the vector space. Here is an example of a TCM (term
co-occurence matrix) GloVe would utilize:

                        the cat sat on mat
                    the  0   1   0   1  1

                    cat  1   0   1   0  0

                    sat  0   1   0   1  0

                    on   1   0   1   0  0

                    mat  1   0   0   0  0 

**Implementation** First thing we are going to need to do is load in our
text files so we can convert it to a corpus.

Each act of *The Tragedy of King Lear*:

``` r
#load packages
library(tidyverse)
```

    ## -- Attaching packages --------------------------------------- tidyverse 1.3.1 --

    ## v ggplot2 3.3.5     v purrr   0.3.4
    ## v tibble  3.1.4     v dplyr   1.0.7
    ## v tidyr   1.1.4     v stringr 1.4.0
    ## v readr   2.0.2     v forcats 0.5.1

    ## Warning: package 'ggplot2' was built under R version 4.1.2

    ## -- Conflicts ------------------------------------------ tidyverse_conflicts() --
    ## x dplyr::filter() masks stats::filter()
    ## x dplyr::lag()    masks stats::lag()

``` r
library(textreadr)
```

    ## Warning: package 'textreadr' was built under R version 4.1.3

``` r
library(tidytext)
```

    ## Warning: package 'tidytext' was built under R version 4.1.3

``` r
library(tidyr)
library("httr")
library(ggplot2)
library(tm)
```

    ## Warning: package 'tm' was built under R version 4.1.3

    ## Loading required package: NLP

    ## 
    ## Attaching package: 'NLP'

    ## The following object is masked from 'package:httr':
    ## 
    ##     content

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     annotate

``` r
library(text2vec)
```

    ## Warning: package 'text2vec' was built under R version 4.1.2

``` r
library(widyr)
```

    ## Warning: package 'widyr' was built under R version 4.1.3

``` r
kinglear_act1 <- read.delim("C:/Users/Zayaan Syed/OneDrive/Documents/KingLearAct1.txt")
kinglear_act2 <- read.delim("C:/Users/Zayaan Syed/OneDrive/Documents/kinglearact2.txt")
kinglear_act3 <- read.delim("C:/Users/Zayaan Syed/OneDrive/Documents/kinglearact3.txt")
kinglear_act4 <- read.delim("C:/Users/Zayaan Syed/OneDrive/Documents/kinglearact4.txt")
finalactkinglear <- read.delim("C:/Users/Zayaan Syed/OneDrive/Documents/finalactkinglear.txt")
```

.. and *The Tragedy of Romeo and Juliet*

``` r
randj_act1 <- read.delim("C:/Users/Zayaan Syed/OneDrive/Documents/RandJAct1.txt")
randj_act2 <- read.delim("C:/Users/Zayaan Syed/OneDrive/Documents/randjact2.txt")
randj_act3 <- read.delim("C:/Users/Zayaan Syed/OneDrive/Documents/randjact3.txt")
randj_act4 <- read.delim("C:/Users/Zayaan Syed/OneDrive/Documents/act4randj.txt")
finalactrandj <- read.delim("C:/Users/Zayaan Syed/OneDrive/Documents/romeoandjulietfinalact.txt")
```

Notice how I load in each act separately. I do this because I want to
see how word relationships change from act to act.

Additionally, unlike *The Tragedy of Romeo and Juliet*, I have not
actually read *The Tragedy of King Lear*. So, I have some idea of how
words will change in Romeo and Juliet but not in King Lear. Part of this
experiment will be to see if after reading King Lear, do the word
embeddings that change align with the plot.

Below, I will outline the general steps I did to get the cosine
similarity for each act’s word embeddings. For simplicity I will use one
act throughout this whole outline.

``` r
# open a connection to file
con <- file("C:/Users/Zayaan Syed/OneDrive/Documents/romeoandjulietfinalact.txt",open="r")
# read file contents
data1 <- readLines(con)
# close the connection
close(con)
```

I intentionally use the final act first in this experiment, because I
want to track how word embeddings change moving backwards. We get a
better sense of what words, events, and character relationships are most
interesting to look at based on the last act because it is when the
resolution of the play occurs. Most major plot issues are addressed here
(or so I suspect).

We need to create vocabulary for which we want to learn word vectors. We
do this in the following way:

``` r
# Create iterator over tokens
tokens = space_tokenizer(data1)
# Create vocabulary. Terms will be unigrams (simple words).
it = itoken(tokens, progressbar = FALSE)
vocab = create_vocabulary(it)
```

We do not want words that are so uncommon. More specifically we cannot
calculate a word vector which only appeared once in the text. So we will
only take words which appear at least five times. We can do that using
text2vec’s prune\_vocabulary() function.

``` r
vocab = prune_vocabulary(vocab, term_count_min = 5L)
```

Now we are ready to contruct the term-co-occurence matrix (TCM).

``` r
# Use our filtered vocabulary
vectorizer = vocab_vectorizer(vocab)
# use window of 5 for context words
tcm = create_tcm(it, vectorizer, skip_grams_window = 5L)
```

Now since we have the TCM matrix we can implement the GloVe algorithm.
text2vec uses a parallel stochastic gradient descent algorithm.

Now to fitting the model.

``` r
glove = GlobalVectors$new(rank = 50, x_max = 10)
wv_main = glove$fit_transform(tcm, n_iter = 10, convergence_tol = 0.01, n_threads = 8)
```

    ## INFO  [17:43:21.239] epoch 1, loss 0.1863 
    ## INFO  [17:43:21.284] epoch 2, loss 0.0872 
    ## INFO  [17:43:21.300] epoch 3, loss 0.0618 
    ## INFO  [17:43:21.303] epoch 4, loss 0.0476 
    ## INFO  [17:43:21.306] epoch 5, loss 0.0381 
    ## INFO  [17:43:21.308] epoch 6, loss 0.0311 
    ## INFO  [17:43:21.312] epoch 7, loss 0.0258 
    ## INFO  [17:43:21.316] epoch 8, loss 0.0217 
    ## INFO  [17:43:21.316] epoch 9, loss 0.0184 
    ## INFO  [17:43:21.321] epoch 10, loss 0.0157

``` r
dim(wv_main)
```

    ## [1] 119  50

The model learns two sets of word vectors - main and context.
Essentially they are the same since the model is symmetric. From
experience, learning two sets of word vectors leads to higher quality
embeddings.

``` r
wv_context = glove$components
dim(wv_context)
```

    ## [1]  50 119

The GloVe paper teaches us that it is usually better to average or take
the sum of the main and context vector:

``` r
word_vectors = wv_main + t(wv_context)
```

Now for the interesting part. I tried several vector sums of character
names to see which would give us an interesting time series. **First
major problem:** MOST of them did not even work. As in, for some reason
R would not create a word vector representation for Romeo+Juliet,
Juliet+Friar, Mercutio+Romeo, etc. The single word pair that ended up
working was Friar+Romeo. I suspect the reason I could not get any other
pairs to work is because they were not mentioned within the same context
in a meaningful enough way according to the GloVe algorithm.

``` r
Romeo = word_vectors["Friar", , drop = FALSE]
cos_sim = sim2(x = word_vectors, y = Romeo, method = "cosine", norm = "l2")
head(sort(cos_sim[,1], decreasing = TRUE), 50)
```

    ##      Friar        yet      would      Paris        not        her        are 
    ## 1.00000000 0.41891480 0.36122454 0.33381557 0.32417668 0.27606457 0.26649557 
    ##      John.      Enter       must        How        thy       doth      Chief 
    ## 0.24953494 0.23503986 0.23427854 0.23365251 0.23250723 0.21794521 0.21590822 
    ##       your       What        bid       make        did       that        but 
    ## 0.21474464 0.21378884 0.21255046 0.21152858 0.20916305 0.19867906 0.18713160 
    ##        you         me          I     master       Bal.       Give         it 
    ## 0.17380705 0.16836377 0.15560116 0.15388114 0.15034929 0.15018946 0.13633123 
    ##         no        for        The         As        and       thou       more 
    ## 0.13146072 0.13116118 0.12992934 0.11951063 0.09789601 0.09076517 0.08277589 
    ##       have         on       That         to    Juliet.          a         so 
    ## 0.08245212 0.07942369 0.07719628 0.07620153 0.07523319 0.07113265 0.07075965 
    ##       upon      thee,       some    Romeo's         an       will    Prince. 
    ## 0.06422999 0.06328455 0.06170484 0.05998031 0.05774756 0.05727778 0.05475022 
    ##       dead 
    ## 0.05362019

So one word that I find interesting and appears in the output is
“death.” All this is really saying is that there exists some
relationship between the “death” vector and “Romeo+Friar,” which makes a
lot of sense since in the final act both the main character Romeo and
Juliet die, and the Friar has a major hand in that.

At this point, I just go back and apply this same vector combination to
each preceding act. I find that the correlation between “Romeo+Friar”
and “death” in Act 5 is .16648, Act 4 is .13923, Act 3 is .12508, and
does not appear in Acts 1 and 2. With this information lets generate a
time series (even though it’s only three points):

``` r
Act <- c(3,4,5)
death.association <- c(.12508,.13923,.16648)
plot(Act,death.association)
```

![](Research_Project_files/figure-gfm/unnamed-chunk-10-1.png)<!-- -->

It makes sense that there is a positive association between “death” and
“Romeo+Friar” from Acts 3-5 since the death is frequently alluded to up
until the climax in Act 5.

I went ahead and applied the same method above to *The Tragedy of King
Lear*. The only character name pair that I could use in the model is
“Cordelia+Lear.” Again, for some reason I could not create a sum out of
any other name pair (perhaps due to some restriction within within our
corpus). I find that the only words that consistently appears within
most Acts with respect to the vector sum is “father.” I find this a bit
redundant since the word pair is father+daughter in the play. So the
cosine similarity between this word pair and “father” is nonexistent is
Act 5, .14949 in Act 4, .02297 in Act 3, .31483 in Act 2, and .04600 in
Act 1.

``` r
Acts <- c(1,2,3,4)
father.association <- c(.04600,.31483,.02297, .14949)
plot(Acts,father.association)
```

![](Research_Project_files/figure-gfm/unnamed-chunk-11-1.png)<!-- -->

According to this graph, the word association as the acts progress is
inconsistent. This makes sense when considering the play itself. King
Lear disowns his daughter Cordelia at the very beginning of the play
which sets most of the plot in motion. He then switches between
regretting that decision on and off throughout the play, so his
sentiment towards Cordelia changes. Hence, it makes sense that the word
association is unpredictable.

**Conclusion**

The biggest issue in this experiment is that I did not have the ability
to explore multiple word associations throughout each act. This is
because the corpi was very limiting. We would probably need more in the
corpus to build up the semantics of the system. Additionally, I suspect
the the GloVe algorithm limits what we can vectorize.

If I were to do this experiment again I would use less restrictive
vocabulary. One idea is using a very large historical newspaper (think
one million or more) collection so I can track more meaningful word
associations over a large time period.
