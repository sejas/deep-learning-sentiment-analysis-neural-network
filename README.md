
# Sentiment Classification & How To "Frame Problems" for a Neural Network

by Andrew Trask

- **Twitter**: @iamtrask
- **Blog**: http://iamtrask.github.io

### What You Should Already Know

- neural networks, forward and back-propagation
- stochastic gradient descent
- mean squared error
- and train/test splits

### Where to Get Help if You Need it
- Re-watch previous Udacity Lectures
- Leverage the recommended Course Reading Material - [Grokking Deep Learning](https://www.manning.com/books/grokking-deep-learning) (Check inside your classroom for a discount code)
- Shoot me a tweet @iamtrask


### Tutorial Outline:

- Intro: The Importance of "Framing a Problem" (this lesson)

- [Curate a Dataset](#lesson_1)
- [Developing a "Predictive Theory"](#lesson_2)
- [**PROJECT 1**: Quick Theory Validation](#project_1)


- [Transforming Text to Numbers](#lesson_3)
- [**PROJECT 2**: Creating the Input/Output Data](#project_2)


- Putting it all together in a Neural Network (video only - nothing in notebook)
- [**PROJECT 3**: Building our Neural Network](#project_3)


- [Understanding Neural Noise](#lesson_4)
- [**PROJECT 4**: Making Learning Faster by Reducing Noise](#project_4)


- [Analyzing Inefficiencies in our Network](#lesson_5)
- [**PROJECT 5**: Making our Network Train and Run Faster](#project_5)


- [Further Noise Reduction](#lesson_6)
- [**PROJECT 6**: Reducing Noise by Strategically Reducing the Vocabulary](#project_6)


- [Analysis: What's going on in the weights?](#lesson_7)

# Lesson: Curate a Dataset<a id='lesson_1'></a>
The cells from here until Project 1 include code Andrew shows in the videos leading up to mini project 1. We've included them so you can run the code along with the videos without having to type in everything.


```python
def pretty_print_review_and_label(i):
    print(labels[i] + "\t:\t" + reviews[i][:80] + "...")

g = open('reviews.txt','r') # What we know!
reviews = list(map(lambda x:x[:-1],g.readlines()))
g.close()

g = open('labels.txt','r') # What we WANT to know!
labels = list(map(lambda x:x[:-1].upper(),g.readlines()))
g.close()
```

**Note:** The data in `reviews.txt` we're using has already been preprocessed a bit and contains only lower case characters. If we were working from raw data, where we didn't know it was all lower case, we would want to add a step here to convert it. That's so we treat different variations of the same word, like `The`, `the`, and `THE`, all the same way.


```python
len(reviews)
```




    25000




```python
reviews[0]
```




    'bromwell high is a cartoon comedy . it ran at the same time as some other programs about school life  such as  teachers  . my   years in the teaching profession lead me to believe that bromwell high  s satire is much closer to reality than is  teachers  . the scramble to survive financially  the insightful students who can see right through their pathetic teachers  pomp  the pettiness of the whole situation  all remind me of the schools i knew and their students . when i saw the episode in which a student repeatedly tried to burn down the school  i immediately recalled . . . . . . . . . at . . . . . . . . . . high . a classic line inspector i  m here to sack one of your teachers . student welcome to bromwell high . i expect that many adults of my age think that bromwell high is far fetched . what a pity that it isn  t   '




```python
labels[0]
```




    'POSITIVE'



# Lesson: Develop a Predictive Theory<a id='lesson_2'></a>


```python
print("labels.txt \t : \t reviews.txt\n")
pretty_print_review_and_label(2137)
pretty_print_review_and_label(12816)
pretty_print_review_and_label(6267)
pretty_print_review_and_label(21934)
pretty_print_review_and_label(5297)
pretty_print_review_and_label(4998)
```

    labels.txt 	 : 	 reviews.txt
    
    NEGATIVE	:	this movie is terrible but it has some good effects .  ...
    POSITIVE	:	adrian pasdar is excellent is this film . he makes a fascinating woman .  ...
    NEGATIVE	:	comment this movie is impossible . is terrible  very improbable  bad interpretat...
    POSITIVE	:	excellent episode movie ala pulp fiction .  days   suicides . it doesnt get more...
    NEGATIVE	:	if you haven  t seen this  it  s terrible . it is pure trash . i saw this about ...
    POSITIVE	:	this schiffer guy is a real genius  the movie is of excellent quality and both e...


# Project 1: Quick Theory Validation<a id='project_1'></a>

There are multiple ways to implement these projects, but in order to get your code closer to what Andrew shows in his solutions, we've provided some hints and starter code throughout this notebook.

You'll find the [Counter](https://docs.python.org/2/library/collections.html#collections.Counter) class to be useful in this exercise, as well as the [numpy](https://docs.scipy.org/doc/numpy/reference/) library.


```python
from collections import Counter
import numpy as np
```

We'll create three `Counter` objects, one for words from postive reviews, one for words from negative reviews, and one for all the words.


```python
# Create three Counter objects to store positive, negative and total counts
positive_counts = Counter()
negative_counts = Counter()
total_counts = Counter()
```

**TODO:** Examine all the reviews. For each word in a positive review, increase the count for that word in both your positive counter and the total words counter; likewise, for each word in a negative review, increase the count for that word in both your negative counter and the total words counter.

**Note:** Throughout these projects, you should use `split(' ')` to divide a piece of text (such as a review) into individual words. If you use `split()` instead, you'll get slightly different results than what the videos and solutions show.


```python
# TODO: Loop over all the words in all the reviews and increment the counts in the appropriate counter objects
positive_counts.clear()
negative_counts.clear()
total_counts.clear()
for i, review in enumerate(reviews) :
    words = review.lower().split(' ')
    if ("NEGATIVE" == labels[i]) :
        negative_counts.update(words)
    else :
        positive_counts.update(words)
    total_counts.update(words)
    
print(total_counts)
```

    Counter({'': 1111930, 'the': 336713, '.': 327192, 'and': 164107, 'a': 163009, 'of': 145864, 'to': 135720, 'is': 107328, 'br': 101872, 'it': 96352, 'in': 93968, 'i': 87623, 'this': 76000, 'that': 73245, 's': 65361, 'was': 48208, 'as': 46933, 'for': 44343, 'with': 44125, 'movie': 44039, 'but': 42603, 'film': 40155, 'you': 34230, 'on': 34200, 't': 34081, 'not': 30626, 'he': 30138, 'are': 29430, 'his': 29374, 'have': 27731, 'be': 26957, 'one': 26789, 'all': 23978, 'at': 23513, 'they': 22906, 'by': 22546, 'an': 21560, 'who': 21433, 'so': 20617, 'from': 20498, 'like': 20276, 'there': 18832, 'her': 18421, 'or': 18004, 'just': 17771, 'about': 17374, 'out': 17113, 'if': 16803, 'has': 16790, 'what': 16159, 'some': 15747, 'good': 15143, 'can': 14654, 'more': 14251, 'she': 14223, 'when': 14182, 'very': 14069, 'up': 13291, 'time': 12724, 'no': 12717, 'even': 12651, 'my': 12503, 'would': 12436, 'which': 12047, 'story': 11988, 'only': 11918, 'really': 11738, 'see': 11478, 'their': 11385, 'had': 11290, 'we': 10859, 'were': 10783, 'me': 10773, 'well': 10659, 'than': 9919, 'much': 9763, 'get': 9309, 'bad': 9308, 
    ...
     'spotlessly': 1, 'fastidiously': 1, 'superhu': 1, 'inchworms': 1, 'yez': 1, 'accelerant': 1, 'marinaro': 1, 'lagomorph': 1, 'billionare': 1, 'hightly': 1, 'ziller': 1, 'deamon': 1, 'yaks': 1, 'hoodies': 1, 'insulation': 1, 'mwuhahahaa': 1, 'slagged': 1, 'bellwood': 1, 'pressurized': 1, 'malkovitchesque': 1, 'muppified': 1, 'whelk': 1, 'hued': 1})


Run the following two cells to list the words used in positive reviews and negative reviews, respectively, ordered from most to least commonly used. 


```python
# Examine the counts of the most common words in positive reviews
positive_counts.most_common()
```




    [('', 550468),
     ('the', 173324),
     ('.', 159654),
     ('and', 89722),
     ('a', 83688),
     ('of', 76855),
     ('to', 66746),
     ('is', 57245),
     ('in', 50215),
     ('br', 49235),
     ('it', 48025),
     ('i', 40743),
     ('that', 35630),
    ...
     ('store', 289),
     ('hoping', 288),
     ('waiting', 288),
     ...]



As you can see, common words like "the" appear very often in both positive and negative reviews. Instead of finding the most common words in positive or negative reviews, what you really want are the words found in positive reviews more often than in negative reviews, and vice versa. To accomplish this, you'll need to calculate the **ratios** of word usage between positive and negative reviews.

**TODO:** Check all the words you've seen and calculate the ratio of postive to negative uses and store that ratio in `pos_neg_ratios`. 
>Hint: the positive-to-negative ratio for a given word can be calculated with `positive_counts[word] / float(negative_counts[word]+1)`. Notice the `+1` in the denominator – that ensures we don't divide by zero for words that are only seen in positive reviews.


```python
# Create Counter object to store positive/negative ratios
pos_neg_ratios = Counter()

# TODO: Calculate the ratios of positive and negative uses of the most common words
#       Consider words to be "common" if they've been used at least 100 times
for word, positive_repetition in positive_counts.most_common() :
    pos_neg_ratios[word] = positive_repetition / float(negative_counts[word]+1)
print(pos_neg_ratios)
```

    Counter({'edie': 109.0, 'antwone': 88.0, 'din': 82.0, 'gunga': 66.0, 'goldsworthy': 65.0, 'gypo': 60.0, 'yokai': 60.0, 'paulie': 59.0, 'visconti': 51.0, 'flavia': 51.0, 'blandings': 48.0, 'kells': 48.0, 'brashear': 47.0, 'gino': 46.0, 'deathtrap': 45.0, 'harilal': 41.0, 'panahi': 41.0, 'ossessione': 39.0, 'tsui': 38.0, 'caruso': 38.0, 'sabu': 37.0, 'ahmad': 37.0, 'khouri': 36.0, 'dominick': 36.0, 'aweigh': 35.0, 'mj': 35.0, 'mcintire': 34.0, 'kriemhild': 34.0, 'blackie': 33.0, 'daisies': 33.0, 'newcombe': 33.0, 'kei': 32.0, 'trelkovsky': 32.0, 'jaffar': 31.0, 'hilliard': 31.0, 'gundam': 30.666666666666668, 'bathsheba': 30.0, 'pazu': 30.0, 'sheeta': 30.0, 'krell': 30.0, 'offside': 30.0, 'venoms': 29.0, 'fineman': 29.0, 'paine': 28.0, 'pimlico': 28.0, 'ranma': 28.0, 'ronny': 28.0, 'abhay': 27.0, 'iturbi': 26.5, 'kipling': 26.0, 'pym': 26.0, 'gabe': 25.0, 'audiard': 25.0, 'kelso': 25.0, 'milverton': 25.0, 'scalise': 25.0, 'giovanna': 24.0, 'feinstone': 24.0, 'grisby': 24.0, 'mukhsin': 24.0, 'xica': 24.0, 'moonwalker': 24.0, 'felix': 23.4, 'chikatilo': 23.0, 'togar': 23.0, 'heaton': 23.0, 'jannings': 23.0, 'luzhin': 22.5, 'miklos': 22.0, 'pidgeon': 22.0, 'soha': 22.0, 'matuschek': 22.0, 'leonora': 22.0, 'desdemona': 22.0, 'fanfan': 22.0, 'matador': 22.0, 'philo': 21.5, 'lindy': 21.0, 'firemen': 21.0, 'joss': 21.0, 'microfilm': 21.0, 'maradona': 21.0, 'reda': 21.0, 'gauri': 21.0, 'bjm': 21.0, 'capote': 20.333333333333332, 'fido': 20.25, 'quibble': 20.0, 'emory': 20.0, 'carrre': 20.0, 'prote': 20.0, 'coe': 20.0, 'mcintyre': 20.0, 'siegfried': 20.0, 'coonskin': 20.0, 'excellently': 19.666666666666668, 'clutter': 19.5, 'vance': 19.0, 'anchors': 19.0, 'versatility': 19.0, 'knockout': 19.0, 'digicorp': 19.0, 'malfique': 19.0, 'schlesinger': 19.0, 'magnus': 19.0, 'burakov': 19.0, 'ackland': 19.0, 'rvd': 19.0, 'baloo': 19.0, 'hillyer': 19.0, 'ferdie': 19.0, 'pakeezah': 19.0, 'petiot': 19.0, 'pinjar': 19.0,     
    ...
    0.037037037037037035, 'gamera': 0.03571428571428571, 'devgan': 0.034482758620689655, 'forwarding': 0.03333333333333333, 'interminable': 0.03225806451612903, 'sabretooth': 0.03225806451612903, 'deathstalker': 0.02857142857142857, 'welch': 0.02702702702702703, 'steaming': 0.02631578947368421, 'seagal': 0.026143790849673203, 'awfulness': 0.02564102564102564, 'grendel': 0.020833333333333332, 'ajay': 0.020833333333333332, 'wayans': 0.0196078431372549, 'dahmer': 0.018518518518518517, 'beowulf': 0.01639344262295082, 'thunderbirds': 0.016129032258064516, 'uwe': 0.00980392156862745, 'boll': 0.006944444444444444})


Examine the ratios you've calculated for a few words:


```python
print("Pos-to-neg ratio for 'the' = {}".format(pos_neg_ratios["the"]))
print("Pos-to-neg ratio for 'amazing' = {}".format(pos_neg_ratios["amazing"]))
print("Pos-to-neg ratio for 'terrible' = {}".format(pos_neg_ratios["terrible"]))
```

    Pos-to-neg ratio for 'the' = 1.0607993145235326
    Pos-to-neg ratio for 'amazing' = 4.022813688212928
    Pos-to-neg ratio for 'terrible' = 0.17744252873563218


Looking closely at the values you just calculated, we see the following:

* Words that you would expect to see more often in positive reviews – like "amazing" – have a ratio greater than 1. The more skewed a word is toward postive, the farther from 1 its positive-to-negative ratio  will be.
* Words that you would expect to see more often in negative reviews – like "terrible" – have positive values that are less than 1. The more skewed a word is toward negative, the closer to zero its positive-to-negative ratio will be.
* Neutral words, which don't really convey any sentiment because you would expect to see them in all sorts of reviews – like "the" – have values very close to 1. A perfectly neutral word – one that was used in exactly the same number of positive reviews as negative reviews – would be almost exactly 1. The `+1` we suggested you add to the denominator slightly biases words toward negative, but it won't matter because it will be a tiny bias and later we'll be ignoring words that are too close to neutral anyway.

Ok, the ratios tell us which words are used more often in postive or negative reviews, but the specific values we've calculated are a bit difficult to work with. A very positive word like "amazing" has a value above 4, whereas a very negative word like "terrible" has a value around 0.18. Those values aren't easy to compare for a couple of reasons:

* Right now, 1 is considered neutral, but the absolute value of the postive-to-negative rations of very postive words is larger than the absolute value of the ratios for the very negative words. So there is no way to directly compare two numbers and see if one word conveys the same magnitude of positive sentiment as another word conveys negative sentiment. So we should center all the values around netural so the absolute value fro neutral of the postive-to-negative ratio for a word would indicate how much sentiment (positive or negative) that word conveys.
* When comparing absolute values it's easier to do that around zero than one. 

To fix these issues, we'll convert all of our ratios to new values using logarithms.

**TODO:** Go through all the ratios you calculated and convert them to logarithms. (i.e. use `np.log(ratio)`)

In the end, extremely positive and extremely negative words will have positive-to-negative ratios with similar magnitudes but opposite signs.


```python
# TODO: Convert ratios to logs
for word, ratio in pos_neg_ratios.items() :
    pos_neg_ratios[word] = np.log(ratio)
print(pos_neg_ratios)
```

    Counter({'edie': 4.6913478822291435, 'antwone': 4.477336814478207, 'din': 4.406719247264253, 'gunga': 4.189654742026425, 'goldsworthy': 4.174387269895637, 'gypo': 4.0943445622221, 'yokai': 4.0943445622221, 'paulie': 4.07753744390572, 'visconti': 3.9318256327243257, 'flavia': 3.9318256327243257, 'blandings': 3.871201010907891, 'kells': 3.871201010907891, 'brashear': 3.8501476017100584, 'gino': 3.828641396489095, 'deathtrap': 3.8066624897703196, 'harilal': 3.713572066704308, 'panahi': 3.713572066704308, 'ossessione': 3.6635616461296463, 'tsui': 3.6375861597263857, 'caruso': 3.6375861597263857, 'sabu': 3.6109179126442243, 'ahmad': 3.6109179126442243, 'khouri': 3.58351893845611, 'dominick': 3.58351893845611, 'aweigh': 3.5553480614894135, 'mj': 3.5553480614894135, 'mcintire': 3.5263605246161616, 'kriemhild': 3.5263605246161616, 'blackie': 3.4965075614664802, 'daisies': 3.4965075614664802, 'newcombe': 3.4965075614664802, 'kei': 3.4657359027997265, 'trelkovsky': 3.4657359027997265, 'jaffar': 3.4339872044851463, 'hilliard': 3.4339872044851463, 'gundam': 3.4231762883809305, 'bathsheba': 
    ...
    'devgan': -3.367295829986474, 'forwarding': -3.4011973816621555, 'interminable': -3.4339872044851463, 'sabretooth': -3.4339872044851463, 'deathstalker': -3.5553480614894135, 'welch': -3.6109179126442243, 'steaming': -3.6375861597263857, 'seagal': -3.644143560272545, 'awfulness': -3.6635616461296463, 'grendel': -3.871201010907891, 'ajay': -3.871201010907891, 'wayans': -3.9318256327243257, 'dahmer': -3.9889840465642745, 'beowulf': -4.110873864173311, 'thunderbirds': -4.127134385045092, 'uwe': -4.624972813284271, 'boll': -4.969813299576001})


Examine the new ratios you've calculated for the same words from before:


```python
print("Pos-to-neg ratio for 'the' = {}".format(pos_neg_ratios["the"]))
print("Pos-to-neg ratio for 'amazing' = {}".format(pos_neg_ratios["amazing"]))
print("Pos-to-neg ratio for 'terrible' = {}".format(pos_neg_ratios["terrible"]))
```

    Pos-to-neg ratio for 'the' = 0.05902269426102881
    Pos-to-neg ratio for 'amazing' = 1.3919815802404802
    Pos-to-neg ratio for 'terrible' = -1.7291085042663878


If everything worked, now you should see neutral words with values close to zero. In this case, "the" is near zero but slightly positive, so it was probably used in more positive reviews than negative reviews. But look at "amazing"'s ratio - it's above `1`, showing it is clearly a word with positive sentiment. And "terrible" has a similar score, but in the opposite direction, so it's below `-1`. It's now clear that both of these words are associated with specific, opposing sentiments.

Now run the following cells to see more ratios. 

The first cell displays all the words, ordered by how associated they are with postive reviews. (Your notebook will most likely truncate the output so you won't actually see *all* the words in the list.)

The second cell displays the 30 words most associated with negative reviews by reversing the order of the first list and then looking at the first 30 words. (If you want the second cell to display all the words, ordered by how associated they are with negative reviews, you could just write `reversed(pos_neg_ratios.most_common())`.)

You should continue to see values similar to the earlier ones we checked – neutral words will be close to `0`, words will get more positive as their ratios approach and go above `1`, and words will get more negative as their ratios approach and go below `-1`. That's why we decided to use the logs instead of the raw ratios.


```python
# words most frequently seen in a review with a "POSITIVE" label
pos_neg_ratios.most_common()
```




    [('edie', 4.6913478822291435),
     ('antwone', 4.477336814478207),
     ('din', 4.406719247264253),
     ('gunga', 4.189654742026425),
     ('goldsworthy', 4.174387269895637),
     ('gypo', 4.0943445622221),
     ('yokai', 4.0943445622221),
     ('paulie', 4.07753744390572),
     ('visconti', 3.9318256327243257),
     ('flavia', 3.9318256327243257),
     ('blandings', 3.871201010907891),
     ('kells', 3.871201010907891),
     ('brashear', 3.8501476017100584),
     ('gino', 3.828641396489095),
     ('deathtrap', 3.8066624897703196),
     ('harilal', 3.713572066704308),
     ('panahi', 3.713572066704308),
    ....
     ('doghi', 1.9459101490553132),
     ('gordone', 1.9459101490553132),
     ('evacuee', 1.9459101490553132),
     ('jeter', 1.9459101490553132),
     ('cosimo', 1.9459101490553132),
     ('heyerdahl', 1.9459101490553132),
     ('kasturba', 1.9459101490553132),
     ...]




```python
# words most frequently seen in a review with a "NEGATIVE" label
list(reversed(pos_neg_ratios.most_common()))[0:30]

# Note: Above is the code Andrew uses in his solution video, 
#       so we've included it here to avoid confusion.
#       If you explore the documentation for the Counter class, 
#       you will see you could also find the 30 least common
#       words like this: pos_neg_ratios.most_common()[:-31:-1]
```




    [('boll', -4.969813299576001),
     ('uwe', -4.624972813284271),
     ('thunderbirds', -4.127134385045092),
     ('beowulf', -4.110873864173311),
     ('dahmer', -3.9889840465642745),
     ('wayans', -3.9318256327243257),
     ('ajay', -3.871201010907891),
     ('grendel', -3.871201010907891),
     ('awfulness', -3.6635616461296463),
     ('seagal', -3.644143560272545),
     ('steaming', -3.6375861597263857),
     ('welch', -3.6109179126442243),
     ('deathstalker', -3.5553480614894135),
     ('sabretooth', -3.4339872044851463),
     ('interminable', -3.4339872044851463),
     ('forwarding', -3.4011973816621555),
     ('devgan', -3.367295829986474),
     ('gamera', -3.332204510175204),
     ('varma', -3.295836866004329),
     ('picker', -3.295836866004329),
     ('razzie', -3.295836866004329),
     ('dreck', -3.270835563798912),
     ('unwatchable', -3.258096538021482),
     ('nada', -3.2188758248682006),
     ('stinker', -3.2088254890146994),
     ('kirkland', -3.1780538303479458),
     ('nostril', -3.1780538303479458),
     ('giamatti', -3.1780538303479458),
     ('aag', -3.1354942159291497),
     ('demi', -3.1354942159291497)]



# End of Project 1. 
## Watch the next video to see Andrew's solution, then continue on to the next lesson.

# Transforming Text into Numbers<a id='lesson_3'></a>
The cells here include code Andrew shows in the next video. We've included it so you can run the code along with the video without having to type in everything.


```python
from IPython.display import Image

review = "This was a horrible, terrible movie."

Image(filename='sentiment_network.png')
```




![png](images/output_31_0.png)




```python
review = "The movie was excellent"

Image(filename='sentiment_network_pos.png')
```




![png](images/output_32_0.png)



# Project 2: Creating the Input/Output Data<a id='project_2'></a>

**TODO:** Create a [set](https://docs.python.org/3/tutorial/datastructures.html#sets) named `vocab` that contains every word in the vocabulary.


```python
# TODO: Create set named "vocab" containing all of the words from all of the reviews
vocab = set(total_counts)
# vocab = set(total_counts.keys())
```

Run the following cell to check your vocabulary size. If everything worked correctly, it should print **74074**


```python
vocab_size = len(vocab)
print(vocab_size)
```

    74074


Take a look at the following image. It represents the layers of the neural network you'll be building throughout this notebook. `layer_0` is the input layer, `layer_1` is a hidden layer, and `layer_2` is the output layer.


```python
from IPython.display import Image
Image(filename='sentiment_network_2.png')
```




![png](images/output_38_0.png)



**TODO:** Create a numpy array called `layer_0` and initialize it to all zeros. You will find the [zeros](https://docs.scipy.org/doc/numpy/reference/generated/numpy.zeros.html) function particularly helpful here. Be sure you create `layer_0` as a 2-dimensional matrix with 1 row and `vocab_size` columns. 


```python
# TODO: Create layer_0 matrix with dimensions 1 by vocab_size, initially filled with zeros
layer_0 = np.array([np.zeros(vocab_size)])
# layer_0 = np.zeros((1, vocab_size))
```

Run the following cell. It should display `(1, 74074)`


```python
layer_0.shape
```




    (1, 74074)




```python
from IPython.display import Image
Image(filename='sentiment_network.png')
```




![png](images/output_43_0.png)



`layer_0` contains one entry for every word in the vocabulary, as shown in the above image. We need to make sure we know the index of each word, so run the following cell to create a lookup table that stores the index of every word.


```python
# Create a dictionary of words in the vocabulary mapped to index positions
# (to be used in layer_0)
word2index = {}
for i,word in enumerate(vocab):
    word2index[word] = i
    
# display the map of words to indices
word2index
```




    {'': 0,
     'accentuate': 1,
     'prettily': 2,
     'taxidriver': 3,
     'spotters': 4,
     'enki': 5,
     'rather': 6,
     'slezak': 7,
     'curdled': 8,
     'lord': 9,
     'chitchatting': 10,
     'responsiveness': 11,
     'endures': 12,
     'alienation': 13,
     'hook': 14,
     'nabakov': 15,
     'nomm': 16,
     'riedelsheimer': 17,
      ...
     'grievously': 990,
     'insight': 991,
     'fighters': 992,
     'minoring': 993,
     'scorning': 994,
     'dorf': 995,
     'desis': 996,
     'bottomline': 997,
     'manson': 998,
     'answears': 999,
     ...}



**TODO:**  Complete the implementation of `update_input_layer`. It should count 
          how many times each word is used in the given review, and then store
          those counts at the appropriate indices inside `layer_0`.


```python
def update_input_layer(review):
    """ Modify the global layer_0 to represent the vector form of review.
    The element at a given index of layer_0 should represent
    how many times the given word occurs in the review.
    Args:
        review(string) - the string of the review
    Returns:
        None
    """
    global layer_0
    # clear out previous state by resetting the layer to be all 0s
    layer_0 *= 0
    
    # TODO: count how many times each word is used in the given review and store the results in layer_0 
    counted_words = Counter(review.lower().split(' '))
    for word, count in counted_words.items() :
        word_index_in_layer = word2index[word]
        # print(word, count, word_index_in_layer)
        layer_0[0][word_index_in_layer] = count
```

Run the following cell to test updating the input layer with the first review. The indices assigned may not be the same as in the solution, but hopefully you'll see some non-zero values in `layer_0`.  


```python
update_input_layer(reviews[0])
layer_0
```




    array([[18.,  0.,  0., ...,  0.,  0.,  0.]])



**TODO:** Complete the implementation of `get_target_for_labels`. It should return `0` or `1`, 
          depending on whether the given label is `NEGATIVE` or `POSITIVE`, respectively.


```python
def get_target_for_label(label):
    """Convert a label to `0` or `1`.
    Args:
        label(string) - Either "POSITIVE" or "NEGATIVE".
    Returns:
        `0` or `1`.
    """
    # TODO: Your code here
    return 0 if "NEGATIVE" == label else 1
```

Run the following two cells. They should print out`'POSITIVE'` and `1`, respectively.


```python
labels[0]
```




    'POSITIVE'




```python
get_target_for_label(labels[0])
```




    1



Run the following two cells. They should print out `'NEGATIVE'` and `0`, respectively.


```python
labels[1]
```




    'NEGATIVE'




```python
get_target_for_label(labels[1])
```




    0



# End of Project 2. 
## Watch the next video to see Andrew's solution, then continue on to the next lesson.

# Project 3: Building a Neural Network<a id='project_3'></a>

**TODO:** We've included the framework of a class called `SentimentNetork`. Implement all of the items marked `TODO` in the code. These include doing the following:
- Create a basic neural network much like the networks you've seen in earlier lessons and in Project 1, with an input layer, a hidden layer, and an output layer. 
- Do **not** add a non-linearity in the hidden layer. That is, do not use an activation function when calculating the hidden layer outputs.
- Re-use the code from earlier in this notebook to create the training data (see `TODO`s in the code)
- Implement the `pre_process_data` function to create the vocabulary for our training data generating functions
- Ensure `train` trains over the entire corpus

### Where to Get Help if You Need it
- Re-watch earlier Udacity lectures
- Chapters 3-5 - [Grokking Deep Learning](https://www.manning.com/books/grokking-deep-learning) - (Check inside your classroom for a discount code)


```python
import time
import sys
import numpy as np

# Encapsulate our neural network in a class
class SentimentNetwork:
    def __init__(self, reviews, labels, hidden_nodes = 10, learning_rate = 0.1):
        """Create a SentimenNetwork with the given settings
        Args:
            reviews(list) - List of reviews used for training
            labels(list) - List of POSITIVE/NEGATIVE labels associated with the given reviews
            hidden_nodes(int) - Number of nodes to create in the hidden layer
            learning_rate(float) - Learning rate to use while training
        
        """
        # Assign a seed to our random number generator to ensure we get
        # reproducable results during development 
        np.random.seed(1)

        # process the reviews and their associated labels so that everything
        # is ready for training
        self.pre_process_data(reviews, labels)
        
        # Build the network to have the number of hidden nodes and the learning rate that
        # were passed into this initializer. Make the same number of input nodes as
        # there are vocabulary words and create a single output node.
        self.init_network(len(self.review_vocab),hidden_nodes, 1, learning_rate)

    def pre_process_data(self, reviews, labels):
        
        review_vocab = set()
        # TODO: populate review_vocab with all of the words in the given reviews
        #       Remember to split reviews into individual words 
        #       using "split(' ')" instead of "split()".
        review_vocab_counter = Counter()
        for i, review in enumerate(reviews) :
            review_vocab_counter.update(review.lower().split(" "))
        review_vocab = review_vocab_counter.keys()
        
        # Convert the vocabulary set to a list so we can access words via indices
        self.review_vocab = list(review_vocab)
        
        # TODO: populate label_vocab with all of the words in the given labels.
        #       There is no need to split the labels because each one is a single word.
        # Convert the label vocabulary set to a list so we can access labels via indices
        label_vocab = set(labels)
        
        self.label_vocab = list(label_vocab)
        
        # Store the sizes of the review and label vocabularies.
        self.review_vocab_size = len(self.review_vocab)
        self.label_vocab_size = len(self.label_vocab)
        print("SIZES: ",self.review_vocab_size,         self.label_vocab_size )
        
        # Create a dictionary of words in the vocabulary mapped to index positions
        self.word2index = {}
        # TODO: populate self.word2index with indices for all the words in self.review_vocab
        #       like you saw earlier in the notebook
        for i, word in enumerate(self.review_vocab):
            self.word2index[word] = i
        # Create a dictionary of labels mapped to index positions
        self.label2index = {}
        # TODO: do the same thing you did for self.word2index and self.review_vocab, 
        #       but for self.label2index and self.label_vocab instead
        for i, wordLabel in enumerate(self.label_vocab):
            self.label2index[wordLabel] = i
        
    def init_network(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Store the number of nodes in input, hidden, and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Store the learning rate
        self.learning_rate = learning_rate

        # Initialize weights
        
        # TODO: initialize self.weights_0_1 as a matrix of zeros. These are the weights between
        #       the input layer and the hidden layer.
        # DOUBT: SHAPE?
        self.weights_0_1 = np.zeros((self.input_nodes,self.hidden_nodes))
        
        # TODO: initialize self.weights_1_2 as a matrix of random values. 
        #       These are the weights between the hidden layer and the output layer.
        # DOUBT: SHAPE?
        self.weights_1_2 = np.random.normal(0.0, self.output_nodes**-0.5, 
                                                (self.hidden_nodes, self.output_nodes))
        
        # TODO: Create the input layer, a two-dimensional matrix with shape 
        #       1 x input_nodes, with all values initialized to zero
        self.layer_0 = np.zeros((1,input_nodes))
    
        
    def update_input_layer(self,review):
        # DONE: You can copy most of the code you wrote for update_input_layer 
        #       earlier in this notebook. 
        #
        #       However, MAKE SURE YOU CHANGE ALL VARIABLES TO REFERENCE
        #       THE VERSIONS STORED IN THIS OBJECT, NOT THE GLOBAL OBJECTS.
        #       For example, replace "layer_0 *= 0" with "self.layer_0 *= 0"
            # clear out previous state by resetting the layer to be all 0s
        self.layer_0 *= 0
    
        # count how many times each word is used in the given review and store the results in layer_0 
        counted_words = Counter(review.lower().split(' '))
        for word, count in counted_words.items() :
            if(word in self.word2index.keys()):
                word_index_in_layer = self.word2index[word]
                # print(word, count, word_index_in_layer)
                self.layer_0[0][word_index_in_layer] = count
                
    def get_target_for_label(self,label):
        # DONE: Copy the code you wrote for get_target_for_label 
        #       earlier in this notebook. 
        """Convert a label to `0` or `1`.
        Args:
            label(string) - Either "POSITIVE" or "NEGATIVE".
        Returns:
            `0` or `1`.
        """
        return 0 if "NEGATIVE" == label else 1
        
    def sigmoid(self,x):
        # DONE: Return the result of calculating the sigmoid activation function
        #       shown in the lectures
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_output_2_derivative(self,output):
        # DONE: Return the derivative of the sigmoid activation function, 
        #       where "output" is the original output from the sigmoid fucntion 
        # dy/dx = f(x)' = f(x) * (1 - f(x))
        return output * (1 - output)

    def train(self, training_reviews, training_labels):
        
        # make sure out we have a matching number of reviews and labels
        assert(len(training_reviews) == len(training_labels))
        
        # Keep track of correct predictions to display accuracy during training 
        correct_so_far = 0
        
        # Remember when we started for printing time statistics
        start = time.time()

        # loop through all the given reviews and run a forward and backward pass,
        # updating weights for every item
        for i in range(len(training_reviews)):
            
            # DONE: Get the next review and its correct label
            review = training_reviews[i]
            label = training_labels[i]
            
            # DONE: Implement the forward pass through the network. 
            #       That means use the given review to update the input layer, 
            #       then calculate values for the hidden layer,
            #       and finally calculate the output layer.
            # 
            #       Do not use an activation function for the hidden layer,
            #       but use the sigmoid activation function for the output layer.
            
            # Update Input layer
            self.update_input_layer(review)
            # Hidden layer
            layer_1 = self.layer_0.dot(self.weights_0_1)
            # Output layer
            layer_2 = self.sigmoid(layer_1.dot(self.weights_1_2))
            
            
            # DONE: Implement the back propagation pass here. 
            #       That means calculate the error for the forward pass's prediction
            #       and update the weights in the network according to their
            #       contributions toward the error, as calculated via the
            #       gradient descent and back propagation algorithms you 
            #       learned in class.

            ### Backward pass ###
            # Output error
            layer_2_error = layer_2 - self.get_target_for_label(label) # Output layer error is the difference between desired target and actual output.
            layer_2_delta = layer_2_error * self.sigmoid_output_2_derivative(layer_2)
            # Backpropagated error
            layer_1_error = layer_2_delta.dot(self.weights_1_2.T) # errors propagated to the hidden layer
            layer_1_delta = layer_1_error # hidden layer gradients - no nonlinearity so it's the same as the error
            # Update the weights
            self.weights_1_2 -= layer_1.T.dot(layer_2_delta) * self.learning_rate # update hidden-to-output weights with gradient descent step
            self.weights_0_1 -= self.layer_0.T.dot(layer_1_delta) * self.learning_rate # update input-to-hidden weights with gradient descent step

            
            # DONE: Keep track of correct predictions. To determine if the prediction was
            #       correct, check that the absolute value of the output error 
            #       is less than 0.5. If so, add one to the correct_so_far count.
             # Keep track of correct predictions.
            if(layer_2 >= 0.5 and label == 'POSITIVE'):
                correct_so_far += 1
            elif(layer_2 < 0.5 and label == 'NEGATIVE'):
                correct_so_far += 1
            
            # For debug purposes, print out our prediction accuracy and speed 
            # throughout the training process. 

            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0
            
            sys.stdout.write("\rProgress:" + str(100 * i/float(len(training_reviews)))[:4] \
                             + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                             + " #Correct:" + str(correct_so_far) + " #Trained:" + str(i+1) \
                             + " Training Accuracy:" + str(correct_so_far * 100 / float(i+1))[:4] + "%")
            if(i % 2500 == 0):
                print("")
    
    def test(self, testing_reviews, testing_labels):
        """
        Attempts to predict the labels for the given testing_reviews,
        and uses the test_labels to calculate the accuracy of those predictions.
        """
        
        # keep track of how many correct predictions we make
        correct = 0

        # we'll time how many predictions per second we make
        start = time.time()

        # Loop through each of the given reviews and call run to predict
        # its label. 
        for i in range(len(testing_reviews)):
            pred = self.run(testing_reviews[i])
            if(pred == testing_labels[i]):
                correct += 1
            
            # For debug purposes, print out our prediction accuracy and speed 
            # throughout the prediction process. 

            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0
            
            sys.stdout.write("\rProgress:" + str(100 * i/float(len(testing_reviews)))[:4] \
                             + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                             + " #Correct:" + str(correct) + " #Tested:" + str(i+1) \
                             + " Testing Accuracy:" + str(correct * 100 / float(i+1))[:4] + "%")
    
    def run(self, review):
        """
        Returns a POSITIVE or NEGATIVE prediction for the given review.
        """
        # DONE: Run a forward pass through the network, like you did in the
        #       "train" function. That means use the given review to 
        #       update the input layer, then calculate values for the hidden layer,
        #       and finally calculate the output layer.
        #
        #       Note: The review passed into this function for prediction 
        #             might come from anywhere, so you should convert it 
        #             to lower case prior to using it.
        self.update_input_layer(review.lower())
        layer1 = self.layer_0.dot(self.weights_0_1)
        layer2 = self.sigmoid(layer1.dot(self.weights_1_2))
        
        # DONE: The output layer should now contain a prediction. 
        #       Return `POSITIVE` for predictions greater-than-or-equal-to `0.5`, 
        #       and `NEGATIVE` otherwise.
        prediction = layer2[0]
        print("Result: ",prediction, "POSITIVE" if prediction >= 0.5 else "NEGATIVE")
        return  "POSITIVE" if prediction >= 0.5 else "NEGATIVE"
        

```

Run the following cell to create a `SentimentNetwork` that will train on all but the last 1000 reviews (we're saving those for testing). Here we use a learning rate of `0.1`.


```python
mlp = SentimentNetwork(reviews[:-1000],labels[:-1000], learning_rate=0.1)
```

    SIZES:  72810 2


Run the following cell to test the network's performance against the last 1000 reviews (the ones we held out from our training set). 

**We have not trained the model yet, so the results should be about 50% as it will just be guessing and there are only two possible values to choose from.**


```python
mlp.test(reviews[-1000:],labels[-1000:])
```

    Result:  [0.5] POSITIVE
    Progress:0.0% Speed(reviews/sec):0.0 #Correct:1 #Tested:1 Testing Accuracy:100.%Result:  [0.5] POSITIVE
    Progress:0.1% Speed(reviews/sec):166.8 #Correct:1 #Tested:2 Testing Accuracy:50.0%Result:  [0.5] POSITIVE
    Progress:0.2% Speed(reviews/sec):283.0 #Correct:2 #Tested:3 Testing Accuracy:66.6%Result:  [0.5] POSITIVE
    Progress:0.3% Speed(reviews/sec):370.8 #Correct:2 #Tested:4 Testing Accuracy:50.0%Result:  [0.5] POSITIVE
    Progress:0.4% Speed(reviews/sec):436.6 #Correct:3 #Tested:5 Testing Accuracy:60.0%Result:  [0.5] POSITIVE
    Progress:0.5% Speed(reviews/sec):489.0 #Correct:3 #Tested:6 Testing Accuracy:50.0%Result:  [0.5] POSITIVE
    Progress:0.6% Speed(reviews/sec):522.3 #Correct:4 #Tested:7 Testing Accuracy:57.1%Result:  [0.5] POSITIVE
    Progress:0.7% Speed(reviews/sec):557.9 #Correct:4 #Tested:8 Testing Accuracy:50.0%Result:  [0.5] POSITIVE
    Progress:0.8% Speed(reviews/sec):584.3 #Correct:5 #Tested:9 Testing Accuracy:55.5%Result:  [0.5] POSITIVE
    Progress:0.9% Speed(reviews/sec):588.5 #Correct:5 #Tested:10 Testing Accuracy:50.0%Result:  [0.5] POSITIVE
    Progress:1.0% Speed(reviews/sec):607.7 #Correct:6 #Tested:11 Testing Accuracy:54.5%Result:  [0.5] POSITIVE
    Progress:1.1% Speed(reviews/sec):630.1 #Correct:6 #Tested:12 Testing Accuracy:50.0%Result:  [0.5] POSITIVE
    Progress:1.2% Speed(reviews/sec):647.1 #Correct:7 #Tested:13 Testing Accuracy:53.8%Result:  [0.5] POSITIVE
    Progress:1.3% Speed(reviews/sec):662.9 #Correct:7 #Tested:14 Testing Accuracy:50.0%Result:  [0.5] POSITIVE
    Progress:1.4% Speed(reviews/sec):677.0 #Correct:8 #Tested:15 Testing Accuracy:53.3%Result:  [0.5] POSITIVE
    Progress:1.5% Speed(reviews/sec):688.8 #Correct:8 #Tested:16 Testing Accuracy:50.0%Result:  [0.5] POSITIVE
    Progress:1.6% Speed(reviews/sec):689.5 #Correct:9 #Tested:17 Testing Accuracy:52.9%Result:  [0.5] POSITIVE
    ...
    Progress:99.2% Speed(reviews/sec):871.7 #Correct:497 #Tested:993 Testing Accuracy:50.0%Result:  [0.5] POSITIVE
    Progress:99.3% Speed(reviews/sec):871.9 #Correct:497 #Tested:994 Testing Accuracy:50.0%Result:  [0.5] POSITIVE
    Progress:99.4% Speed(reviews/sec):871.8 #Correct:498 #Tested:995 Testing Accuracy:50.0%Result:  [0.5] POSITIVE
    Progress:99.5% Speed(reviews/sec):871.9 #Correct:498 #Tested:996 Testing Accuracy:50.0%Result:  [0.5] POSITIVE
    Progress:99.6% Speed(reviews/sec):872.0 #Correct:499 #Tested:997 Testing Accuracy:50.0%Result:  [0.5] POSITIVE
    Progress:99.7% Speed(reviews/sec):872.1 #Correct:499 #Tested:998 Testing Accuracy:50.0%Result:  [0.5] POSITIVE
    Progress:99.8% Speed(reviews/sec):872.2 #Correct:500 #Tested:999 Testing Accuracy:50.0%Result:  [0.5] POSITIVE
    Progress:99.9% Speed(reviews/sec):872.3 #Correct:500 #Tested:1000 Testing Accuracy:50.0%

Run the following cell to actually train the network. During training, it will display the model's accuracy repeatedly as it trains so you can see how well it's doing.


```python
mlp.train(reviews[:-1000],labels[:-1000])
```

    Progress:0.0% Speed(reviews/sec):0.0 #Correct:1 #Trained:1 Training Accuracy:100.%
    Progress:10.4% Speed(reviews/sec):219.8 #Correct:1251 #Trained:2501 Training Accuracy:50.0%
    Progress:20.8% Speed(reviews/sec):185.7 #Correct:2501 #Trained:5001 Training Accuracy:50.0%
    Progress:31.2% Speed(reviews/sec):169.3 #Correct:3751 #Trained:7501 Training Accuracy:50.0%
    Progress:41.6% Speed(reviews/sec):133.5 #Correct:5001 #Trained:10001 Training Accuracy:50.0%
    Progress:52.0% Speed(reviews/sec):128.0 #Correct:6251 #Trained:12501 Training Accuracy:50.0%
    Progress:93.7% Speed(reviews/sec):117.6 #Correct:11251 #Trained:22501 Training Accuracy:50.0%
    Progress:99.9% Speed(reviews/sec):119.4 #Correct:12000 #Trained:24000 Training Accuracy:50.0%

That most likely didn't train very well. Part of the reason may be because the learning rate is too high. Run the following cell to recreate the network with a smaller learning rate, `0.01`, and then train the new network.


```python
mlp = SentimentNetwork(reviews[:-1000],labels[:-1000], learning_rate=0.01)
mlp.train(reviews[:-1000],labels[:-1000])
```

    SIZES:  72810 2
    Progress:0.0% Speed(reviews/sec):0.0 #Correct:1 #Trained:1 Training Accuracy:100.%
    Progress:10.4% Speed(reviews/sec):173.2 #Correct:1248 #Trained:2501 Training Accuracy:49.9%
    Progress:20.8% Speed(reviews/sec):170.2 #Correct:2498 #Trained:5001 Training Accuracy:49.9%
    Progress:31.2% Speed(reviews/sec):183.4 #Correct:3748 #Trained:7501 Training Accuracy:49.9%
    Progress:41.6% Speed(reviews/sec):179.6 #Correct:4998 #Trained:10001 Training Accuracy:49.9%
    Progress:52.0% Speed(reviews/sec):155.2 #Correct:6248 #Trained:12501 Training Accuracy:49.9%
    Progress:62.5% Speed(reviews/sec):165.8 #Correct:7495 #Trained:15001 Training Accuracy:49.9%
    Progress:65.7% Speed(reviews/sec):168.7 #Correct:7891 #Trained:15792 Training Accuracy:49.9%

    /usr/local/lib/python3.7/site-packages/ipykernel_launcher.py:127: RuntimeWarning: overflow encountered in exp


    Progress:72.9% Speed(reviews/sec):170.6 #Correct:8759 #Trained:17501 Training Accuracy:50.0%
    Progress:83.3% Speed(reviews/sec):172.4 #Correct:10019 #Trained:20001 Training Accuracy:50.0%
    Progress:93.7% Speed(reviews/sec):168.4 #Correct:11269 #Trained:22501 Training Accuracy:50.0%
    Progress:99.9% Speed(reviews/sec):169.1 #Correct:12018 #Trained:24000 Training Accuracy:50.0%

That probably wasn't much different. Run the following cell to recreate the network one more time with an even smaller learning rate, `0.001`, and then train the new network.


```python
mlp = SentimentNetwork(reviews[:-1000],labels[:-1000], learning_rate=0.001)
mlp.train(reviews[:-1000],labels[:-1000])
```

    SIZES:  72810 2
    Progress:0.0% Speed(reviews/sec):0.0 #Correct:1 #Trained:1 Training Accuracy:100.%
    Progress:10.4% Speed(reviews/sec):85.97 #Correct:1254 #Trained:2501 Training Accuracy:50.1%
    Progress:20.8% Speed(reviews/sec):108.5 #Correct:2602 #Trained:5001 Training Accuracy:52.0%
    Progress:31.2% Speed(reviews/sec):118.4 #Correct:4054 #Trained:7501 Training Accuracy:54.0%
    Progress:41.6% Speed(reviews/sec):126.5 #Correct:5601 #Trained:10001 Training Accuracy:56.0%
    Progress:52.0% Speed(reviews/sec):130.9 #Correct:7135 #Trained:12501 Training Accuracy:57.0%
    Progress:62.5% Speed(reviews/sec):134.6 #Correct:8753 #Trained:15001 Training Accuracy:58.3%
    Progress:72.9% Speed(reviews/sec):137.1 #Correct:10357 #Trained:17501 Training Accuracy:59.1%
    Progress:83.3% Speed(reviews/sec):139.4 #Correct:12026 #Trained:20001 Training Accuracy:60.1%
    Progress:93.7% Speed(reviews/sec):141.2 #Correct:13708 #Trained:22501 Training Accuracy:60.9%
    Progress:99.9% Speed(reviews/sec):142.0 #Correct:14612 #Trained:24000 Training Accuracy:60.8%

With a learning rate of `0.001`, the network should finally have started to improve during training. It's still not very good, but it shows that this solution has potential. We will improve it in the next lesson.

# End of Project 3. 
## Watch the next video to see Andrew's solution, then continue on to the next lesson.

# Understanding Neural Noise<a id='lesson_4'></a>

The following cells include includes the code Andrew shows in the next video. We've included it here so you can run the cells along with the video without having to type in everything.


```python
from IPython.display import Image
Image(filename='sentiment_network.png')
```




![png](images/output_76_0.png)




```python
def update_input_layer(review):
    
    global layer_0
    
    # clear out previous state, reset the layer to be all 0s
    layer_0 *= 0
    for word in review.split(" "):
        layer_0[0][word2index[word]] += 1

update_input_layer(reviews[0])
```


```python
layer_0
```




    array([[18.,  0.,  0., ...,  0.,  0.,  0.]])




```python
# review_counter = Counter()
```


```python
#for word in reviews[0].split(" "):
#    review_counter[word] += 1
review_counter = Counter(reviews[0].split(" "))
```


```python
review_counter.most_common()
```




    [('.', 27),
     ('', 18),
     ('the', 9),
     ('to', 6),
     ('high', 5),
     ('i', 5),
     ('bromwell', 4),
     ('is', 4),
     ('a', 4),
     ('teachers', 4),
     ('that', 4),
     ('of', 4),
     ('it', 2),
     ('at', 2),
     ('as', 2),
     ('school', 2),
     ('my', 2),
     ('in', 2),
     ('me', 2),
     ('students', 2),
     ('their', 2),
     ('student', 2),
     ('cartoon', 1),
     ('comedy', 1),
     ('ran', 1),
     ('same', 1),
     ('time', 1),
     ('some', 1),
     ('other', 1),
     ('programs', 1),
     ('about', 1),
     ('life', 1),
     ('such', 1),
     ('years', 1),
     ('teaching', 1),
     ('profession', 1),
     ('lead', 1),
     ('believe', 1),
     ('s', 1),
     ('satire', 1),
     ('much', 1),
     ('closer', 1),
     ('reality', 1),
     ('than', 1),
     ('scramble', 1),
     ('survive', 1),
     ('financially', 1),
     ('insightful', 1),
     ('who', 1),
     ('can', 1),
     ('see', 1),
     ('right', 1),
     ('through', 1),
     ('pathetic', 1),
     ('pomp', 1),
     ('pettiness', 1),
     ('whole', 1),
     ('situation', 1),
     ('all', 1),
     ('remind', 1),
     ('schools', 1),
     ('knew', 1),
     ('and', 1),
     ('when', 1),
     ('saw', 1),
     ('episode', 1),
     ('which', 1),
     ('repeatedly', 1),
     ('tried', 1),
     ('burn', 1),
     ('down', 1),
     ('immediately', 1),
     ('recalled', 1),
     ('classic', 1),
     ('line', 1),
     ('inspector', 1),
     ('m', 1),
     ('here', 1),
     ('sack', 1),
     ('one', 1),
     ('your', 1),
     ('welcome', 1),
     ('expect', 1),
     ('many', 1),
     ('adults', 1),
     ('age', 1),
     ('think', 1),
     ('far', 1),
     ('fetched', 1),
     ('what', 1),
     ('pity', 1),
     ('isn', 1),
     ('t', 1)]



# Project 4: Reducing Noise in Our Input Data<a id='project_4'></a>

**TODO:** Attempt to reduce the noise in the input data like Andrew did in the previous video. Specifically, do the following:
* Copy the `SentimentNetwork` class you created earlier into the following cell.
* Modify `update_input_layer` so it does not count how many times each word is used, but rather just stores whether or not a word was used. 


```python
# TODO: -Copy the SentimentNetwork class from Projet 3 lesson
#       -Modify it to reduce noise, like in the video 
import time
import sys
import numpy as np

# Encapsulate our neural network in a class
class SentimentNetwork:
    def __init__(self, reviews, labels, hidden_nodes = 10, learning_rate = 0.1):
        """Create a SentimenNetwork with the given settings
        Args:
            reviews(list) - List of reviews used for training
            labels(list) - List of POSITIVE/NEGATIVE labels associated with the given reviews
            hidden_nodes(int) - Number of nodes to create in the hidden layer
            learning_rate(float) - Learning rate to use while training
        
        """
        # Assign a seed to our random number generator to ensure we get
        # reproducable results during development 
        np.random.seed(1)

        # process the reviews and their associated labels so that everything
        # is ready for training
        self.pre_process_data(reviews, labels)
        
        # Build the network to have the number of hidden nodes and the learning rate that
        # were passed into this initializer. Make the same number of input nodes as
        # there are vocabulary words and create a single output node.
        self.init_network(len(self.review_vocab),hidden_nodes, 1, learning_rate)

    def pre_process_data(self, reviews, labels):
        
        review_vocab = set()
        # TODO: populate review_vocab with all of the words in the given reviews
        #       Remember to split reviews into individual words 
        #       using "split(' ')" instead of "split()".
        review_vocab_counter = Counter()
        for i, review in enumerate(reviews) :
            review_vocab_counter.update(review.lower().split(" "))
        review_vocab = review_vocab_counter.keys()
        
        # Convert the vocabulary set to a list so we can access words via indices
        self.review_vocab = list(review_vocab)
        
        # TODO: populate label_vocab with all of the words in the given labels.
        #       There is no need to split the labels because each one is a single word.
        # Convert the label vocabulary set to a list so we can access labels via indices
        label_vocab = set(labels)
        
        self.label_vocab = list(label_vocab)
        
        # Store the sizes of the review and label vocabularies.
        self.review_vocab_size = len(self.review_vocab)
        self.label_vocab_size = len(self.label_vocab)
        print("SIZES: ",self.review_vocab_size,         self.label_vocab_size )
        
        # Create a dictionary of words in the vocabulary mapped to index positions
        self.word2index = {}
        # TODO: populate self.word2index with indices for all the words in self.review_vocab
        #       like you saw earlier in the notebook
        for i, word in enumerate(self.review_vocab):
            self.word2index[word] = i
        # Create a dictionary of labels mapped to index positions
        self.label2index = {}
        # TODO: do the same thing you did for self.word2index and self.review_vocab, 
        #       but for self.label2index and self.label_vocab instead
        for i, wordLabel in enumerate(self.label_vocab):
            self.label2index[wordLabel] = i
        
    def init_network(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Store the number of nodes in input, hidden, and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Store the learning rate
        self.learning_rate = learning_rate

        # Initialize weights
        
        # TODO: initialize self.weights_0_1 as a matrix of zeros. These are the weights between
        #       the input layer and the hidden layer.
        # DOUBT: SHAPE?
        self.weights_0_1 = np.zeros((self.input_nodes,self.hidden_nodes))
        
        # TODO: initialize self.weights_1_2 as a matrix of random values. 
        #       These are the weights between the hidden layer and the output layer.
        # DOUBT: SHAPE?
        self.weights_1_2 = np.random.normal(0.0, self.output_nodes**-0.5, 
                                                (self.hidden_nodes, self.output_nodes))
        
        # TODO: Create the input layer, a two-dimensional matrix with shape 
        #       1 x input_nodes, with all values initialized to zero
        self.layer_0 = np.zeros((1,input_nodes))
    
    def silly_words(self):
        return ['.', '', 'the', 'to', 'i', 'is', 'a', 'that', 'of', 'it', 'at', 'as', 'my', 'in', 'me']
    def update_input_layer(self,review):
        # DONE: You can copy most of the code you wrote for update_input_layer 
        #       earlier in this notebook. 
        #
        #       However, MAKE SURE YOU CHANGE ALL VARIABLES TO REFERENCE
        #       THE VERSIONS STORED IN THIS OBJECT, NOT THE GLOBAL OBJECTS.
        #       For example, replace "layer_0 *= 0" with "self.layer_0 *= 0"
            # clear out previous state by resetting the layer to be all 0s
        self.layer_0 *= 0
    
        # count how many times each word is used in the given review and store the results in layer_0 
        counted_words = Counter(review.lower().split(' '))
        for word, count in counted_words.items() :
            if(word in self.word2index.keys() and word not in self.silly_words()):
                word_index_in_layer = self.word2index[word]
                # We remove the counter, and we set it binary.
                # Instead to search how many repetition of each word, 
                # we center our search if the review contains that word.
                self.layer_0[0][word_index_in_layer] = 1
                
    def get_target_for_label(self,label):
        # DONE: Copy the code you wrote for get_target_for_label 
        #       earlier in this notebook. 
        """Convert a label to `0` or `1`.
        Args:
            label(string) - Either "POSITIVE" or "NEGATIVE".
        Returns:
            `0` or `1`.
        """
        return 0 if "NEGATIVE" == label else 1
        
    def sigmoid(self,x):
        # DONE: Return the result of calculating the sigmoid activation function
        #       shown in the lectures
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_output_2_derivative(self,output):
        # DONE: Return the derivative of the sigmoid activation function, 
        #       where "output" is the original output from the sigmoid fucntion 
        # dy/dx = f(x)' = f(x) * (1 - f(x))
        return output * (1 - output)

    def train(self, training_reviews, training_labels):
        
        # make sure out we have a matching number of reviews and labels
        assert(len(training_reviews) == len(training_labels))
        
        # Keep track of correct predictions to display accuracy during training 
        correct_so_far = 0
        
        # Remember when we started for printing time statistics
        start = time.time()

        # loop through all the given reviews and run a forward and backward pass,
        # updating weights for every item
        for i in range(len(training_reviews)):
            
            # DONE: Get the next review and its correct label
            review = training_reviews[i]
            label = training_labels[i]
            
            # DONE: Implement the forward pass through the network. 
            #       That means use the given review to update the input layer, 
            #       then calculate values for the hidden layer,
            #       and finally calculate the output layer.
            # 
            #       Do not use an activation function for the hidden layer,
            #       but use the sigmoid activation function for the output layer.
            
            # Update Input layer
            self.update_input_layer(review)
            # Hidden layer
            layer_1 = self.layer_0.dot(self.weights_0_1)
            # Output layer
            layer_2 = self.sigmoid(layer_1.dot(self.weights_1_2))
            
            
            # DONE: Implement the back propagation pass here. 
            #       That means calculate the error for the forward pass's prediction
            #       and update the weights in the network according to their
            #       contributions toward the error, as calculated via the
            #       gradient descent and back propagation algorithms you 
            #       learned in class.

            ### Backward pass ###
            # Output error
            layer_2_error = layer_2 - self.get_target_for_label(label) # Output layer error is the difference between desired target and actual output.
            layer_2_delta = layer_2_error * self.sigmoid_output_2_derivative(layer_2)
            # Backpropagated error
            layer_1_error = layer_2_delta.dot(self.weights_1_2.T) # errors propagated to the hidden layer
            layer_1_delta = layer_1_error # hidden layer gradients - no nonlinearity so it's the same as the error
            # Update the weights
            self.weights_1_2 -= layer_1.T.dot(layer_2_delta) * self.learning_rate # update hidden-to-output weights with gradient descent step
            self.weights_0_1 -= self.layer_0.T.dot(layer_1_delta) * self.learning_rate # update input-to-hidden weights with gradient descent step

            
            # DONE: Keep track of correct predictions. To determine if the prediction was
            #       correct, check that the absolute value of the output error 
            #       is less than 0.5. If so, add one to the correct_so_far count.
             # Keep track of correct predictions.
            if(layer_2 >= 0.5 and label == 'POSITIVE'):
                correct_so_far += 1
            elif(layer_2 < 0.5 and label == 'NEGATIVE'):
                correct_so_far += 1
            
            # For debug purposes, print out our prediction accuracy and speed 
            # throughout the training process. 

            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0
            
            sys.stdout.write("\rProgress:" + str(100 * i/float(len(training_reviews)))[:4] \
                             + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                             + " #Correct:" + str(correct_so_far) + " #Trained:" + str(i+1) \
                             + " Training Accuracy:" + str(correct_so_far * 100 / float(i+1))[:4] + "%")
            if(i % 2500 == 0):
                print("")
    
    def test(self, testing_reviews, testing_labels):
        """
        Attempts to predict the labels for the given testing_reviews,
        and uses the test_labels to calculate the accuracy of those predictions.
        """
        
        # keep track of how many correct predictions we make
        correct = 0

        # we'll time how many predictions per second we make
        start = time.time()

        # Loop through each of the given reviews and call run to predict
        # its label. 
        for i in range(len(testing_reviews)):
            pred = self.run(testing_reviews[i])
            if(pred == testing_labels[i]):
                correct += 1
            
            # For debug purposes, print out our prediction accuracy and speed 
            # throughout the prediction process. 

            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0
            
            sys.stdout.write("\rProgress:" + str(100 * i/float(len(testing_reviews)))[:4] \
                             + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                             + " #Correct:" + str(correct) + " #Tested:" + str(i+1) \
                             + " Testing Accuracy:" + str(correct * 100 / float(i+1))[:4] + "%")
    
    def run(self, review):
        """
        Returns a POSITIVE or NEGATIVE prediction for the given review.
        """
        # DONE: Run a forward pass through the network, like you did in the
        #       "train" function. That means use the given review to 
        #       update the input layer, then calculate values for the hidden layer,
        #       and finally calculate the output layer.
        #
        #       Note: The review passed into this function for prediction 
        #             might come from anywhere, so you should convert it 
        #             to lower case prior to using it.
        self.update_input_layer(review.lower())
        layer1 = self.layer_0.dot(self.weights_0_1)
        layer2 = self.sigmoid(layer1.dot(self.weights_1_2))
        
        # DONE: The output layer should now contain a prediction. 
        #       Return `POSITIVE` for predictions greater-than-or-equal-to `0.5`, 
        #       and `NEGATIVE` otherwise.
        prediction = layer2[0]
        print("Result: ",prediction, "POSITIVE" if prediction >= 0.5 else "NEGATIVE")
        return  "POSITIVE" if prediction >= 0.5 else "NEGATIVE"
        
```

Run the following cell to recreate the network and train it. Notice we've gone back to the higher learning rate of `0.1`.


```python
mlp = SentimentNetwork(reviews[:-1000],labels[:-1000], learning_rate=0.1)
mlp.train(reviews[:-1000],labels[:-1000])
```

    SIZES:  72810 2
    Progress:0.0% Speed(reviews/sec):0.0 #Correct:1 #Trained:1 Training Accuracy:100.%
    Progress:10.4% Speed(reviews/sec):146.5 #Correct:1881 #Trained:2501 Training Accuracy:75.2%
    Progress:20.8% Speed(reviews/sec):147.4 #Correct:3885 #Trained:5001 Training Accuracy:77.6%
    Progress:31.2% Speed(reviews/sec):148.1 #Correct:5990 #Trained:7501 Training Accuracy:79.8%
    Progress:41.6% Speed(reviews/sec):147.9 #Correct:8127 #Trained:10001 Training Accuracy:81.2%
    Progress:52.0% Speed(reviews/sec):147.8 #Correct:10260 #Trained:12501 Training Accuracy:82.0%
    Progress:62.5% Speed(reviews/sec):148.1 #Correct:12399 #Trained:15001 Training Accuracy:82.6%
    Progress:72.9% Speed(reviews/sec):140.7 #Correct:14533 #Trained:17501 Training Accuracy:83.0%
    Progress:83.3% Speed(reviews/sec):139.5 #Correct:16714 #Trained:20001 Training Accuracy:83.5%
    Progress:93.7% Speed(reviews/sec):145.0 #Correct:18894 #Trained:22501 Training Accuracy:83.9%
    Progress:99.9% Speed(reviews/sec):146.3 #Correct:20223 #Trained:24000 Training Accuracy:84.2%

That should have trained much better than the earlier attempts. It's still not wonderful, but it should have improved dramatically. Run the following cell to test your model with 1000 predictions.


```python
mlp.test(reviews[-1000:],labels[-1000:])
```

    Result:  [0.8810176] POSITIVE
    Progress:0.0% Speed(reviews/sec):0.0 #Correct:1 #Tested:1 Testing Accuracy:100.%Result:  [0.54456883] POSITIVE
    Progress:0.1% Speed(reviews/sec):313.6 #Correct:1 #Tested:2 Testing Accuracy:50.0%Result:  [0.68924115] POSITIVE
    Progress:0.2% Speed(reviews/sec):470.4 #Correct:2 #Tested:3 Testing Accuracy:66.6%Result:  [0.23220539] NEGATIVE
    Progress:0.3% Speed(reviews/sec):539.9 #Correct:3 #Tested:4 Testing Accuracy:75.0%Result:  [0.83427606] POSITIVE
    Progress:0.4% Speed(reviews/sec):621.9 #Correct:4 #Tested:5 Testing Accuracy:80.0%Result:  [0.00135583] NEGATIVE
    Progress:0.5% Speed(reviews/sec):652.4 #Correct:5 #Tested:6 Testing Accuracy:83.3%Result:  [0.92476774] POSITIVE
    Progress:0.6% Speed(reviews/sec):680.8 #Correct:6 #Tested:7 Testing Accuracy:85.7%Result:  [0.00201876] NEGATIVE
    Progress:0.7% Speed(reviews/sec):713.7 #Correct:7 #Tested:8 Testing Accuracy:87.5%Result:  [0.93337182] POSITIVE
    Progress:0.8% Speed(reviews/sec):717.9 #Correct:8 #Tested:9 Testing Accuracy:88.8%Result:  [0.00038804] NEGATIVE
    Progress:0.9% Speed(reviews/sec):681.6 #Correct:9 #Tested:10 Testing Accuracy:90.0%Result:  [0.98666568] POSITIVE
    Progress:1.0% Speed(reviews/sec):688.6 #Correct:10 #Tested:11 Testing Accuracy:90.9%Result:  [0.01333673] NEGATIVE
    Progress:1.1% Speed(reviews/sec):698.0 #Correct:11 #Tested:12 Testing Accuracy:91.6%Result:  [0.6069431] POSITIVE
    Progress:1.2% Speed(reviews/sec):706.0 #Correct:12 #Tested:13 Testing Accuracy:92.3%Result:  [0.00010382] NEGATIVE
    Progress:1.3% Speed(reviews/sec):721.3 #Correct:13 #Tested:14 Testing Accuracy:92.8%Result:  [0.9909137] POSITIVE
    Progress:1.4% Speed(reviews/sec):731.6 #Correct:14 #Tested:15 Testing Accuracy:93.3%Result:  [0.0003928] NEGATIVE
    Progress:1.5% Speed(reviews/sec):739.4 #Correct:15 #Tested:16 Testing Accuracy:93.7%Result:  [0.99843609] POSITIVE
    Progress:1.6% Speed(reviews/sec):734.0 #Correct:16 #Tested:17 Testing Accuracy:94.1%Result:  [0.00266565] NEGATIVE
    Progress:1.7% Speed(reviews/sec):736.2 #Correct:17 #Tested:18 Testing Accuracy:94.4%Result:  [0.99024983] POSITIVE
    Progress:1.8% Speed(reviews/sec):747.5 #Correct:18 #Tested:19 Testing Accuracy:94.7%Result:  [0.73994744] POSITIVE
    Progress:1.9% Speed(reviews/sec):756.5 #Correct:18 #Tested:20 Testing Accuracy:90.0%Result:  [0.99937013] POSITIVE
    Progress:2.0% Speed(reviews/sec):756.2 #Correct:19 #Tested:21 Testing Accuracy:90.4%Result:  [0.01273337] NEGATIVE
    Progress:2.1% Speed(reviews/sec):760.6 #Correct:20 #Tested:22 Testing Accuracy:90.9%Result:  [0.99894043] POSITIVE
    Progress:2.2% Speed(reviews/sec):759.7 #Correct:21 #Tested:23 Testing Accuracy:91.3%Result:  [0.00062674] NEGATIVE
    Progress:2.3% Speed(reviews/sec):758.2 #Correct:22 #Tested:24 Testing Accuracy:91.6%Result:  [0.99998029] POSITIVE
    Progress:2.4% Speed(reviews/sec):745.3 #Correct:23 #Tested:25 Testing Accuracy:92.0%Result:  [0.01738631] NEGATIVE
    Progress:2.5% Speed(reviews/sec):751.2 #Correct:24 #Tested:26 Testing Accuracy:92.3%Result:  [0.83919037] POSITIVE
    Progress:2.6% Speed(reviews/sec):754.1 #Correct:25 #Tested:27 Testing Accuracy:92.5%Result:  [3.8636324e-05] NEGATIVE
    Progress:2.7% Speed(reviews/sec):742.3 #Correct:26 #Tested:28 Testing Accuracy:92.8%Result:  [0.96014953] POSITIVE
    Progress:2.8% Speed(reviews/sec):742.5 #Correct:27 #Tested:29 Testing Accuracy:93.1%Result:  [8.47810586e-05] NEGATIVE
    ...
    Progress:97.3% Speed(reviews/sec):765.5 #Correct:833 #Tested:974 Testing Accuracy:85.5%Result:  [0.9983918] POSITIVE
    Progress:97.4% Speed(reviews/sec):765.6 #Correct:834 #Tested:975 Testing Accuracy:85.5%Result:  [0.01202945] NEGATIVE
    Progress:97.5% Speed(reviews/sec):765.8 #Correct:835 #Tested:976 Testing Accuracy:85.5%Result:  [0.99954931] POSITIVE
    Progress:97.6% Speed(reviews/sec):766.0 #Correct:836 #Tested:977 Testing Accuracy:85.5%Result:  [0.595302] POSITIVE
    Progress:97.7% Speed(reviews/sec):766.0 #Correct:836 #Tested:978 Testing Accuracy:85.4%Result:  [0.94190109] POSITIVE
    Progress:97.8% Speed(reviews/sec):766.3 #Correct:837 #Tested:979 Testing Accuracy:85.4%Result:  [0.00384568] NEGATIVE
    Progress:97.9% Speed(reviews/sec):766.4 #Correct:838 #Tested:980 Testing Accuracy:85.5%Result:  [0.73346816] POSITIVE
    Progress:98.0% Speed(reviews/sec):766.7 #Correct:839 #Tested:981 Testing Accuracy:85.5%Result:  [1.19005113e-05] NEGATIVE
    Progress:98.1% Speed(reviews/sec):766.6 #Correct:840 #Tested:982 Testing Accuracy:85.5%Result:  [0.92012401] POSITIVE
    Progress:98.2% Speed(reviews/sec):766.7 #Correct:841 #Tested:983 Testing Accuracy:85.5%Result:  [0.02391535] NEGATIVE
    Progress:98.3% Speed(reviews/sec):766.6 #Correct:842 #Tested:984 Testing Accuracy:85.5%Result:  [0.86987516] POSITIVE
    Progress:98.4% Speed(reviews/sec):766.9 #Correct:843 #Tested:985 Testing Accuracy:85.5%Result:  [0.02822387] NEGATIVE
    Progress:98.5% Speed(reviews/sec):767.1 #Correct:844 #Tested:986 Testing Accuracy:85.5%Result:  [0.81647705] POSITIVE
    Progress:98.6% Speed(reviews/sec):767.4 #Correct:845 #Tested:987 Testing Accuracy:85.6%Result:  [0.04405014] NEGATIVE
    Progress:98.7% Speed(reviews/sec):767.4 #Correct:846 #Tested:988 Testing Accuracy:85.6%Result:  [0.99962487] POSITIVE
    Progress:98.8% Speed(reviews/sec):767.6 #Correct:847 #Tested:989 Testing Accuracy:85.6%Result:  [0.00244277] NEGATIVE
    Progress:98.9% Speed(reviews/sec):767.7 #Correct:848 #Tested:990 Testing Accuracy:85.6%Result:  [0.99620324] POSITIVE
    Progress:99.0% Speed(reviews/sec):767.9 #Correct:849 #Tested:991 Testing Accuracy:85.6%Result:  [0.62985566] POSITIVE
    Progress:99.1% Speed(reviews/sec):768.0 #Correct:849 #Tested:992 Testing Accuracy:85.5%Result:  [0.84131817] POSITIVE
    Progress:99.2% Speed(reviews/sec):768.2 #Correct:850 #Tested:993 Testing Accuracy:85.5%Result:  [0.02480022] NEGATIVE
    Progress:99.3% Speed(reviews/sec):768.3 #Correct:851 #Tested:994 Testing Accuracy:85.6%Result:  [0.74265668] POSITIVE
    Progress:99.4% Speed(reviews/sec):768.1 #Correct:852 #Tested:995 Testing Accuracy:85.6%Result:  [0.00289758] NEGATIVE
    Progress:99.5% Speed(reviews/sec):768.1 #Correct:853 #Tested:996 Testing Accuracy:85.6%Result:  [0.99463354] POSITIVE
    Progress:99.6% Speed(reviews/sec):768.3 #Correct:854 #Tested:997 Testing Accuracy:85.6%Result:  [0.02361965] NEGATIVE
    Progress:99.7% Speed(reviews/sec):768.4 #Correct:855 #Tested:998 Testing Accuracy:85.6%Result:  [0.27956032] NEGATIVE
    Progress:99.8% Speed(reviews/sec):768.5 #Correct:855 #Tested:999 Testing Accuracy:85.5%Result:  [0.00074097] NEGATIVE
    Progress:99.9% Speed(reviews/sec):768.6 #Correct:856 #Tested:1000 Testing Accuracy:85.6%

# End of Project 4. 
## Andrew's solution was actually in the previous video, so rewatch that video if you had any problems with that project. Then continue on to the next lesson.
# Analyzing Inefficiencies in our Network<a id='lesson_5'></a>
The following cells include the code Andrew shows in the next video. We've included it here so you can run the cells along with the video without having to type in everything.


```python
Image(filename='sentiment_network_sparse.png')
```




![png](images/output_89_0.png)




```python
layer_0 = np.zeros(10)
```


```python
layer_0
```




    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])




```python
layer_0[4] = 1
layer_0[9] = 1
```


```python
layer_0
```




    array([0., 0., 0., 0., 1., 0., 0., 0., 0., 1.])




```python
weights_0_1 = np.random.randn(10,5)
weights_0_1
```




    array([[ 1.46210794, -2.06014071, -0.3224172 , -0.38405435,  1.13376944],
           [-1.09989127, -0.17242821, -0.87785842,  0.04221375,  0.58281521],
           [-1.10061918,  1.14472371,  0.90159072,  0.50249434,  0.90085595],
           [-0.68372786, -0.12289023, -0.93576943, -0.26788808,  0.53035547],
           [-0.69166075, -0.39675353, -0.6871727 , -0.84520564, -0.67124613],
           [-0.0126646 , -1.11731035,  0.2344157 ,  1.65980218,  0.74204416],
           [-0.19183555, -0.88762896, -0.74715829,  1.6924546 ,  0.05080775],
           [-0.63699565,  0.19091548,  2.10025514,  0.12015895,  0.61720311],
           [ 0.30017032, -0.35224985, -1.1425182 , -0.34934272, -0.20889423],
           [ 0.58662319,  0.83898341,  0.93110208,  0.28558733,  0.88514116]])




```python
layer_0.dot(weights_0_1)
```




    array([-0.10503756,  0.44222989,  0.24392938, -0.55961832,  0.21389503])




```python
indices = [4,9]
```


```python
layer_1 = np.zeros(5)
```


```python
for index in indices:
    layer_1 += (1 * weights_0_1[index])
```


```python
layer_1
```




    array([-0.10503756,  0.44222989,  0.24392938, -0.55961832,  0.21389503])




```python
Image(filename='sentiment_network_sparse_2.png')
```




![png](images/output_100_0.png)




```python
layer_1 = np.zeros(5)
```


```python
for index in indices:
    layer_1 += (weights_0_1[index])
```


```python
layer_1
```




    array([-0.10503756,  0.44222989,  0.24392938, -0.55961832,  0.21389503])



# Project 5: Making our Network More Efficient<a id='project_5'></a>
**TODO:** Make the `SentimentNetwork` class more efficient by eliminating unnecessary multiplications and additions that occur during forward and backward propagation. To do that, you can do the following:
* Copy the `SentimentNetwork` class from the previous project into the following cell.
* Remove the `update_input_layer` function - you will not need it in this version.
* Modify `init_network`:
>* You no longer need a separate input layer, so remove any mention of `self.layer_0`
>* You will be dealing with the old hidden layer more directly, so create `self.layer_1`, a two-dimensional matrix with shape 1 x hidden_nodes, with all values initialized to zero
* Modify `train`:
>* Change the name of the input parameter `training_reviews` to `training_reviews_raw`. This will help with the next step.
>* At the beginning of the function, you'll want to preprocess your reviews to convert them to a list of indices (from `word2index`) that are actually used in the review. This is equivalent to what you saw in the video when Andrew set specific indices to 1. Your code should create a local `list` variable named `training_reviews` that should contain a `list` for each review in `training_reviews_raw`. Those lists should contain the indices for words found in the review.
>* Remove call to `update_input_layer`
>* Use `self`'s  `layer_1` instead of a local `layer_1` object.
>* In the forward pass, replace the code that updates `layer_1` with new logic that only adds the weights for the indices used in the review.
>* When updating `weights_0_1`, only update the individual weights that were used in the forward pass.
* Modify `run`:
>* Remove call to `update_input_layer` 
>* Use `self`'s  `layer_1` instead of a local `layer_1` object.
>* Much like you did in `train`, you will need to pre-process the `review` so you can work with word indices, then update `layer_1` by adding weights for the indices used in the review.


```python
# DONE: -Copy the SentimentNetwork class from Project 4 lesson
#       -Modify it according to the above instructions 

import time
import sys
import numpy as np

# Encapsulate our neural network in a class
class SentimentNetwork:
    def __init__(self, reviews, labels, hidden_nodes = 10, learning_rate = 0.1):
        """Create a SentimenNetwork with the given settings
        Args:
            reviews(list) - List of reviews used for training
            labels(list) - List of POSITIVE/NEGATIVE labels associated with the given reviews
            hidden_nodes(int) - Number of nodes to create in the hidden layer
            learning_rate(float) - Learning rate to use while training
        
        """
        # Assign a seed to our random number generator to ensure we get
        # reproducable results during development 
        np.random.seed(1)

        # process the reviews and their associated labels so that everything
        # is ready for training
        self.pre_process_data(reviews, labels)
        
        # Build the network to have the number of hidden nodes and the learning rate that
        # were passed into this initializer. Make the same number of input nodes as
        # there are vocabulary words and create a single output node.
        self.init_network(len(self.review_vocab),hidden_nodes, 1, learning_rate)

    def pre_process_data(self, reviews, labels):
        
        review_vocab = set()
        # TODO: populate review_vocab with all of the words in the given reviews
        #       Remember to split reviews into individual words 
        #       using "split(' ')" instead of "split()".
        review_vocab_counter = Counter()
        for i, review in enumerate(reviews) :
            review_vocab_counter.update(review.lower().split(" "))
        review_vocab = review_vocab_counter.keys()
        
        # Convert the vocabulary set to a list so we can access words via indices
        self.review_vocab = list(review_vocab)
        
        # TODO: populate label_vocab with all of the words in the given labels.
        #       There is no need to split the labels because each one is a single word.
        # Convert the label vocabulary set to a list so we can access labels via indices
        label_vocab = set(labels)
        
        self.label_vocab = list(label_vocab)
        
        # Store the sizes of the review and label vocabularies.
        self.review_vocab_size = len(self.review_vocab)
        self.label_vocab_size = len(self.label_vocab)
        print("SIZES: ",self.review_vocab_size,         self.label_vocab_size )
        
        # Create a dictionary of words in the vocabulary mapped to index positions
        self.word2index = {}
        # TODO: populate self.word2index with indices for all the words in self.review_vocab
        #       like you saw earlier in the notebook
        for i, word in enumerate(self.review_vocab):
            self.word2index[word] = i
        # Create a dictionary of labels mapped to index positions
        self.label2index = {}
        # TODO: do the same thing you did for self.word2index and self.review_vocab, 
        #       but for self.label2index and self.label_vocab instead
        for i, wordLabel in enumerate(self.label_vocab):
            self.label2index[wordLabel] = i
        
    def init_network(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Store the number of nodes in input, hidden, and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Store the learning rate
        self.learning_rate = learning_rate

        # Initialize weights
        
        # TODO: initialize self.weights_0_1 as a matrix of zeros. These are the weights between
        #       the input layer and the hidden layer.
        # DOUBT: SHAPE?
        self.weights_0_1 = np.zeros((self.input_nodes, self.hidden_nodes))
        
        # TODO: initialize self.weights_1_2 as a matrix of random values. 
        #       These are the weights between the hidden layer and the output layer.
        # DOUBT: SHAPE?
        self.weights_1_2 = np.random.normal(0.0, self.output_nodes**-0.5, 
                                                (self.hidden_nodes, self.output_nodes))
        
        #       1 x hidden_nodes, with all values initialized to zero
        self.layer_1 = np.zeros((1,hidden_nodes))
    
    def silly_words(self):
        return ['.', '', 'the', 'to', 'i', 'is', 'a', 'that', 'of', 'it', 'at', 'as', 'my', 'in', 'me']
                
    def get_target_for_label(self,label):
        # DONE: Copy the code you wrote for get_target_for_label 
        #       earlier in this notebook. 
        """Convert a label to `0` or `1`.
        Args:
            label(string) - Either "POSITIVE" or "NEGATIVE".
        Returns:
            `0` or `1`.
        """
        return 0 if "NEGATIVE" == label else 1
        
    def sigmoid(self,x):
        # DONE: Return the result of calculating the sigmoid activation function
        #       shown in the lectures
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_output_2_derivative(self,output):
        # DONE: Return the derivative of the sigmoid activation function, 
        #       where "output" is the original output from the sigmoid fucntion 
        # dy/dx = f(x)' = f(x) * (1 - f(x))
        return output * (1 - output)

    def train(self, training_reviews_raw, training_labels):
        
        # Create Indexes list 
        training_reviews = list()
        for review in training_reviews_raw:
            indices = set()
            for word in review.split(" "):
                if(word in self.word2index.keys()):
                    indices.add(self.word2index[word])
            training_reviews.append(list(indices))

        
        # make sure out we have a matching number of reviews and labels
        assert(len(training_reviews) == len(training_labels))
        
        # Keep track of correct predictions to display accuracy during training 
        correct_so_far = 0
        
        # Remember when we started for printing time statistics
        start = time.time()

        # loop through all the given reviews and run a forward and backward pass,
        # updating weights for every item
        for i in range(len(training_reviews)):
            
            # DONE: Get the next review and its correct label
            review = training_reviews[i]
            label = training_labels[i]
            
            # DONE: Implement the forward pass through the network. 
            #       That means use the given review to update the input layer, 
            #       then calculate values for the hidden layer,
            #       and finally calculate the output layer.
            # 
            #       Do not use an activation function for the hidden layer,
            #       but use the sigmoid activation function for the output layer.
            
            # Hidden layer

            # review is Review Indexes of its words.

            self.layer_1 *= 0
            for index in review:
                self.layer_1 += self.weights_0_1[index]
            # Output layer
            layer_2 = self.sigmoid(self.layer_1.dot(self.weights_1_2))
            
            
            # DONE: Implement the back propagation pass here. 
            #       That means calculate the error for the forward pass's prediction
            #       and update the weights in the network according to their
            #       contributions toward the error, as calculated via the
            #       gradient descent and back propagation algorithms you 
            #       learned in class.

            ### Backward pass ###
            # Output error
            layer_2_error = layer_2 - self.get_target_for_label(label) # Output layer error is the difference between desired target and actual output.
            layer_2_delta = layer_2_error * self.sigmoid_output_2_derivative(layer_2)
            # Backpropagated error
            layer_1_error = layer_2_delta.dot(self.weights_1_2.T) # errors propagated to the hidden layer
            layer_1_delta = layer_1_error # hidden layer gradients - no nonlinearity so it's the same as the error
            # Update the weights
            self.weights_1_2 -= self.layer_1.T.dot(layer_2_delta) * self.learning_rate # update hidden-to-output weights with gradient descent step

            ## New for Project 5: Only update the weights that were used in the forward pass
            # DOUBT:
            for index in review:
                self.weights_0_1[index] -= layer_1_delta[0] * self.learning_rate # update input-to-hidden weights with gradient descent step

            
            # DONE: Keep track of correct predictions. To determine if the prediction was
            #       correct, check that the absolute value of the output error 
            #       is less than 0.5. If so, add one to the correct_so_far count.
             # Keep track of correct predictions.
            if(layer_2 >= 0.5 and label == 'POSITIVE'):
                correct_so_far += 1
            elif(layer_2 < 0.5 and label == 'NEGATIVE'):
                correct_so_far += 1
            
            # For debug purposes, print out our prediction accuracy and speed 
            # throughout the training process. 

            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0
            
            sys.stdout.write("\rProgress:" + str(100 * i/float(len(training_reviews)))[:4] \
                             + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                             + " #Correct:" + str(correct_so_far) + " #Trained:" + str(i+1) \
                             + " Training Accuracy:" + str(correct_so_far * 100 / float(i+1))[:4] + "%")
            if(i % 2500 == 0):
                print("")
    
    def test(self, testing_reviews, testing_labels):
        """
        Attempts to predict the labels for the given testing_reviews,
        and uses the test_labels to calculate the accuracy of those predictions.
        """
        
        # keep track of how many correct predictions we make
        correct = 0

        # we'll time how many predictions per second we make
        start = time.time()

        # Loop through each of the given reviews and call run to predict
        # its label. 
        for i in range(len(testing_reviews)):
            pred = self.run(testing_reviews[i])
            if(pred == testing_labels[i]):
                correct += 1
            
            # For debug purposes, print out our prediction accuracy and speed 
            # throughout the prediction process. 

            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0
            
            sys.stdout.write("\rProgress:" + str(100 * i/float(len(testing_reviews)))[:4] \
                             + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                             + " #Correct:" + str(correct) + " #Tested:" + str(i+1) \
                             + " Testing Accuracy:" + str(correct * 100 / float(i+1))[:4] + "%")
    
    def run(self, review):
        """
        Returns a POSITIVE or NEGATIVE prediction for the given review.
        """
        # DONE: Run a forward pass through the network, like you did in the
        #       "train" function. That means use the given review to 
        #       update the input layer, then calculate values for the hidden layer,
        #       and finally calculate the output layer.
        #
        #       Note: The review passed into this function for prediction 
        #             might come from anywhere, so you should convert it 
        #             to lower case prior to using it.
        self.layer_1 *= 0
        unique_indices = set()
        for word in review.lower().split(" "):
            if word in self.word2index.keys():
                unique_indices.add(self.word2index[word])

        for index in unique_indices:
            self.layer_1 += self.weights_0_1[index]
        
        layer2 = self.sigmoid(self.layer_1.dot(self.weights_1_2))
        
        # DONE: The output layer should now contain a prediction. 
        #       Return `POSITIVE` for predictions greater-than-or-equal-to `0.5`, 
        #       and `NEGATIVE` otherwise.
        prediction = layer2[0]
        print("Result: ",prediction, "POSITIVE" if prediction >= 0.5 else "NEGATIVE")
        return  "POSITIVE" if prediction >= 0.5 else "NEGATIVE"
```

Run the following cell to recreate the network and train it once again.


```python
mlp = SentimentNetwork(reviews[:-1000],labels[:-1000], learning_rate=0.1)
mlp.train(reviews[:-1000],labels[:-1000])
```

    SIZES:  72810 2
    Progress:0.0% Speed(reviews/sec):0.0 #Correct:1 #Trained:1 Training Accuracy:100.%
    Progress:10.4% Speed(reviews/sec):954.6 #Correct:1825 #Trained:2501 Training Accuracy:72.9%
    Progress:20.8% Speed(reviews/sec):904.7 #Correct:3798 #Trained:5001 Training Accuracy:75.9%
    Progress:31.2% Speed(reviews/sec):947.2 #Correct:5869 #Trained:7501 Training Accuracy:78.2%
    Progress:41.6% Speed(reviews/sec):1011. #Correct:8008 #Trained:10001 Training Accuracy:80.0%
    Progress:52.0% Speed(reviews/sec):1065. #Correct:10131 #Trained:12501 Training Accuracy:81.0%
    Progress:62.5% Speed(reviews/sec):1114. #Correct:12255 #Trained:15001 Training Accuracy:81.6%
    Progress:72.9% Speed(reviews/sec):1146. #Correct:14369 #Trained:17501 Training Accuracy:82.1%
    Progress:83.3% Speed(reviews/sec):1173. #Correct:16546 #Trained:20001 Training Accuracy:82.7%
    Progress:93.7% Speed(reviews/sec):1193. #Correct:18737 #Trained:22501 Training Accuracy:83.2%
    Progress:99.9% Speed(reviews/sec):1205. #Correct:20058 #Trained:24000 Training Accuracy:83.5%

That should have trained much better than the earlier attempts. Run the following cell to test your model with 1000 predictions.


```python
mlp.test(reviews[-1000:],labels[-1000:])
```

    Result:  [0.95138388] POSITIVE
    Progress:0.0% Speed(reviews/sec):0.0 #Correct:1 #Tested:1 Testing Accuracy:100.%Result:  [0.63222426] POSITIVE
    Progress:0.1% Speed(reviews/sec):779.3 #Correct:1 #Tested:2 Testing Accuracy:50.0%Result:  [0.49110523] NEGATIVE
    Progress:0.2% Speed(reviews/sec):1166. #Correct:1 #Tested:3 Testing Accuracy:33.3%Result:  [0.17223164] NEGATIVE
    Progress:0.3% Speed(reviews/sec):1337. #Correct:2 #Tested:4 Testing Accuracy:50.0%Result:  [0.94307075] POSITIVE
    Progress:0.4% Speed(reviews/sec):1493. #Correct:3 #Tested:5 Testing Accuracy:60.0%Result:  [0.00161075] NEGATIVE
    Progress:0.5% Speed(reviews/sec):1362. #Correct:4 #Tested:6 Testing Accuracy:66.6%Result:  [0.92463613] POSITIVE
    Progress:0.6% Speed(reviews/sec):1347. #Correct:5 #Tested:7 Testing Accuracy:71.4%Result:  [0.00405227] NEGATIVE
    Progress:0.7% Speed(reviews/sec):1228. #Correct:6 #Tested:8 Testing Accuracy:75.0%Result:  [0.91973465] POSITIVE
    Progress:0.8% Speed(reviews/sec):1236. #Correct:7 #Tested:9 Testing Accuracy:77.7%Result:  [0.00179741] NEGATIVE
    Progress:0.9% Speed(reviews/sec):1086. #Correct:8 #Tested:10 Testing Accuracy:80.0%Result:  [0.96250962] POSITIVE
    Progress:1.0% Speed(reviews/sec):1128. #Correct:9 #Tested:11 Testing Accuracy:81.8%Result:  [0.01338331] NEGATIVE
    Progress:1.1% Speed(reviews/sec):1065. #Correct:10 #Tested:12 Testing Accuracy:83.3%Result:  [0.54420271] POSITIVE
    Progress:1.2% Speed(reviews/sec):1099. #Correct:11 #Tested:13 Testing Accuracy:84.6%Result:  [0.00012144] NEGATIVE
    Progress:1.3% Speed(reviews/sec):1141. #Correct:12 #Tested:14 Testing Accuracy:85.7%Result:  [0.98860438] POSITIVE
    Progress:1.4% Speed(reviews/sec):1152. #Correct:13 #Tested:15 Testing Accuracy:86.6%Result:  [0.00031434] NEGATIVE
    Progress:1.5% Speed(reviews/sec):1174. #Correct:14 #Tested:16 Testing Accuracy:87.5%Result:  [0.99883284] POSITIVE
    Progress:1.6% Speed(reviews/sec):1114. #Correct:15 #Tested:17 Testing Accuracy:88.2%Result:  [0.00161851] NEGATIVE
    Progress:1.7% Speed(reviews/sec):1098. #Correct:16 #Tested:18 Testing Accuracy:88.8%Result:  [0.99264946] POSITIVE
    Progress:1.8% Speed(reviews/sec):1134. #Correct:17 #Tested:19 Testing Accuracy:89.4%Result:  [0.81005829] POSITIVE
    Progress:1.9% Speed(reviews/sec):1136. #Correct:17 #Tested:20 Testing Accuracy:85.0%Result:  [0.99817604] POSITIVE
    Progress:2.0% Speed(reviews/sec):1110. #Correct:18 #Tested:21 Testing Accuracy:85.7%Result:  [0.01533085] NEGATIVE
    ...
    Progress:99.2% Speed(reviews/sec):1027. #Correct:847 #Tested:993 Testing Accuracy:85.2%Result:  [0.01798188] NEGATIVE
    Progress:99.3% Speed(reviews/sec):1027. #Correct:848 #Tested:994 Testing Accuracy:85.3%Result:  [0.67328968] POSITIVE
    Progress:99.4% Speed(reviews/sec):1027. #Correct:849 #Tested:995 Testing Accuracy:85.3%Result:  [0.00526858] NEGATIVE
    Progress:99.5% Speed(reviews/sec):1027. #Correct:850 #Tested:996 Testing Accuracy:85.3%Result:  [0.99759338] POSITIVE
    Progress:99.6% Speed(reviews/sec):1027. #Correct:851 #Tested:997 Testing Accuracy:85.3%Result:  [0.12086879] NEGATIVE
    Progress:99.7% Speed(reviews/sec):1028. #Correct:852 #Tested:998 Testing Accuracy:85.3%Result:  [0.52001354] POSITIVE
    Progress:99.8% Speed(reviews/sec):1028. #Correct:853 #Tested:999 Testing Accuracy:85.3%Result:  [0.0009184] NEGATIVE
    Progress:99.9% Speed(reviews/sec):1029. #Correct:854 #Tested:1000 Testing Accuracy:85.4%

# End of Project 5. 
## Watch the next video to see Andrew's solution, then continue on to the next lesson.
# Further Noise Reduction<a id='lesson_6'></a>


```python
Image(filename='sentiment_network_sparse_2.png')
```




![png](images/output_111_0.png)




```python
# words most frequently seen in a review with a "POSITIVE" label
pos_neg_ratios.most_common()
```




    [('edie', 4.6913478822291435),
     ('antwone', 4.477336814478207),
     ('din', 4.406719247264253),
     ('gunga', 4.189654742026425),
     ('goldsworthy', 4.174387269895637),
     ('gypo', 4.0943445622221),
     ('yokai', 4.0943445622221),
     ('paulie', 4.07753744390572),
     ('visconti', 3.9318256327243257),
     ('flavia', 3.9318256327243257),
     ('blandings', 3.871201010907891),
    ...
     ('evacuee', 1.9459101490553132),
     ('jeter', 1.9459101490553132),
     ('cosimo', 1.9459101490553132),
     ('heyerdahl', 1.9459101490553132),
     ('kasturba', 1.9459101490553132),
     ...]




```python
# words most frequently seen in a review with a "NEGATIVE" label
list(reversed(pos_neg_ratios.most_common()))[0:30]
```




    [('boll', -4.969813299576001),
     ('uwe', -4.624972813284271),
     ('thunderbirds', -4.127134385045092),
     ('beowulf', -4.110873864173311),
     ('dahmer', -3.9889840465642745),
     ('wayans', -3.9318256327243257),
     ('ajay', -3.871201010907891),
     ('grendel', -3.871201010907891),
     ('awfulness', -3.6635616461296463),
     ('seagal', -3.644143560272545),
     ('steaming', -3.6375861597263857),
     ('welch', -3.6109179126442243),
     ('deathstalker', -3.5553480614894135),
     ('sabretooth', -3.4339872044851463),
     ('interminable', -3.4339872044851463),
     ('forwarding', -3.4011973816621555),
     ('devgan', -3.367295829986474),
     ('gamera', -3.332204510175204),
     ('varma', -3.295836866004329),
     ('picker', -3.295836866004329),
     ('razzie', -3.295836866004329),
     ('dreck', -3.270835563798912),
     ('unwatchable', -3.258096538021482),
     ('nada', -3.2188758248682006),
     ('stinker', -3.2088254890146994),
     ('kirkland', -3.1780538303479458),
     ('nostril', -3.1780538303479458),
     ('giamatti', -3.1780538303479458),
     ('aag', -3.1354942159291497),
     ('demi', -3.1354942159291497)]




```python
# !pip install bokeh
from bokeh.models import ColumnDataSource, LabelSet
from bokeh.plotting import figure, show, output_file
from bokeh.io import output_notebook
output_notebook()
```



    <div class="bk-root">
        <a href="https://bokeh.pydata.org" target="_blank" class="bk-logo bk-logo-small bk-logo-notebook"></a>
        <span id="1001">Loading BokehJS ...</span>
    </div>





```python
hist, edges = np.histogram(list(map(lambda x:x[1],pos_neg_ratios.most_common())), density=True, bins=100, normed=True)

p = figure(tools="pan,wheel_zoom,reset,save",
           toolbar_location="above",
           title="Word Positive/Negative Affinity Distribution")
p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], line_color="#555555")
show(p)
```

    /usr/local/lib/python3.7/site-packages/ipykernel_launcher.py:1: DeprecationWarning: The normed argument is ignored when density is provided. In future passing both will result in an error.
      """Entry point for launching an IPython kernel.









  <div class="bk-root" id="590901e9-fbee-4b66-bcc1-04f449265069" data-root-id="1003"></div>






```python
frequency_frequency = Counter()

for word, cnt in total_counts.most_common():
    frequency_frequency[cnt] += 1
```


```python
hist, edges = np.histogram(list(map(lambda x:x[1],frequency_frequency.most_common())), density=True, bins=100, normed=True)

p = figure(tools="pan,wheel_zoom,reset,save",
           toolbar_location="above",
           title="The frequency distribution of the words in our corpus")
p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], line_color="#555555")
show(p)
```

    /usr/local/lib/python3.7/site-packages/ipykernel_launcher.py:1: DeprecationWarning: The normed argument is ignored when density is provided. In future passing both will result in an error.
      """Entry point for launching an IPython kernel.









  <div class="bk-root" id="c01ea868-885f-4089-9c08-3bddee7ac48f" data-root-id="1087"></div>





# Project 6: Reducing Noise by Strategically Reducing the Vocabulary<a id='project_6'></a>

**TODO:** Improve `SentimentNetwork`'s performance by reducing more noise in the vocabulary. Specifically, do the following:
* Copy the `SentimentNetwork` class from the previous project into the following cell.
* Modify `pre_process_data`:
>* Add two additional parameters: `min_count` and `polarity_cutoff`
>* Calculate the positive-to-negative ratios of words used in the reviews. (You can use code you've written elsewhere in the notebook, but we are moving it into the class like we did with other helper code earlier.)
>* Andrew's solution only calculates a postive-to-negative ratio for words that occur at least 50 times. This keeps the network from attributing too much sentiment to rarer words. You can choose to add this to your solution if you would like.  
>* Change so words are only added to the vocabulary if they occur in the vocabulary more than `min_count` times.
>* Change so words are only added to the vocabulary if the absolute value of their postive-to-negative ratio is at least `polarity_cutoff`
* Modify `__init__`:
>* Add the same two parameters (`min_count` and `polarity_cutoff`) and use them when you call `pre_process_data`


```python
# DONE: -Copy the SentimentNetwork class from Project 5 lesson
#       -Modify it according to the above instructions 

import time
import sys
import numpy as np

# Encapsulate our neural network in a class
class SentimentNetwork:
    def __init__(self, reviews, labels, min_count = 10, polarity_cutoff = 0.1, hidden_nodes = 10, learning_rate = 0.1):
        """Create a SentimenNetwork with the given settings
        Args:
            reviews(list) - List of reviews used for training
            labels(list) - List of POSITIVE/NEGATIVE labels associated with the given reviews
            hidden_nodes(int) - Number of nodes to create in the hidden layer
            learning_rate(float) - Learning rate to use while training
        
        """
        # Assign a seed to our random number generator to ensure we get
        # reproducable results during development 
        np.random.seed(1)

        # process the reviews and their associated labels so that everything
        # is ready for training
#        self.pre_process_data(reviews, labels)
        self.pre_process_data(reviews, labels, min_count, polarity_cutoff)
        
        # Build the network to have the number of hidden nodes and the learning rate that
        # were passed into this initializer. Make the same number of input nodes as
        # there are vocabulary words and create a single output node.
        self.init_network(len(self.review_vocab),hidden_nodes, 1, learning_rate)

    # P6: get polarity
    def get_polarity(self, reviews):
        positive_counts = Counter()
        negative_counts = Counter()
        total_counts = Counter()

        for i in range(len(reviews)):
            if(labels[i] == 'POSITIVE'):
                for word in reviews[i].split(" "):
                    positive_counts[word] += 1
                    total_counts[word] += 1
            else:
                for word in reviews[i].split(" "):
                    negative_counts[word] += 1
                    total_counts[word] += 1

        pos_neg_ratios = Counter()

        for term,cnt in list(total_counts.most_common()):
            if(cnt >= 50):
                pos_neg_ratio = positive_counts[term] / float(negative_counts[term]+1)
                pos_neg_ratios[term] = pos_neg_ratio

        for word,ratio in pos_neg_ratios.most_common():
            if(ratio > 1):
                pos_neg_ratios[word] = np.log(ratio)
            else:
                pos_neg_ratios[word] = -np.log((1 / (ratio + 0.01)))
        return positive_counts, negative_counts, total_counts
    def pre_process_data(self, reviews, labels, min_count, polarity_cutoff):
        
        # P6: get polarity
        positive_counts, negative_counts, total_counts = self.get_polarity(reviews)
        
        review_vocab = set()
        # TODO: populate review_vocab with all of the words in the given reviews
        #       Remember to split reviews into individual words 
        #       using "split(' ')" instead of "split()".
        ## review_vocab_counter = Counter()
        ## for i, review in enumerate(reviews) :
        ##     review_vocab_counter.update(review.lower().split(" "))
        ## review_vocab = review_vocab_counter.keys()
        for review in reviews:
            for word in review.split(" "):
                ## New for Project 6: only add words that occur at least min_count times
                #                     and for words with pos/neg ratios, only add words
                #                     that meet the polarity_cutoff
                if(total_counts[word] > min_count):
                    if(word in pos_neg_ratios.keys()):
                        if((pos_neg_ratios[word] >= polarity_cutoff) or (pos_neg_ratios[word] <= -polarity_cutoff)):
                            review_vocab.add(word)
                    else:
                        review_vocab.add(word)
                        
                        
        # Convert the vocabulary set to a list so we can access words via indices
        self.review_vocab = list(review_vocab)
        
        # TODO: populate label_vocab with all of the words in the given labels.
        #       There is no need to split the labels because each one is a single word.
        # Convert the label vocabulary set to a list so we can access labels via indices
        label_vocab = set(labels)
        
        self.label_vocab = list(label_vocab)
        
        # Store the sizes of the review and label vocabularies.
        self.review_vocab_size = len(self.review_vocab)
        self.label_vocab_size = len(self.label_vocab)
        print("SIZES: ",self.review_vocab_size,         self.label_vocab_size )
        
        # Create a dictionary of words in the vocabulary mapped to index positions
        self.word2index = {}
        # TODO: populate self.word2index with indices for all the words in self.review_vocab
        #       like you saw earlier in the notebook
        for i, word in enumerate(self.review_vocab):
            self.word2index[word] = i
        # Create a dictionary of labels mapped to index positions
        self.label2index = {}
        # TODO: do the same thing you did for self.word2index and self.review_vocab, 
        #       but for self.label2index and self.label_vocab instead
        for i, wordLabel in enumerate(self.label_vocab):
            self.label2index[wordLabel] = i
        
    def init_network(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Store the number of nodes in input, hidden, and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Store the learning rate
        self.learning_rate = learning_rate

        # Initialize weights
        
        # TODO: initialize self.weights_0_1 as a matrix of zeros. These are the weights between
        #       the input layer and the hidden layer.
        # DOUBT: SHAPE?
        self.weights_0_1 = np.zeros((self.input_nodes, self.hidden_nodes))
        
        # TODO: initialize self.weights_1_2 as a matrix of random values. 
        #       These are the weights between the hidden layer and the output layer.
        # DOUBT: SHAPE?
        self.weights_1_2 = np.random.normal(0.0, self.output_nodes**-0.5, 
                                                (self.hidden_nodes, self.output_nodes))
        
        #       1 x hidden_nodes, with all values initialized to zero
        self.layer_1 = np.zeros((1,hidden_nodes))
    
    def silly_words(self):
        return ['.', '', 'the', 'to', 'i', 'is', 'a', 'that', 'of', 'it', 'at', 'as', 'my', 'in', 'me']
                
    def get_target_for_label(self,label):
        # DONE: Copy the code you wrote for get_target_for_label 
        #       earlier in this notebook. 
        """Convert a label to `0` or `1`.
        Args:
            label(string) - Either "POSITIVE" or "NEGATIVE".
        Returns:
            `0` or `1`.
        """
        return 0 if "NEGATIVE" == label else 1
        
    def sigmoid(self,x):
        # DONE: Return the result of calculating the sigmoid activation function
        #       shown in the lectures
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_output_2_derivative(self,output):
        # DONE: Return the derivative of the sigmoid activation function, 
        #       where "output" is the original output from the sigmoid fucntion 
        # dy/dx = f(x)' = f(x) * (1 - f(x))
        return output * (1 - output)

    def train(self, training_reviews_raw, training_labels):
        
        # Create Indexes list 
        training_reviews = list()
        for review in training_reviews_raw:
            indices = set()
            for word in review.split(" "):
                if(word in self.word2index.keys()):
                    indices.add(self.word2index[word])
            training_reviews.append(list(indices))

        
        # make sure out we have a matching number of reviews and labels
        assert(len(training_reviews) == len(training_labels))
        
        # Keep track of correct predictions to display accuracy during training 
        correct_so_far = 0
        
        # Remember when we started for printing time statistics
        start = time.time()

        # loop through all the given reviews and run a forward and backward pass,
        # updating weights for every item
        for i in range(len(training_reviews)):
            
            # DONE: Get the next review and its correct label
            review = training_reviews[i]
            label = training_labels[i]
            
            # DONE: Implement the forward pass through the network. 
            #       That means use the given review to update the input layer, 
            #       then calculate values for the hidden layer,
            #       and finally calculate the output layer.
            # 
            #       Do not use an activation function for the hidden layer,
            #       but use the sigmoid activation function for the output layer.
            
            # Hidden layer

            # review is Review Indexes of its words.

            self.layer_1 *= 0
            for index in review:
                self.layer_1 += self.weights_0_1[index]
            # Output layer
            layer_2 = self.sigmoid(self.layer_1.dot(self.weights_1_2))
            
            
            # DONE: Implement the back propagation pass here. 
            #       That means calculate the error for the forward pass's prediction
            #       and update the weights in the network according to their
            #       contributions toward the error, as calculated via the
            #       gradient descent and back propagation algorithms you 
            #       learned in class.

            ### Backward pass ###
            # Output error
            layer_2_error = layer_2 - self.get_target_for_label(label) # Output layer error is the difference between desired target and actual output.
            layer_2_delta = layer_2_error * self.sigmoid_output_2_derivative(layer_2)
            # Backpropagated error
            layer_1_error = layer_2_delta.dot(self.weights_1_2.T) # errors propagated to the hidden layer
            layer_1_delta = layer_1_error # hidden layer gradients - no nonlinearity so it's the same as the error
            # Update the weights
            self.weights_1_2 -= self.layer_1.T.dot(layer_2_delta) * self.learning_rate # update hidden-to-output weights with gradient descent step

            ## New for Project 5: Only update the weights that were used in the forward pass
            # DOUBT:
            for index in review:
                self.weights_0_1[index] -= layer_1_delta[0] * self.learning_rate # update input-to-hidden weights with gradient descent step

            
            # DONE: Keep track of correct predictions. To determine if the prediction was
            #       correct, check that the absolute value of the output error 
            #       is less than 0.5. If so, add one to the correct_so_far count.
             # Keep track of correct predictions.
            if(layer_2 >= 0.5 and label == 'POSITIVE'):
                correct_so_far += 1
            elif(layer_2 < 0.5 and label == 'NEGATIVE'):
                correct_so_far += 1
            
            # For debug purposes, print out our prediction accuracy and speed 
            # throughout the training process. 

            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0
            
            sys.stdout.write("\rProgress:" + str(100 * i/float(len(training_reviews)))[:4] \
                             + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                             + " #Correct:" + str(correct_so_far) + " #Trained:" + str(i+1) \
                             + " Training Accuracy:" + str(correct_so_far * 100 / float(i+1))[:4] + "%")
            if(i % 2500 == 0):
                print("")
    
    def test(self, testing_reviews, testing_labels):
        """
        Attempts to predict the labels for the given testing_reviews,
        and uses the test_labels to calculate the accuracy of those predictions.
        """
        
        # keep track of how many correct predictions we make
        correct = 0

        # we'll time how many predictions per second we make
        start = time.time()

        # Loop through each of the given reviews and call run to predict
        # its label. 
        for i in range(len(testing_reviews)):
            pred = self.run(testing_reviews[i])
            if(pred == testing_labels[i]):
                correct += 1
            
            # For debug purposes, print out our prediction accuracy and speed 
            # throughout the prediction process. 

            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0
            
            sys.stdout.write("\rProgress:" + str(100 * i/float(len(testing_reviews)))[:4] \
                             + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                             + " #Correct:" + str(correct) + " #Tested:" + str(i+1) \
                             + " Testing Accuracy:" + str(correct * 100 / float(i+1))[:4] + "%")
    
    def run(self, review):
        """
        Returns a POSITIVE or NEGATIVE prediction for the given review.
        """
        # DONE: Run a forward pass through the network, like you did in the
        #       "train" function. That means use the given review to 
        #       update the input layer, then calculate values for the hidden layer,
        #       and finally calculate the output layer.
        #
        #       Note: The review passed into this function for prediction 
        #             might come from anywhere, so you should convert it 
        #             to lower case prior to using it.
        self.layer_1 *= 0
        unique_indices = set()
        for word in review.lower().split(" "):
            if word in self.word2index.keys():
                unique_indices.add(self.word2index[word])

        for index in unique_indices:
            self.layer_1 += self.weights_0_1[index]
        
        layer2 = self.sigmoid(self.layer_1.dot(self.weights_1_2))
        
        # DONE: The output layer should now contain a prediction. 
        #       Return `POSITIVE` for predictions greater-than-or-equal-to `0.5`, 
        #       and `NEGATIVE` otherwise.
        prediction = layer2[0]
        print("Result: ",prediction, "POSITIVE" if prediction >= 0.5 else "NEGATIVE")
        return  "POSITIVE" if prediction >= 0.5 else "NEGATIVE"
```

Run the following cell to train your network with a small polarity cutoff.


```python
mlp = SentimentNetwork(reviews[:-1000],labels[:-1000],min_count=20,polarity_cutoff=0.05,learning_rate=0.01)
mlp.train(reviews[:-1000],labels[:-1000])
```

    SIZES:  11700 2
    Progress:0.0% Speed(reviews/sec):0.0 #Correct:1 #Trained:1 Training Accuracy:100.%
    Progress:10.4% Speed(reviews/sec):1492. #Correct:1991 #Trained:2501 Training Accuracy:79.6%
    Progress:20.8% Speed(reviews/sec):1505. #Correct:4059 #Trained:5001 Training Accuracy:81.1%
    Progress:31.2% Speed(reviews/sec):1520. #Correct:6171 #Trained:7501 Training Accuracy:82.2%
    Progress:41.6% Speed(reviews/sec):1538. #Correct:8326 #Trained:10001 Training Accuracy:83.2%
    Progress:52.0% Speed(reviews/sec):1529. #Correct:10493 #Trained:12501 Training Accuracy:83.9%
    Progress:62.5% Speed(reviews/sec):1488. #Correct:12637 #Trained:15001 Training Accuracy:84.2%
    Progress:72.9% Speed(reviews/sec):1478. #Correct:14775 #Trained:17501 Training Accuracy:84.4%
    Progress:83.3% Speed(reviews/sec):1481. #Correct:16955 #Trained:20001 Training Accuracy:84.7%
    Progress:93.7% Speed(reviews/sec):1477. #Correct:19151 #Trained:22501 Training Accuracy:85.1%
    Progress:99.9% Speed(reviews/sec):1463. #Correct:20472 #Trained:24000 Training Accuracy:85.3%

And run the following cell to test it's performance. It should be 


```python
mlp.test(reviews[-1000:],labels[-1000:])
```

    Result:  [0.86502845] POSITIVE
    Progress:0.0% Speed(reviews/sec):0.0 #Correct:1 #Tested:1 Testing Accuracy:100.%Result:  [0.76158792] POSITIVE
    Progress:0.1% Speed(reviews/sec):678.3 #Correct:1 #Tested:2 Testing Accuracy:50.0%Result:  [0.63584607] POSITIVE
    Progress:0.2% Speed(reviews/sec):911.2 #Correct:2 #Tested:3 Testing Accuracy:66.6%Result:  [0.28240509] NEGATIVE
    Progress:0.3% Speed(reviews/sec):1091. #Correct:3 #Tested:4 Testing Accuracy:75.0%Result:  [0.89812223] POSITIVE
    Progress:0.4% Speed(reviews/sec):1227. #Correct:4 #Tested:5 Testing Accuracy:80.0%Result:  [0.00111141] NEGATIVE
    Progress:0.5% Speed(reviews/sec):1210. #Correct:5 #Tested:6 Testing Accuracy:83.3%Result:  [0.93158331] POSITIVE
    Progress:0.6% Speed(reviews/sec):1156. #Correct:6 #Tested:7 Testing Accuracy:85.7%Result:  [0.00208013] NEGATIVE
    Progress:0.7% Speed(reviews/sec):949.7 #Correct:7 #Tested:8 Testing Accuracy:87.5%Result:  [0.94077725] POSITIVE
    Progress:0.8% Speed(reviews/sec):489.9 #Correct:8 #Tested:9 Testing Accuracy:88.8%Result:  [0.00116519] NEGATIVE
    ...
    Progress:99.7% Speed(reviews/sec):968.4 #Correct:858 #Tested:998 Testing Accuracy:85.9%Result:  [0.34205553] NEGATIVE
    Progress:99.8% Speed(reviews/sec):968.8 #Correct:858 #Tested:999 Testing Accuracy:85.8%Result:  [0.00049024] NEGATIVE
    Progress:99.9% Speed(reviews/sec):969.2 #Correct:859 #Tested:1000 Testing Accuracy:85.9%

Run the following cell to train your network with a much larger polarity cutoff.


```python
mlp = SentimentNetwork(reviews[:-1000],labels[:-1000],min_count=20,polarity_cutoff=0.8,learning_rate=0.01)
mlp.train(reviews[:-1000],labels[:-1000])
```

    SIZES:  3214 2
    Progress:0.0% Speed(reviews/sec):0.0 #Correct:1 #Trained:1 Training Accuracy:100.%
    Progress:10.4% Speed(reviews/sec):6080. #Correct:2119 #Trained:2501 Training Accuracy:84.7%
    Progress:20.8% Speed(reviews/sec):6034. #Correct:4238 #Trained:5001 Training Accuracy:84.7%
    Progress:31.2% Speed(reviews/sec):5666. #Correct:6367 #Trained:7501 Training Accuracy:84.8%
    Progress:41.6% Speed(reviews/sec):5702. #Correct:8514 #Trained:10001 Training Accuracy:85.1%
    Progress:52.0% Speed(reviews/sec):5592. #Correct:10661 #Trained:12501 Training Accuracy:85.2%
    Progress:62.5% Speed(reviews/sec):5608. #Correct:12813 #Trained:15001 Training Accuracy:85.4%
    Progress:72.9% Speed(reviews/sec):5423. #Correct:14927 #Trained:17501 Training Accuracy:85.2%
    Progress:83.3% Speed(reviews/sec):5335. #Correct:17117 #Trained:20001 Training Accuracy:85.5%
    Progress:93.7% Speed(reviews/sec):5361. #Correct:19300 #Trained:22501 Training Accuracy:85.7%
    Progress:99.9% Speed(reviews/sec):5365. #Correct:20591 #Trained:24000 Training Accuracy:85.7%

And run the following cell to test it's performance.


```python
mlp.test(reviews[-1000:],labels[-1000:])
```

    Result:  [0.34275465] NEGATIVE
    Progress:0.0% Speed(reviews/sec):0.0 #Correct:0 #Tested:1 Testing Accuracy:0.0%Result:  [0.1535638] NEGATIVE
    Progress:0.1% Speed(reviews/sec):541.4 #Correct:1 #Tested:2 Testing Accuracy:50.0%Result:  [0.68804185] POSITIVE
    Progress:0.2% Speed(reviews/sec):843.8 #Correct:2 #Tested:3 Testing Accuracy:66.6%Result:  [0.06494502] NEGATIVE
    Progress:0.3% Speed(reviews/sec):876.6 #Correct:3 #Tested:4 Testing Accuracy:75.0%Result:  [0.48762603] NEGATIVE
    Progress:0.4% Speed(reviews/sec):1020. #Correct:3 #Tested:5 Testing Accuracy:60.0%Result:  [0.03052888] NEGATIVE
    Progress:0.5% Speed(reviews/sec):1132. #Correct:4 #Tested:6 Testing Accuracy:66.6%Result:  [0.73329719] POSITIVE
    Progress:0.6% Speed(reviews/sec):1195. #Correct:5 #Tested:7 Testing Accuracy:71.4%Result:  [0.00698967] NEGATIVE
    Progress:0.7% Speed(reviews/sec):1283. #Correct:6 #Tested:8 Testing Accuracy:75.0%Result:  [0.82609984] POSITIVE
    Progress:0.8% Speed(reviews/sec):1379. #Correct:7 #Tested:9 Testing Accuracy:77.7%Result:  [0.00998146] NEGATIVE
    Progress:0.9% Speed(reviews/sec):1401. #Correct:8 #Tested:10 Testing Accuracy:80.0%Result:  [0.83714573] POSITIVE
    Progress:1.0% Speed(reviews/sec):1444. #Correct:9 #Tested:11 Testing Accuracy:81.8%Result:  [0.09680347] NEGATIVE
    Progress:1.1% Speed(reviews/sec):1508. #Correct:10 #Tested:12 Testing Accuracy:83.3%Result:  [0.35874272] NEGATIVE
    Progress:1.2% Speed(reviews/sec):1574. #Correct:10 #Tested:13 Testing Accuracy:76.9%Result:  [0.00335004] NEGATIVE
    Progress:1.3% Speed(reviews/sec):1587. #Correct:11 #Tested:14 Testing Accuracy:78.5%Result:  [0.42882705] NEGATIVE
    Progress:1.4% Speed(reviews/sec):1632. #Correct:11 #Tested:15 Testing Accuracy:73.3%Result:  [0.01317975] NEGATIVE
    Progress:1.5% Speed(reviews/sec):1655. #Correct:12 #Tested:16 Testing Accuracy:75.0%Result:  [0.93712203] POSITIVE
    Progress:1.6% Speed(reviews/sec):1654. #Correct:13 #Tested:17 Testing Accuracy:76.4%Result:  [0.00444428] NEGATIVE
    ...
    Progress:99.2% Speed(reviews/sec):1257. #Correct:841 #Tested:993 Testing Accuracy:84.6%Result:  [0.11097736] NEGATIVE
    Progress:99.3% Speed(reviews/sec):1257. #Correct:842 #Tested:994 Testing Accuracy:84.7%Result:  [0.12106898] NEGATIVE
    Progress:99.4% Speed(reviews/sec):1258. #Correct:842 #Tested:995 Testing Accuracy:84.6%Result:  [0.0173449] NEGATIVE
    Progress:99.5% Speed(reviews/sec):1258. #Correct:843 #Tested:996 Testing Accuracy:84.6%Result:  [0.97766593] POSITIVE
    Progress:99.6% Speed(reviews/sec):1201. #Correct:844 #Tested:997 Testing Accuracy:84.6%Result:  [0.00996465] NEGATIVE
    Progress:99.7% Speed(reviews/sec):1202. #Correct:845 #Tested:998 Testing Accuracy:84.6%Result:  [0.36458109] NEGATIVE
    Progress:99.8% Speed(reviews/sec):1202. #Correct:845 #Tested:999 Testing Accuracy:84.5%Result:  [0.00082275] NEGATIVE
    Progress:99.9% Speed(reviews/sec):1203. #Correct:846 #Tested:1000 Testing Accuracy:84.6%

# End of Project 6. 
## Watch the next video to see Andrew's solution, then continue on to the next lesson.

# Analysis: What's Going on in the Weights?<a id='lesson_7'></a>


```python
mlp_full = SentimentNetwork(reviews[:-1000],labels[:-1000],min_count=0,polarity_cutoff=0,learning_rate=0.01)
```

    SIZES:  72810 2



```python
mlp_full.train(reviews[:-1000],labels[:-1000])
```

    Progress:0.0% Speed(reviews/sec):0.0 #Correct:1 #Trained:1 Training Accuracy:100.%
    Progress:10.4% Speed(reviews/sec):572.7 #Correct:1962 #Trained:2501 Training Accuracy:78.4%
    Progress:20.8% Speed(reviews/sec):564.5 #Correct:4002 #Trained:5001 Training Accuracy:80.0%
    Progress:31.2% Speed(reviews/sec):568.0 #Correct:6120 #Trained:7501 Training Accuracy:81.5%
    Progress:41.6% Speed(reviews/sec):567.6 #Correct:8271 #Trained:10001 Training Accuracy:82.7%
    Progress:52.0% Speed(reviews/sec):559.5 #Correct:10431 #Trained:12501 Training Accuracy:83.4%
    Progress:62.5% Speed(reviews/sec):552.9 #Correct:12565 #Trained:15001 Training Accuracy:83.7%
    Progress:72.9% Speed(reviews/sec):547.0 #Correct:14670 #Trained:17501 Training Accuracy:83.8%
    Progress:83.3% Speed(reviews/sec):546.9 #Correct:16833 #Trained:20001 Training Accuracy:84.1%
    Progress:93.7% Speed(reviews/sec):545.6 #Correct:19015 #Trained:22501 Training Accuracy:84.5%
    Progress:99.9% Speed(reviews/sec):545.7 #Correct:20335 #Trained:24000 Training Accuracy:84.7%


```python
Image(filename='sentiment_network_sparse.png')
```




![png](images/output_132_0.png)




```python
def get_most_similar_words(focus = "horrible"):
    most_similar = Counter()

    for word in mlp_full.word2index.keys():
        most_similar[word] = np.dot(mlp_full.weights_0_1[mlp_full.word2index[word]],mlp_full.weights_0_1[mlp_full.word2index[focus]])
    
    return most_similar.most_common()
```


```python
get_most_similar_words("excellent")
```




    [('excellent', 0.13672950757352464),
     ('perfect', 0.1254828608722594),
     ('amazing', 0.09182763392599967),
     ('today', 0.09022366269441416),
     ('wonderful', 0.08935597696221456),
     ('fun', 0.08750446667420683),
     ('great', 0.087141758882292),
     ('best', 0.08581088561788058),
     ('liked', 0.0776976291238434),
     ('definitely', 0.07662878140696602),
     ('brilliant', 0.07342385876927901),
     ('loved', 0.07328542892812212),
     ('favorite', 0.07278113603616075),
     ('superb', 0.07173620717850504),
     ('fantastic', 0.07092219191626618),
     ('job', 0.06916061720763404),
     ('incredible', 0.0664240779526144),
     ('enjoyable', 0.06563256050288879),
     ('rare', 0.06481921266261505),
     ('highly', 0.06388945335097052),
     ('enjoyed', 0.06212754610181292),
     ('wonderfully', 0.062055178604090135),
     ('perfectly', 0.061093208811887366),
     ('fascinating', 0.060663547937493845),
     ('bit', 0.05965542704565304),
     ('gem', 0.05951085929615678),
     ('outstanding', 0.058860808147083006),
     ('beautiful', 0.058613934703162035),
     ('surprised', 0.05827331448256295),
     ('worth', 0.0576574842364712),
     ('especially', 0.05742202078176078),
     ('refreshing', 0.05731053209226574),
     ('entertaining', 0.05661203383562921),
     ('hilarious', 0.05616854103228663),
     ('masterpiece', 0.054993988649431544),
     ('simple', 0.05448408313492406),
     ('subtle', 0.0543688830335086),
     ('funniest', 0.05345716487130268),
     ('solid', 0.05290356474362068),
     ('awesome', 0.052489194202770394),
     ('always', 0.052260328525345255),
     ('noir', 0.05153019472640689),
     ('guys', 0.05110941364564267),
     ('sweet', 0.05081893031752598),
     ('unique', 0.05067016226358917),
     ('very', 0.05013299494852848),
     ('heart', 0.049948058498243596),
     ('moving', 0.04942460116437912),
     ('atmosphere', 0.04884250089591284),
     ('strong', 0.04857088063175918),
     ('remember', 0.0484790369422913),
     ('believable', 0.048415384391603804),
     ('shows', 0.04833604560803956),
     ('love', 0.04731064816092462),
     ('beautifully', 0.04711871744081489),
     ('both', 0.04695727890148032),



```python
get_most_similar_words("terrible")
```




    [('worst', 0.16966107259049845),
     ('awful', 0.12026847019691246),
     ('waste', 0.11945367265311005),
     ('poor', 0.09275888757443547),
     ('terrible', 0.09142538719772793),
     ('dull', 0.0842092716782236),
     ('poorly', 0.08124154451604203),
     ('disappointment', 0.08006475962136872),
     ('fails', 0.07859977372333751),
     ('disappointing', 0.07733948548032335),
     ('boring', 0.07712785874801291),
     ('unfortunately', 0.07550244970585908),
     ('worse', 0.07060183536419465),
     ('mess', 0.07056429962359043),
     ('stupid', 0.06948482283254304),
     ('badly', 0.06688890366622856),
     ('annoying', 0.06568702190337417),
     ('bad', 0.06309381453757214),
     ('save', 0.06288059749586573),
     ('disappointed', 0.06269235381207287),
     ('wasted', 0.06138718302805129),
     ('supposed', 0.060985452957725145),
     ('horrible', 0.06012177233938012),
     ('laughable', 0.05869840628546764),
     ('crap', 0.05810452866788456),
     ('basically', 0.057218840369636155),
     ('nothing', 0.05715822004303422),
     ('ridiculous', 0.056905481068931445),
     ('lacks', 0.05576656588946545),
     ('lame', 0.05561600905811017),
    




```python
import matplotlib.colors as colors

words_to_visualize = list()
for word, ratio in pos_neg_ratios.most_common(500):
    if(word in mlp_full.word2index.keys()):
        words_to_visualize.append(word)
    
#for word, ratio in list(reversed(pos_neg_ratios.most_common()))[0:500]:
for word, ratio in list(reversed(pos_neg_ratios.most_common()))[0:750]:
    if(word in mlp_full.word2index.keys()):
        words_to_visualize.append(word)
```


```python
pos = 0
neg = 0

colors_list = list()
vectors_list = list()
for word in words_to_visualize:
    if word in pos_neg_ratios.keys():
        vectors_list.append(mlp_full.weights_0_1[mlp_full.word2index[word]])
        if(pos_neg_ratios[word] > 0):
            pos+=1
            colors_list.append("#00ff00")
        else:
            neg+=1
            colors_list.append("#000000")
```


```python
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=0)
words_top_ted_tsne = tsne.fit_transform(vectors_list)
```


```python
p = figure(tools="pan,wheel_zoom,reset,save",
           toolbar_location="above",
           title="vector T-SNE for most polarized words")

source = ColumnDataSource(data=dict(x1=words_top_ted_tsne[:,0],
                                    x2=words_top_ted_tsne[:,1],
                                    names=words_to_visualize,
                                    color=colors_list))

p.scatter(x="x1", y="x2", size=8, source=source, fill_color="color")

word_labels = LabelSet(x="x1", y="x2", text="names", y_offset=6,
                  text_font_size="8pt", text_color="#555555",
                  source=source, text_align='center')
# p.add_layout(word_labels)

show(p)

# green indicates positive words, black indicates negative words
```


![png](images/vector-space.png)





  <div class="bk-root" id="48570935-7bff-434e-b4f1-e6adcd750787" data-root-id="1532"></div>




