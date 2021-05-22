# recommender_system
The project about building concrete recommender system by using Content-Based and Collaborative Filtering approaches

---
**Table of Contents**

* [Brief Introduction](#brief-introduction)
* [Directory structure](#directory-structure)
* [Explaination of project](#explaination-of-project)
* [Usage examples](#usage-examples)
* [Discussion of Results](#discussion-of-results)
* [About Author](#about-author)

---
## Brief Introduction



---
## Directory structure
```
.
├── data
│   ├── movies.csv
│   ├── ratings.csv
│   ├── tags.csv
├── packages
│   ├── __init__.py
│   ├── collaborative.py
│   ├── content.py
│   └── utils.py
├── LICENSE
├── README.md
├── Terminal Command Run Examples.txt
├── __init__.py
├── recommender_system.py
├── requirements.txt
```

---
## Explaination of project



---
## Usage examples

The main [recommender_system](https://github.com/singultek/recommender_system/blob/main/recommender_system.py) code can run in two working modes: content based and collaborative filtering based approaches. Following help commands and usage examples can be followed to run the code from command line:


### Content

In order to see the help for content mode, one can run following on the command line:
`$ python3 recommender_system.py content -h`

```

usage: recommender_system.py content [-h] [--tfidf TFIDF] [--lsi LSI] dataset_path userID num_recommendation

Content-Based Approach

positional arguments:
  dataset_path        A dataset folder where the data is stored
  userID              A integer value which indicates the userID to recommend movies
  num_recommendation  A integer value which indicates the number of recommention given for userID

optional arguments:
  -h, --help          show this help message and exit
  --tfidf TFIDF       (default = True) The boolean value which indicates whether TF-IDF technique will be applied to the counting or not
  --lsi LSI           (default = True) The boolean value which indicates whether dictionary reduction with LSI technique will be applied or not

```
One example of usage of train mode:

`$ python3 recommender_system.py content ./data 65 10`: builts a recommender system with content based approach with the dataset on `./data` for userID `65` and `10` maximum number of recommendations.



---

### Discussion of Results


                          
                               
---
### About Author

I'm Sinan Gültekin, a master student on Computer and Automation Engineering at University of Siena. 

For any suggestions or questions, you can contact me via <singultek@gmail.com>

Distributed under the Apache-2.0 License. _See ``LICENSE`` for more information._
