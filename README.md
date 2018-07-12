# Multimodal Author Profiling @ PAN 2018
Detailed description of this task can be found @[PAN 2018](https://pan.webis.de/clef18/pan18-web/author-profiling.html). This code only analyzes tweets in English. 

Dataset:The dataset used for this experiments can be downloaded from the [PAN 2018](https://pan.webis.de/clef18/pan18-web/author-profiling.html).

*Dependencies:*
1. gensim
2. sklearn
3. nltk

Other requirements:

The GloVe models (100d & 200d) are required for word embeddings. 

For image captioning, [image caption generation using chainer](https://github.com/apple2373/chainer-caption) was used. Need to extract image captions before using the above tool and store it in a csv file (format:imageid \t text).


# Running the code

*python master.py training_input_add test_input_add test_output_add*

Output will be a xml file:

  <author id="author-id"
	  lang="en|es|ar"
	  gender_txt="male|female"
	  gender_img="male|female"
	  gender_comb="male|female"
  />


# Reference

Please cite the following paper if you find this code is useful.

B. G. Patra, G. Das, and D. Das. 2018. Multimodal Author Profiling for Twitter - Notebook for PAN at CLEF 2018. In *Proceedings of the PAN 2018 at CLEF-2018*, Avignon, France. [link](https://pan.webis.de/clef18/pan18-web/proceedings.html)

If you have any query please e-mail us. We welcome bug fixes and new features.
