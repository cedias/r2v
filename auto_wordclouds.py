import argparse
import matplotlib.pyplot as plt
import itertools
from wordcloud import WordCloud
from VectReco.R2VModel import R2VModel

parser = argparse.ArgumentParser()
parser.add_argument("--n",default=30, type=int)
parser.add_argument("model", type=str)
parser.add_argument("name_model", type=str)
args = parser.parse_args()


mod = R2VModel.from_w2v_text(args.model,binary=True)

labels = ["r_1.0","r_2.0","r_3.0","r_4.0","r_5.0"]


for l in labels:

    words =  mod.most_similar(vect=mod[l],limit="words",topn=args.n)

    freq = [(word,round(((((sim+1)/2)/1)*100))) for word,sim in words]

    duped = [[w]*int(f) for w,f in freq]
    text = " ".join(itertools.chain.from_iterable(duped))

    # text = " ".join([w for w,s in words])
    wordcloud = WordCloud(width=200, height=200, margin=2, ranks_only=False, prefer_horizontal=0.9, mask=None, scale=1, max_words=200, background_color='white').generate(text)
    # Open a plot of the generated image.
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.savefig("/home/dias/Bureau/wordcloud/{}_{}.png".format(args.name_model,l))