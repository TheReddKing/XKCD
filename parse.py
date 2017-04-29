from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

import csv
import unicodedata
import re
import pdb


from sumy.parsers.html import HtmlParser
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

LANGUAGE = "english"
SENTENCES_COUNT = 2

stemmer = Stemmer(LANGUAGE)
summarizer = Summarizer(stemmer)
summarizer.stop_words = get_stop_words(LANGUAGE)

filenames = ["2013","2014","2015above","2016_03","2016_04"]
# filenames = ["2016_03","2016_04","2013"]
for filename in filenames:
    f = open(filename + '.csv')
    csv_f = csv.reader(f)
    writer = wr = csv.writer(open(filename + '_clean.csv', 'wb'), quoting=csv.QUOTE_ALL)
    for row in csv_f:
        # print row
        # Find the actual number
        #['v_body', 'v_name', 'k_body', 'k_name', 'k_subreddit', 'k_ups', 'v_ups']
        # 0           1           2       3       4               5       6
        choices = row[2].decode('utf-8').encode('ascii', 'ignore').split("/")
        xkcdval = "0"
        for choice in choices:
            if choice.isdigit():
                xkcdval = choice
                break
        # Remove 1053, 37,
        if(xkcdval != "0"):
            val = int(xkcdval)
            if(val > 2000 or val == 1053 or val == 37):
                continue

            line  = row[0].decode('utf-8').encode('ascii', 'ignore').replace("\n"," ").lower()
            line = line.replace("*"," ")
            line = line.replace("~~"," ")

            # Summerize text
            line2 = ""
            for sentence in summarizer(PlaintextParser.from_string(line,Tokenizer(LANGUAGE)).document, SENTENCES_COUNT):
                # pdb.set_trace()
                line2 += str(sentence) + " "

            # line = line.replace("*"," ")
            # line = line.replace("*"," ")
            writer.writerow([line2.decode('utf-8').encode('ascii', 'ignore'), xkcdval])
