# importing packages
import os
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests
import re

import string
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
from nltk.corpus import cmudict
import syllapy


"""
                            *******************
                            | Data Collection |
                            *******************    

1. Textual data from the articles is collected using the BeautifulSoup package.
2. Title and entire text from each article are extracted.
3. The article title and text are saved in the text file for each article.

"""

# create Data folder to store the text files.
if not os.path.exists('Data'):
    os.makedirs('Data')

# read the input data.
input_data = pd.read_csv("input.csv")
print ("Input data loaded into dataframe successfully.")
urlDict = pd.Series(input_data.URL.values,index=input_data.URL_ID).to_dict()

# headers to avoid mod-security errors during web scraping.
headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.12; rv:55.0) Gecko/20100101 Firefox/55.0'}

for keys , values in urlDict.items():
    url = urlDict[keys]
    print ("\nData collection is started from {}th URL - {} ...".format(keys ,url))
    filename = "Data/" + str(keys) + ".txt"
    data  = requests.get(url, headers=headers).text
    soup = BeautifulSoup(data, 'html5lib')
    articleFullText = ""
    articleTitle = ""
    for title in soup.find("title"):
        articleTitle = str(title)
    
    articleText = ""
    paragraphList = []
    paragraphs = soup.find_all('p')
    for paragraph in paragraphs:
        articleText = articleText + " " + str(paragraph)
    cleanedArticleText = re.sub("\<strong>.*?\</strong>", "", articleText)
    cleanedArticleText = re.sub("\<.*?\>", "", cleanedArticleText)    
    
    articleFullText = articleTitle + "\n" + cleanedArticleText
    print ("Data collection is completed from {}th URL.".format(keys))
    file = open(filename, "w")
    file.write(articleFullText)
    file.flush()
    file.close()
print ("\nData collection is completed successfully.")


"""
                        ***************************************
                        | Load StopWords and MasterDictionary |
                        ***************************************

1. Load all types of stopwords from the StopWords folder and convert them into lowercase.
2. Merge all stopwords into a list for further analysis.
3. Load positive and negative words MasterDictionary folder and convert them into lowercase.
4. Merge all positive and negative words in a list for further analysis.

"""

# reading stopwords from StopWords_Auditor.txt file.
with open('StopWords/StopWords_Auditor.txt') as swAuditFile:
    swAudit = swAuditFile.read()
    stopWordAuditList = swAudit.split()

# reading stopwords from StopWords_Currencies.txt file.
stopWordAuditList = []
with open("StopWords/StopWords_Currencies.txt", encoding='latin-1') as stopWordCurrencyFile:
    for line in stopWordCurrencyFile:
        swCurrencyList = line.strip().split('|')
        swCurrency = swCurrencyList[0].strip()
        stopWordAuditList.append(swCurrency)

# reading stopwords from StopWords_DatesandNumbers.txt file.
stopWordDateNumberList = []
with open("StopWords/StopWords_DatesandNumbers.txt") as swDateNumberFile:
    for line in swDateNumberFile:
        swDateNumberList = line.strip().split('|')
        swDatesandNumber = swDateNumberList[0].strip()
        stopWordDateNumberList.append(swDatesandNumber)

# reading stopwords from StopWords_Generic.txt file.
with open('StopWords/StopWords_Generic.txt') as swGenericFile:
    swGeneric = swGenericFile.read()
    stopWordGenericList = swGeneric.split()

# reading stopwords from StopWords_GenericLong.txt file.
with open('StopWords/StopWords_GenericLong.txt') as swGenericLongFile:
    swGenericLong = swGenericLongFile.read()
    stopWordGenericLongList = swGenericLong.split()

# reading stopwords from StopWords_Geographic.txt file.
stopWordGeographicList = []
with open("StopWords/StopWords_Geographic.txt") as swGeographicFile:
    for line in swGeographicFile:
        swGeographicList = line.strip().split('|')
        swGeographic = swGeographicList[0].strip()
        stopWordGeographicList.append(swGeographic)

# reading stopwords from StopWords_Names.txt file.
stopWordNameList = []
with open("StopWords/StopWords_Names.txt") as swNameFile:
    for line in swNameFile:
        swNameList = line.strip().split('|')
        swName = swNameList[0].strip()
        stopWordNameList.append(swName)

# adding all stopwords from all files.
allStopWordList = stopWordAuditList + stopWordAuditList + \
                    stopWordDateNumberList + stopWordGenericList + \
                    stopWordGenericLongList + stopWordGenericLongList + \
                    stopWordGeographicList + stopWordNameList

# convert all stopwords into lower cases and filter the unique ones.
finalStopWordList = []
for stopword in allStopWordList:
    swLowerCase = stopword.lower()
    if swLowerCase not in finalStopWordList:
        finalStopWordList.append(swLowerCase)


# reading negative words from negative-words.txt file.
with open('MasterDictionary/negative-words.txt', encoding='latin-1') as negWordFile:
    negWord = negWordFile.read()
    negWordList = negWord.split()

# reading positive words from positive-words.txt file.
with open('MasterDictionary/positive-words.txt', encoding='latin-1') as posWordFile:
    posWord = posWordFile.read()
    posWordList = posWord.split()

# merge negative and positive word lists to find extra words
negPosWordList = negWordList + posWordList
finalNegPosWordList = []
finalNegPosWordList = [word for word in negPosWordList if word not in finalNegPosWordList]


"""
                            *******************
                            | Text Processing |
                            *******************

01. The text analysis cannot be executed for some articles which are not available in the respective URLs (URL_ID as 44, 57 and 144).
02. Load the article texts (except article title) from each text file and convert all the words into lower case.
03. Prepare the list of words and remove some basic puntuations from the word list for matching with stopwords.
04. Remove stopwords from the word list from the StopWords list created in previous pahse.
05. Find the extra words not present in positive and negative word lists created in previous pahse.
06. Use NLTK sentiment analyzer to compute the sentiment classes (positive and negative) for each additional words based on the compound sentiment scores.
07. Add the extra words to the positive or negative word lists based on their category.
08. Find the Positive and Negative scores for each article based on the frequency of positive and negative word.
09. Calculate Polarity and Subjectivity scores for each articles.
10. Calculate Avg. Sentence Length and Avg. Word Count Per Sentence for each articles.
11. Find Complex Words frequency, Syllable Count for each complex words and frequency percentage of complex words for each article.
12. Calculate Fog Index from Average Sentence Length and Percentage of Complex words frequency.
13. Calculate word count after removing stopwords and punctuations using NLTK library.
14. Extract the Personal Pronouns from each articles.
15. Calculate average word length and refrom each articles.
16. Save the respective calculated values to an output CSV file.

"""

posScoreList = []
negScoreList = []
polarityScoreList = []
subjectivityScoreList = []
avgSentenceLengthList = []
avgWordCountPerSentenceList = []
complexWordCountList = []
syllableCountList = []
complexWordPercentageList = []
fogIndexList = []
revisedCleanWordCountList = []
personalPronounCountList = []
averageWordLengthList = []

sa = SentimentIntensityAnalyzer()

for keys , values in urlDict.items():
    filename = "Data/" + str(keys) + ".txt"
    with open(filename) as articleFile:
        lines = articleFile.readlines()[0:1]
        firstLine = ""
        for word in lines:
            firstLine = firstLine + '' + word
        if firstLine == "Page not found\n":
            posScoreList.append(np.nan)
            negScoreList.append(np.nan)
            polarityScoreList.append(np.nan)
            subjectivityScoreList.append(np.nan)
            avgSentenceLengthList.append(np.nan)
            avgWordCountPerSentenceList.append(np.nan)
            complexWordCountList.append(np.nan)
            syllableCountList.append(np.nan)
            complexWordPercentageList.append(np.nan)
            fogIndexList.append(np.nan)
            revisedCleanWordCountList.append(np.nan)
            personalPronounCountList.append(np.nan)
            averageWordLengthList.append(np.nan)
            print ("\nText Analysis cannot be executed for {}th article.".format(keys))
            continue
    with open(filename) as articleFile:
        print ("\nText Analysis started from {}th article ...".format(keys))
        initialWordList = []
        for line in articleFile:
            strLine = str(line)
            lowerWords = strLine.lower()  #convert the article into lower case
            words = lowerWords.split()
            initialWordList.append(words)
    
    initialWords = []
    for line in initialWordList:
        for word in line:
            initialWords.append(word)    

    # remove punctuations from article to match the words with stopwords.
    initialWords = [item.strip('“') for item in initialWords]
    initialWords = [item.strip('”') for item in initialWords]
    initialWords = [item.strip('’') for item in initialWords]
    initialWords = [item.strip('‘') for item in initialWords]
    initialWords = [item.strip(',') for item in initialWords]
    initialWords = [item.strip(';') for item in initialWords]
    initialWords = [item.strip('.') for item in initialWords]
    initialWords = [item.strip('(') for item in initialWords]
    initialWords = [item.strip(')') for item in initialWords]
    initialWords = [word for word in initialWords if word]

    # remove stopwords of the article based on the StopWords list
    wordList = [word for word in initialWords if word not in finalStopWordList]
    articleCleanedText = ""
    for word in wordList:
        articleCleanedText = articleCleanedText + " " + word
    finalWordList = []
    for token in word_tokenize(articleCleanedText):  #tokenize the article
        finalWordList.append(token)
    
    # find extra words not present in positive and negative word lists.
    extraWordList = [word for word in finalWordList if word not in finalNegPosWordList]
    
    # add extra words to the positive and negative word lists based on their sentiment scores.
    if len(extraWordList) >= 1:
        for word in extraWordList:
            wordSentimentPolarity = sa.polarity_scores(word)["compound"]
            if wordSentimentPolarity >= 0.5:
                posWordList.append(word)
            else:
                negWordList.append(word)
    
    # find the positive and negative scores for each articles.
    posScore = 0
    negScore = 0
    for word in finalWordList:
        if word in posWordList:
            posScore = posScore + 1
        else:
            negScore = negScore - 1
    negScore = negScore * (-1)  # multiply the negative score with -1 to get a positive number
    posScoreList.append(posScore)
    negScoreList.append(negScore)
    
    # calculate polarity score from negative and positive scores.
    polarityScore = (posScore - negScore) / ((posScore + negScore) + 0.000001)
    polarityScoreList.append(polarityScore)
    
    # calculate subjectivity score from negative score, positive score and total number of words after cleaning.
    finalWordCount = len(finalWordList)
    subjectivityScore = (posScore + negScore) / ((finalWordCount) + 0.000001)
    subjectivityScoreList.append(subjectivityScore)

    # calculate number of sentences and find average word count per sentences from the article.
    sentenceList = sent_tokenize(strLine)
    sentenceCount = len(sentenceList)
    avgSentenceLength = avgWordCountPerSentence = finalWordCount / sentenceCount
    avgSentenceLengthList.append(avgSentenceLength)
    avgWordCountPerSentenceList.append(avgWordCountPerSentence)
    
    # calculate complex word frquency, syllable count per complex word, percentage of complex word frequency in the article
    d = cmudict.dict()
    complexWordCount = 0
    syllableCountDict = {}
    for word in finalWordList:
        try:
            syllableCount = [len(list(y for y in x if y[-1].isdigit())) for x in d[word.lower()]][0]
        except KeyError:
            syllableCount = syllapy.count(word)  #if word not found in cmudict
        if syllableCount > 1:
            complexWordCount += 1
            syllableCountDict[word] = syllableCount
    complexWordCountList.append(complexWordCount)
    syllableCountList.append(syllableCountDict)
    complexWordPercentage = complexWordCount / finalWordCount
    complexWordPercentageList.append(complexWordPercentage)

    # calculate Fog Index from Average Sentence Length and Percentage of Complex words.
    fogIndex = 0.4 * (avgWordCountPerSentence + complexWordPercentage)
    fogIndexList.append(fogIndex)

    # calculate word count after removing stopwords and punctuations using NLTK library.
    stop_words = set(stopwords.words('english'))
    revisedCleanWords = [word for word in finalWordList if not word in stop_words]  #remove stopwords from article using NLTK package
    revisedCleanWords = list(filter(lambda token: token not in string.punctuation, revisedCleanWords))
    revisedCleanWords = list(filter(lambda token: token != '’', revisedCleanWords))
    revisedCleanWords = list(filter(lambda token: token != '‘', revisedCleanWords))
    revisedCleanWords = list(filter(lambda token: token != '–', revisedCleanWords))
    revisedCleanWordCount = len(revisedCleanWords)
    revisedCleanWordCountList.append(revisedCleanWordCount)

    # find the personal pronouns from the article.
    revisedArticleCleanedText = ""
    for word in revisedCleanWords:
        revisedArticleCleanedText = revisedArticleCleanedText + " " + word
    pronounRegex = re.compile(r'\b(I|we|my|ours|(?-i:us))\b',re.I)
    personalPronoun = pronounRegex.findall(revisedArticleCleanedText)
    personalPronounCount = len(personalPronoun)
    personalPronounCountList.append(personalPronounCount)

    # calculate average word length from the article.
    words = revisedArticleCleanedText.split()
    averageWordLength = sum(len(word) for word in words) / len(words)
    averageWordLengthList.append(averageWordLength)
    print ("Text Analysis is completed from {}th article.".format(keys))

print ("\nText Analysis is completed successfully.")

# load the output file format.
output_data = pd.read_csv("Output Data Structure.csv")

# save the calculated values for each article.
output_data['POSITIVE SCORE'] = posScoreList
output_data['NEGATIVE SCORE'] = negScoreList
output_data['POLARITY SCORE'] = polarityScoreList
output_data['SUBJECTIVITY SCORE'] = subjectivityScoreList
output_data['AVG SENTENCE LENGTH'] = avgSentenceLengthList
output_data['PERCENTAGE OF COMPLEX WORDS'] = complexWordPercentageList
output_data['FOG INDEX'] = fogIndexList
output_data['AVG NUMBER OF WORDS PER SENTENCE'] = avgWordCountPerSentenceList
output_data['COMPLEX WORD COUNT'] = complexWordCountList
output_data['WORD COUNT'] = revisedCleanWordCountList
output_data['SYLLABLE PER WORD'] = syllableCountList
output_data['PERSONAL PRONOUNS'] = personalPronounCountList
output_data['AVG WORD LENGTH'] = averageWordLengthList

# save the output dataframe to csv file.
output_data.to_csv("Final Outout.csv", index = False)
print ("\nThe final output file is saved to the directory.")

# End of code.