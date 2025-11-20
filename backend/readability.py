import textstat

def readability_score(text):
    return textstat.flesch_kincaid_grade(text)
