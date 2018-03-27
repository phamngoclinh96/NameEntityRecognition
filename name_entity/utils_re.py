import re

def get_pattern(token,expressions):
    return [len(re.findall(expression,token)) for expression in expressions]