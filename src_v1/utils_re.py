import re



expressions =[
    r"[A-Z]",
    r"[0-9]{10,12}",
    r"[0-9]+",
    r"(?:[0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F])+",
    r"([a-zA-Z0-9][a-zA-Z0-9_.+-]*@[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)+)",
    r"(?:ftp|http|https)?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
    r"[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)+",
    r"(?:[0-9]+\.)+[0-9]+",
    r"(?:[0-9]+\.){3}[0-9]+",
    r"(?:[0-9]+\.){5}[0-9]+",
    r"(?:AS|as)[0-9]+",
    r"(?:CVE|exploit)",
    r"(?:HK|hk)[a-zA-Z0-9/]+"]
def get_pattern(token):
    return [len(re.findall(expression,token)) for expression in expressions]