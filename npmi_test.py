from datasets import twenty_newsgroups
from datasets import npmi

data = twenty_newsgroups.TwentyNewsgroups()
npmi.npmi_dict(data.X[:1000,:1000],"test.pkl")
