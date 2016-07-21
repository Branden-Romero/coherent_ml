from datasets import twenty_newsgroups
from datasets import npmi

data = twenty_newsgroups.TwentyNewsgroups()
npmi.npmi_dict(data.X,"/lustre/janus_scratch/brro5352/20_ng_npmi_dict.pkl")
