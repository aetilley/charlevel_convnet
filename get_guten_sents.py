import nltk
import io

for fileid in gut.fileids():
    write_file_name = "gutenberg_"+fileid[-4]+"_sents.txt"
    with io.open("guten_data/"+write_file_name, 'w', encoding = 'UTF-8') as f:
        raw_text = gut.raw(fileids = fileid)
        sent_string_list = nltk.sent_tokenize(raw_text)
        for sent in sent_string_list:
        no_newline = sent.replace("\n", "")
        f.write(no_newline+"\n")

