import sys

filetoread = sys.argv[1]
fdtoread = open(filetoread)
fileprefix = ".".join(filetoread.split('.')[:-1])
filetowrite = fileprefix+".md"
if len(sys.argv) > 2:
    shorttitle = sys.argv[2]
else:
    shorttitle=""
title=shorttitle

buffer=""
counter=0
poundcounter=0
for line in fdtoread:
    if line[0:2]=='# ':#assume title
        poundcounter +=1
        if poundcounter==1 and counter <=10:
            print("TITLE: "+line)
            title = line.strip()[2:]
        else:
            buffer = buffer + line
    else:
        if counter !=0 and (line.find("warnings.warn")==-1 and line.find("UserWarning")==-1):
            buffer = buffer + line
    counter +=1
fdtoread.close()
if not shorttitle:
    shorttitle=title
preamble = "---\ntitle: {}\nshorttitle: {}\nnotebook: {}\nnoline: 1\n".format(title, shorttitle, fileprefix+".ipynb")
fdtowrite=open(filetowrite, "w")
fdtowrite.write(preamble+buffer)
fdtowrite.close()
