import sys, os

ipynbfile=sys.argv[1]
fileprefix = ".".join(ipynbfile.split('.')[:-1])
os.system("mv {}.md {}.imd".format(fileprefix, fileprefix))
