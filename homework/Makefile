MEXT = md
IPEXT = ipynb
FILESEXT = files

IPYSRC = $(wildcard *.$(IPEXT))
MD=$(IPYSRC:.ipynb=.md)
IMD=$(IPYSRC:.ipynb=.imd)

all: $(MD)
.SECONDARY: $(IMD)

%.md:	%.imd
	python ./_support/nbmd.py $<

%.imd:   %.ipynb
	jupyter nbconvert --to markdown --template ./_support/markdown.tpl $<
	python ./_support/convert.py $<

clean:
	rm -rf *_files
	touch *.ipynb

mdclean:
	rm -f $(MD)

imdclean:
	rm -f $(IMD)
