SHELL:=/bin/bash
OO=out

VPATH:=images/:exp-results/

$(OO)/main.pdf: main_filtered.bib 


# Nice way for optional dependencies
maybe_shared_bib=$(wildcard /z/home/dhiman/wrk/group-bib/shared.bib)

main_filtered.bib: main.bib $(maybe_shared_bib)
	aux2bib $(OO)/main.aux > $@

include pdflatex.mk
