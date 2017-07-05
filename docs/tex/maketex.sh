#!/bin/bash

# Trasforms a static web site in a PDF document

if [[ `basename \`pwd\`` != tex ]]; then
	echo "You must be in the tex/ directory to invoke this script"
	exit 1
fi

TARGET=$1
if [[ $TARGET == '' ]]; then
	TARGET='document'
fi

echo "\documentclass[a4paper]{article}" > ${TARGET}.tex
echo "\usepackage[T1]{fontenc}" >> ${TARGET}.tex
echo "\usepackage[utf8]{inputenc}" >> ${TARGET}.tex
echo "\usepackage{microtype}" >> ${TARGET}.tex
echo "\usepackage{booktabs}" >> ${TARGET}.tex
echo "\usepackage{hyperref}" >> ${TARGET}.tex
echo "\usepackage{csquotes}" >> ${TARGET}.tex
echo "\setcounter{section}{-1}" >> ${TARGET}.tex
echo "\title{$TARGET}" >> ${TARGET}.tex
echo "\author{Giorgio Giuffrè}" >> ${TARGET}.tex
echo "\date{}" >> ${TARGET}.tex
echo "" >> ${TARGET}.tex
echo "\begin{document}" >> ${TARGET}.tex
echo "\maketitle" >> ${TARGET}.tex
echo "\tableofcontents" >> ${TARGET}.tex
echo "\newpage" >> ${TARGET}.tex

echo "Querying index.html..."
xsltproc --html get_index.xsl ../index.html | grep html > _${TARGET}_index

echo "Inserting the abstract..."
printf "\n\n" >> ${TARGET}.tex
cat abstract.tex >> ${TARGET}.tex

echo "Converting XHTML to LaTeX..."
while read f; do
	echo $f
	f="../$f"
	name=`basename $f`
	name=${name%.*}
	./texents.sh $f > _${TARGET}_entities_tmp
	printf "\n\n" >> ${TARGET}.tex
	xsltproc --html html_tex.xsl _${TARGET}_entities_tmp >> ${TARGET}.tex
	rm -f _${TARGET}_entities_tmp
done < _${TARGET}_index
rm -f _${TARGET}_index

if [ -f biblio.tex ]; then
	echo "Inserting the bibliography..."
	printf "\n" >> ${TARGET}.tex
	cat biblio.tex >> ${TARGET}.tex
fi

printf "\n\\\end{document}\n" >> ${TARGET}.tex

echo "Generating the PDF document..."
# the command is invoked twice, for the cross-references:
pdflatex -halt-on-error ${TARGET}.tex > _${TARGET}_log rm -f _${TARGET}_log && pdflatex -halt-on-error ${TARGET}.tex > _${TARGET}_log && rm -f _${TARGET}_log && echo "File generated"

# clean temporary files:
rm -f *.log *.toc *.aux *.out _${TARGET}*.tex
