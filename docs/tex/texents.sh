#!/bin/bash

# prende in input un file contenente entità HTML
# e restituisce lo stesso file con simboli LaTeX
# al posto delle entità

if [[ $# != 1 ]]; then
	echo "Uso: texents <file>"
	exit
fi

cat $1 > _sweki_tmp_0
sed 's/&quot;/"/g' _sweki_tmp_0 > _sweki_tmp_1
sed 's/&amp;/\\&{}/g' _sweki_tmp_1 > _sweki_tmp_0
sed 's/&lt;/\\textless{}/g' _sweki_tmp_0 > _sweki_tmp_1
sed 's/&gt;/\\textgreater{}/g' _sweki_tmp_1 > _sweki_tmp_0
sed "s/&apos;/\'/g" _sweki_tmp_0 > _sweki_tmp_1
sed 's/&mdash;/---/g' _sweki_tmp_1 > _sweki_tmp_0
sed 's/&sdot;/\\cdot{}/g' _sweki_tmp_0 > _sweki_tmp_1
sed 's/%/\\%/g' _sweki_tmp_1 > _sweki_tmp_0

cat _sweki_tmp_0
rm _sweki_tmp_0
rm _sweki_tmp_1
