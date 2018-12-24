#!/bin/bash
cat << _end_ | gnuplot
set terminal postscript eps color
set output 'Descriptivo.eps'
set key right bottom
set xlabel "numero de categorias"
set ylabel "ratio de reconocimiento (%)"
plot  "Fsiv.dat" using 1:2 t " SURF" w l, "Fsiv.dat" using 1:3 t "SIFT" w l, "Fsiv.dat" using 1:4 t "SIFT DENSO" w l
_end_
