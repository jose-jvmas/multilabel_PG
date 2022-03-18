set terminal postscript eps enhanced color font ",18"

set output '../Results/Plots/LabelPowerset.eps'

set datafile separator ","

set ylabel "Hamming Loss"
set xlabel "Size (%)"

set bmargin at screen 0.15
set tmargin at screen 0.9
set key top center horizontal outside samplen 2
set boxwidth 0.9
set xrange [0:110]
# set yrange [0.13:0.165]
set format y "%.3f"


plot '../Results/Plots/LabelPowerset/ND.csv'  using 6:5 with linespoints pt 71 ps 1.75 dt 2 pi -1 lw 2 lc rgb "#000000" notitle,\
    '../Results/Plots/LabelPowerset/ALL.csv' using 6:5 ti 'ALL' pt 5 ps 1.5 lc rgb "#000000",\
    '../Results/Plots/LabelPowerset/MChen.csv' using 6:5 ti 'Chen' pt 13 ps 1.5 lc rgb "#0000DD",\
    '../Results/Plots/LabelPowerset/MRHC.csv' using 6:5 ti 'RHC' pt 11 ps 1.5 lc rgb "#DD00DD",\
    '../Results/Plots/LabelPowerset/MRSP1.csv' using 6:5 ti 'RSP1' pt 9 ps 1.5 lc rgb "#DD0000",\
    '../Results/Plots/LabelPowerset/MRSP2.csv' using 6:5 ti 'RSP2' pt 9 ps 1.5 lc rgb "#00DD00",\
    '../Results/Plots/LabelPowerset/MRSP3.csv' using 6:5 ti 'RSP3' pt 9 ps 1.5 lc rgb "#555555"

