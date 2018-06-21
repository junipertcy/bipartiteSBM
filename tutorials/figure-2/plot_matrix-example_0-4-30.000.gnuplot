# this code requires gnuplot-colorbrewer
# https://github.com/aschn/gnuplot-colorbrewer
# Please see also: http://www.gnuplotting.org/color-maps-from-colorbrewer/
set loadpath '~/Workspace/research-tools/gnuplot-colorbrewer/diverging' \
    '~/Workspace/research-tools/gnuplot-colorbrewer/qualitative' \
    '~/Workspace/research-tools/gnuplot-colorbrewer/sequential'

set size ratio 1
#set palette gray negative
# grid lines
set x2tics 1 format '' scale 0,0.001
set y2tics 1 format '' scale 0,0.001
set mx2tics 2
set my2tics 2

# labeling
set xtics 20 nomirror
set ytics 20 nomirror
set border lw 1.5
set colorbox noborder
set cbtics 1 nomirror border

set grid front mx2tics my2tics lw 1.5 lt -1 lc rgb 'white'

set xrange[-0.5:7.5]
set yrange[-0.5:5.5]
set x2range[-0.5:6.5]
set y2range[-0.5:5.5]

load 'Blues.plt'
#set palette positive

set palette defined (0 "#eff3ff", 0.8 "#eff3ff", 1.6 "#bdd7e7", 2.4 "#6baed6", 3.2 "#08306b")

plot "example_0-4-30.000.txt" matrix w image notitle ,\
    '' matrix using 1:2:(sprintf('%.3f', $3)) with labels font ',16' notitle

set autoscale fix
set link x
set link y
pause -1 "is it okay?"
