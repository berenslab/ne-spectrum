# -*- mode: sh -*-
exec >&2
redo-ifchange "$(dirname $2)/default.runalgo"

prepped="$(dirname $2)/prepped/grad"
prev_target="$(dirname $(dirname $2))/$(basename $2)"

if [ ! -f "$(dirname $2)/data.root" ]; then
    redo-ifchange $prev_target.glist
    cat "$prev_target.flist" >> $3
fi

if [ -d $prepped ]; then
    i=0
    while [ -f "$prepped/$i.npy" ]; do
          printf "%s\0" "$prepped/$i.npy"
          i=$((i+1))
    done | xargs -I '{}' -0 readlink -f -- {} >> $3
fi
