# -*- mode: sh -*-
exec >&2
redo-ifchange "$(dirname $2)/default.runalgo"

prepped_target="$(dirname $2)/prepped/$(basename $2)"
prev_target="$(dirname $(dirname $2))/$(basename $2)"

if [ -f $prepped_target ]; then
    ln $prepped_target $3
elif [ ! -f "$(dirname $2)/data.root" ]; then
    redo-ifchange $prev_target
    ln $prev_target $3
else
    printf "%s couldn't be created recursively\n" $2 1>&2
    exit 1
fi
