exec >&2
redo-ifchange "$2.png"
convert -resize 1980 "$2.png" "$3"
