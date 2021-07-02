# -*- mode: sh -*-

# https://github.com/dylanaraps/pure-sh-bible#strip-pattern-from-end-of-string
rstrip() {
    # Usage: rstrip "string" "pattern"
    printf '%s\n' "${1%%$2}"
}

FILE=$(rstrip $2 ".png")
redo-ifchange "$FILE.pdf"

# leaving out a root name (second arg) when using `pdftoppm`
# apparently writes the output to stdout.  This is not documented
# anywhere in the man page as far as I can see.
pdftoppm -r 300 -png -singlefile "$FILE.pdf" > $3
