# -*- mode: sh -*-
redo-ifchange $2.mp4

# from
# https://superuser.com/questions/624563/how-to-resize-a-video-to-make-it-smaller-with-ffmpeg#624564
ffmpeg -y -i $2.mp4 -filter:v scale="1920:trunc(ow/a/2)*2" -c:a copy -f mp4 $3
