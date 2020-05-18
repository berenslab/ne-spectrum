exec >&2

redo-ifchange $2.scaled.mp4

python jnb_msc/misc/telegram_send.py $2.scaled.mp4
