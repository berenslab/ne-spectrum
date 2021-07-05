exec >&2

redo-ifchange $2.png
python jnb_msc/misc/telegram_send.py $2.png
