redo-always
redo-ifchange $2
EXITCODE=$?
if [ $EXITCODE -eq 0 ]; then
    ICON=finish
else
    ICON=error
fi
DONETIME=$(date +'%H:%M (%d. %b %y)')
notify-send --icon $ICON "$2" ""
