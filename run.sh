docker run --network="host" --dns 8.8.8.8 --privileged -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v /home/daniel/exjobb:/ws -it --rm  danneengelson/exjobb