python3 analysis/update.py -d data/ -i imgs/ -v -th 60
ffmpeg -framerate 2 -i imgs/wc_%04d.png -r 60 -vcodec copy -acodec copy -vcodec libx264 -pix_fmt yuv420p -y imgs/wc.mp4
ffmpeg -i imgs/wc.mp4 -filter_complex "fps=2,scale=-1:640,setsar=1,palettegen" -y imgs/palette.png
ffmpeg -i imgs/wc.mp4 -i imgs/palette.png -filter_complex "[0]fps=2,scale=-1:640,setsar=1[x];[x][1:v]paletteuse" -y imgs/wc.gif
