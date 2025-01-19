for file in videos/UHD/*.mp4; do
    clear
    filename=$(basename "$file")
    ffmpeg -hwaccel cuda -i "$file" -c:v h264_nvenc -profile:v high -preset p7 -cq:v 17 -r 60 -s 2560x1440 "videos/QHD/$filename" -y
done