for file in videos/UHD/*.mp4; do
    filename=$(basename "$file")
    ffmpeg -hwaccel cuda -i "$file" -c:v h264_nvenc -profile:v high -preset p7 -cq:v 17 -r 60 -s 1920x1080 "videos/FHD/$filename" -y
done