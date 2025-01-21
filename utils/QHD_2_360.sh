for file in videos/QHD/*.mp4; do
    clear
    filename=$(basename "$file")
    ffmpeg -i "$file" -c:v h264_nvenc -profile:v high -preset p7 -cq:v 17 -r 60 -s 640x360 "videos/360p/$filename" -y
done