for f in $(ls input_images); do
  echo input_images/$f > config/$f.txt
  echo output_images/$f >> config/$f.txt
  echo "32 22" >> config/$f.txt
  echo 32 >> config/$f.txt
done;
