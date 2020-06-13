for f in config/*.txt; do
  echo $f
  python3 pyxeled.py < $f 
done
