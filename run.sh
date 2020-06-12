for f in config/*.txt; do
  python3 pyxeled.py < $f 
done
