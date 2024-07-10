[ -e deploy ] && rm -r deploy

mkdir deploy
cp *.py deploy
cp *.css deploy
cp *.html deploy
# cp *.md deploy
# cp *.txt deploy

python3 script/rename_py_files.py deploy
python3 script/update_py_paths.py deploy/index.html deploy/index.html
python3 script/update_epoch.py deploy/index.html deploy/index.html
#python3 script/update_py_paths.py deploy/presentation.html deploy/presentation.html
#python3 script/update_epoch.py deploy/presentation.html deploy/presentation.html

echo "== Prepared deploy directory. =="


echo "-- Root --"
ls

echo "-- Deploy --"
ls deploy
