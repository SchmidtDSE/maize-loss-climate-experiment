cd viz
bash render_images.sh
cd ..

python3 fill_template.py ./paper.md ./outputs/stats.json ./paper_filled.md

pandoc -o paper_filled.pdf --citeproc --number-sections --template=default.tex paper_filled.md
pandoc -o paper_filled.tex --citeproc --number-sections --template=default.tex paper_filled.md
pandoc -o paper_filled.docx --citeproc --number-sections paper_filled.md

rm -r arxiv
mkdir arxiv

cp paper_filled.* arxiv
cp -r img arxiv/img

cd arxiv
zip arxiv.zip *.*
cd ..
mv arxiv/arxiv.zip arxiv.zip

rm paper_filled.*
