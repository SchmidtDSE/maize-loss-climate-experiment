cd viz
bash update_data.sh
bash render_images.sh
cd ..

python3 fill_template.py ./paper.md ./outputs/stats.json ./paper_filled.md
python3 fill_template.py ./supplemental.md ./outputs/stats.json ./supplemental_filled.md

pandoc -o paper_filled.pdf --citeproc --number-sections --filter pandoc-tablenos --filter pandoc-fignos --filter pandoc-eqnos --template=default.tex paper_filled.md
pandoc -o paper_filled.tex --citeproc --number-sections --filter pandoc-tablenos --filter pandoc-fignos --filter pandoc-eqnos --template=default.tex paper_filled.md
pandoc -o paper_filled.docx --citeproc --number-sections --filter pandoc-tablenos --filter pandoc-fignos --filter pandoc-eqnos paper_filled.md

pandoc -o supplemental.pdf --citeproc --number-sections --filter pandoc-tablenos --filter pandoc-fignos --filter pandoc-eqnos --template=default.tex supplemental_filled.md
pandoc -o supplemental.tex --citeproc --number-sections --filter pandoc-tablenos --filter pandoc-fignos --filter pandoc-eqnos --template=default.tex supplemental_filled.md
pandoc -o supplemental.docx --citeproc --number-sections --filter pandoc-tablenos --filter pandoc-fignos --filter pandoc-eqnos supplemental_filled.md

rm -r arxiv
mkdir arxiv

cp paper_filled.* arxiv
cp -r img arxiv/img
cp supplemental.* arxiv

cd arxiv
zip arxiv.zip *.*
cd ..

mkdir paper_rendered
cp arxiv/arxiv.zip paper_rendered/arxiv.zip
cp arxiv/arxiv.zip arxiv.zip

rm paper_filled.*
