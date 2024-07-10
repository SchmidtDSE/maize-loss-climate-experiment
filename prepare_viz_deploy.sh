cd paper/viz
bash script/prepare_deploy.sh
cd ../..
cp -r paper/viz/deploy deploy
zip deploy.zip -r deploy