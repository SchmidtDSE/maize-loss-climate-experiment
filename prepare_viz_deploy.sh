cd paper/viz
bash script/prepare_deploy.sh
zip deploy.zip -r deploy
cd ../..
cp paper/viz/deploy.zip deploy.zip