mkdir paper_rendered

docker build -t dse/maize-climate-adapt-paper .

docker run -it -d \
  --name ag_paper_build \
  --mount type=bind,source="$(pwd)"/paper_rendered,target=/workspace/paper_rendered \
  dse/maize-climate-adapt-paper bash

docker exec -it \
  ag_paper_build bash render.sh

docker stop ag_paper_build