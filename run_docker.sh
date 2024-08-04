mkdir workspace

docker build -t dse/maize-climate-adapt .

docker run -it -d \
  --name ag_pipeline_run \
  --mount type=bind,source="$(pwd)"/workspace,target=/pipeline/workspace \
  dse/maize-climate-adapt bash

docker exec -it \
  -e USE_AWS=$USE_AWS \
  -e SOURCE_DATA_LOC=$SOURCE_DATA_LOC \
  -e AWS_ACCESS_KEY=$AWS_ACCESS_KEY \
  -e AWS_ACCESS_SECRET=$AWS_ACCESS_SECRET \
  ag_pipeline_run bash run.sh

docker stop ag_pipeline_run
