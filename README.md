To test the containers executing locally prior to working on sagemaker:

```
cd container
docker build . -t sagemaker_byo
cd local_test
sh train_local.sh
sh serve_local.sh
```

Then in a new terminal cd into local_test and run ```sh predict.sh payload.csv```