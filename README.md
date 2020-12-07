To develop in the container environment:

```
cd container
docker build . -t sagemaker_disaster_tweets
docker run -it -v $(pwd)/decision_trees:/opt/program sagemaker_disaster_tweets
```

To test the containers executing locally prior to working on sagemaker:

```
cd container
docker build . -t sagemaker_disaster_tweets
cd local_test
sh train_local.sh
sh serve_local.sh
```

Then in a separate new terminal cd into local_test and run ```sh predict.sh payload.csv```