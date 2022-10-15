# Decision-making-model-based-on-historical-and-real-time-forex-data
#Used API to get the historical forex data of EURUSD, USDAUD, and GBPUSD from the polygon.io website, and imported more than 32k rows of data into the MongoDB database.
#Preprocessed the data in the database, built cluster model, calculated basic statistics (average and standard deviation) of clusters, and compared the accuracy of different classifier (KNN, Random Forest. Decision Tree).
#Developed trading strategies, collected real-time forex data, cleaned data, did the sanity check, and finally automatically analyzed each new data point to generate decisions (buy, sell, or do nothing).
