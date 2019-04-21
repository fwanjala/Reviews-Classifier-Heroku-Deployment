I attended the live streams of [Kaggle CareerCon 2019](https://www.kaggle.com/careercon2019). [Rachael Tatman](https://twitter.com/rctatman) of Kaggle took sessions of wrapping machine learning models into REST API endpoints and then deploying them on Heroku and GCP's AppEngine. 

I wanted to try the deployment part as I am aware of how to wrap a machine learning model as a REST API endpoint. Decided to start with something a bit complicated intentionally but the end result was very satisfactory to me. I used the Zomato reviews' data from [this repository](https://github.com/Nilabhra/kolkata_nlp_workshop_2019) and built a sentiment classifier out of it using TF 2.0. I then built a `sklearn` pipeline including the `tf-keras` model (had to wrap it in a `sklearn` compatible model class) and a `CountVectorizer`. 

The simple API documentation is available [here](https://github.com/sayakpaul/Reviews-Classifier-Heroku-Deployment/blob/master/openapi.yaml). You can preview this on the Swagger Editor which should look like: 

![](https://i.ibb.co/zmV7t7f/Screen-Shot-2019-04-21-at-2-24-41-PM.png)

![](https://i.ibb.co/QfzWdyB/Screen-Shot-2019-04-21-at-2-24-57-PM.png)

Here's how to test the API endpoint in Postman (it is currently deployed on Heroku): 

![](https://i.ibb.co/xmsZyxC/Screen-Shot-2019-04-21-at-11-54-59-PM.png)
