### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

Worked mostly with Anaconda base installation and installed packages as necessary.  pip install -r requirements.txt  

## Project Motivation<a name="motivation"></a>

For this project, I was interested in making article recommendations for IBM Watson Studio platform:

1. Exploratory Data Analysis to get a better understanding of the data and uncover any trends
2. What are most popular articles on the platform?
3. How can we make personalized recommendation to better engage the user?
4. What about cold start?  New users or article without any history.


## File Descriptions <a name="files"></a> 
There is 1 notebook available here to showcase work related to the above questions.  Notebook is exploratory in searching through the data pertaining to the questions showcased by the notebook title.  Markdown cells were used to assist in walking through the thought process for individual steps.

There is an additional `.py` file that is used for testing.

## Recommendation System
### Rank Based Recommendations
This is the first approach.  Simply recommend the most popular articles to all users.  In this case, it would be the most clicked articles. This approach is okay as it is not personalized and most likely will not engage the user.

### User-User Based Collaborative Filtering
In order to build better recommendations for the users of IBM's platform, we could look at users that are similar in terms of the items they have interacted with. These items could then be recommended to the similar users. This would be a step in the right direction towards more personal recommendations for the users

### Content Based Recommendations
Given the amount of content available for each article, there are a number of different ways in which someone might choose to implement a content based recommendations system. Use NLP to develop a content based recommendation system. 

### Matrix Factorization
ML approach to building recommendations.  Learn from user-item interactions to make predictions on articles a user may be interested in.


## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Please give credit to author and feel free to use the code here as you would like!

