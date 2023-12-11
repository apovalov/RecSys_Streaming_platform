# RECSYS-LIVE-STREAMING-PLATFORM

What kind of recommender systems are there?

![Alt text](/img/image.png)


## Popularity based

Let's imagine that a new user comes to us, whose preferences we don't know. What should we do?
Let's recommend the most popular products!


## Content based

While recommending the most popular streamers is a great and simple solution, it doesn't take into account user interests, not everyone likes the most popular, right? Maybe someone is only watching narrowly targeted content! Then let's take into account the history of each user by stream category!

The idea of content-based is based on the attributes of a user or object. Attributes can be video topic, authors, region or other characteristics.

One of the advantages of content-based recommendations is user independence - a user does not need information about other users to make recommendations.


## Collaborative filtering
What about recommending something new to a user? Is there some algorithm that will consider interests based on other users?

Yes! This algorithm is collaborative filtering, its main idea is to generate recommendations based on data about other users with similar interests. Filtering comes in user-based and item-based.

User-based algorithms are based on finding users whose interests are as similar as possible and then recommending to one of them what the other user has tried.
Item-based recommendations look at the task from the opposite direction: find similar items and see how they have been rated before, with similarity determined based on all users!

## Matrix Factorization
In collaborative filtering, engineers are often faced with large matrix sizes that are not easy to work with. We can have hundreds of thousands of users and streamers, such large matrices can require huge computational machinery! So how do we fit such a large elephant into a refrigerator?

If we can't fit the whole elephant in the fridge, let's divide it into parts. There is a special name for these parts - latent factors, but more often they are called embeddings.

Returning to our task of recommender systems, let's remember that we have streamers and users. So, we can decompose our matrix into embeddings of users and streamers. Decomposition into smaller matrices while preserving the structure is the basic principle of matrix factorization.

Intuitively, we can visualize the matrix decomposition as in the picture below:
![Alt text](/img/image-1.png)


ALS (alternating least squares) is a popular iterative algorithm for decomposing the preference matrix into the product of two matrices: user factors (User vector) and item factors (Item vector).

The algorithm works on the principle of minimizing the mean square error. Optimization is performed one by one, first by user factors, then by item factors. Also, regularization coefficients are added to the RMS error to bypass overtraining.

![Alt text](/img/image-2.png)

### Metrics
Normalized Discounted Cumulative Gain at K is a popular metric in a ranking task that takes into account the order of items in the rendition. You can see more details in our NDCG challenge!

Mean Avarage Precision at K - also a popular metric that takes into account the order of elements in the sample, as well as takes into account the measure of relevance.


