##########################################################################################################################
# HarvardX PH125.9x Data Science Capstone Movielens Project
# author: "Sharmin Shabnam"
# date: "1/10/2020"
##########################################################################################################################title: "HarvardX Data Science Capstone Project: MovieLens"


#Suppress warnings and set digits and scientific notation limit
options(warn=-1, digits=6, scipen=10000)

#Load required libraries, install them if the library is not found
if(!require(tidyverse)) install.packages("tidyverse")
if(!require(ggplot2)) install.packages("ggplot2")
if(!require(dslabs)) install.packages("dslabs")
if(!require(knitr)) install.packages("rpart.plot")
if(!require(kableExtra)) install.packages("rpart.plot")
if(!require(lubridate)) install.packages("rpart.plot")
if(!require(formatR)) install.packages("formatR")
if(!require(data.table)) install.packages("data.table")
if(!require(caret)) install.packages("caret")

library(tidyverse)
library(ggplot2)
library(dslabs)
library(knitr)
library(kableExtra)
library(lubridate)
library(formatR)
library(data.table)
library(caret)

######################################################################
# Data Analysis - MovieLens Dataset
######################################################################

#Following data analysis is based on:
#https://rafalab.github.io/dsbook/ 

# The RMSE function that will be used in this project is:
RMSE <- function(true_ratings = NULL, predicted_ratings = NULL) {
  sqrt(mean((true_ratings - predicted_ratings)^2))
}


#################################################
# Download data, create test and training set
#################################################

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip


#Download MovieLens data file
dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

#Read ratings.dat and movie.dat file and assign column names
ratings <- fread(text = gsub("::", "\t",
                             readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId",
                               "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")),
                          "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

#Create dataframe for movies
movies <- as.data.frame(movies) %>% 
  mutate(movieId = as.numeric(levels(movieId))[movieId],
         title = as.character(title),
         genres = as.character(genres))
#Join movies and ratings dataframe
movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1000, sample.kind="Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]


#userId and movieId in validation set are 
#implemented according to training set edx
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Addition of rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

#Remove unnecessary data and clean up memory
rm(dl, ratings, movies, test_index, temp, movielens, removed)
gc()

######################################################################
# Data Exploration - MovieLens Train Dataset
######################################################################

#edx is the training Set and Validation is the test set
#Summarise Data
head(edx, 5)
summary(edx)

#Users, Movies and Genres in Database
edx %>% summarise(
  uniq_users = n_distinct(userId),
  uniq_movies = n_distinct(movieId),
  uniq_genres = n_distinct(genres))

#Mean of all the ratings
mean(edx$rating)

#Exploratory Data Analysis
#Histogram of Different Ratings
edx %>%
  ggplot(aes(rating)) + theme_classic()+
  geom_histogram(binwidth = 0.5, color = "steelblue",fill="steelblue", alpha=0.4) +
  xlab("Rating") +
  ylab("Count") +
  ggtitle("Histogram of Different Ratings") 

#Histogram of Total Number of Ratings
edx %>% 
  count(userId) %>%
  ggplot(aes(n)) + theme_classic()+
  geom_histogram(color = "tomato",fill="tomato", alpha=0.4) +
  scale_x_log10() +
  xlab("Number of Ratings") +
  ylab("Number of Users") +
  ggtitle("Histogram of Total Number of Ratings") 

#Histogram of Average Ratings
edx %>%
  group_by(userId) %>%
  summarise(mean_r = mean(rating)) %>%
  ggplot(aes(mean_r)) + theme_classic()+
  geom_histogram(color = "yellowgreen",fill="yellowgreen", alpha=0.4) +
  ggtitle("Histogram of Average Ratings") +
  xlab("Average Rating") +
  ylab("Number of Users") 

#Variation of movie ratings with time
edx %>%
  mutate(timestamp = round_date(as_datetime(timestamp), unit = "month")) %>% 
  group_by(timestamp) %>% summarize(mean_r = mean(rating)) %>% 
  ggplot(aes(timestamp, mean_r)) + theme_classic() +
  geom_point() +
  geom_smooth(color='steelblue', span = 0.5) +
  ggtitle("Variation of movie ratings with time") + 
  ylab("Average Rating") +
  xlab("Time (years)") 




######################################################################
# Naive Mean-Baseline Model
######################################################################
#Simple Prediction using Mean Rating
mu <- mean(edx$rating)
mu

#Calculate RMSE of Naive Mean-Baseline Model
rmse_naive_mean <- RMSE(validation$rating, mu)
rmse_naive_mean

#Save Results in a Data Frame
rmse_results <- data_frame(Method = "Naive Mean Baseline Model",
                           RMSE = as.numeric(rmse_naive_mean)) %>% 
                            mutate_if(is.numeric, round, digits = 3)

rm(rmse_naive_mean)
######################################################################
# Model with Movie effects
######################################################################
#Estimate movie effects, b_i
movie_avgs <- edx %>%
  group_by(movieId) %>%
  summarise(b_i = mean(rating - mu))


#Make predictions using Movie Effects
predicted_ratings <- mu + validation %>%
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)


#Calculate RMSE of Model with Movie effects
rmse_movie_effects <- RMSE(predicted_ratings, validation$rating)
rmse_movie_effects

#Save Results in a Data Frame
rmse_results <- bind_rows(rmse_results, 
                          data_frame(Method="Movie Effects Model",
                                     RMSE = rmse_movie_effects))

# Clean up memory
rm(predicted_ratings)
gc()


######################################################################
# Model with Both Movie and User Effects
######################################################################
#Estimate User Effects
user_avgs <- edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i)) 

#Make predictions using Movie and User Effects
predicted_ratings_bi_bu <- validation %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>% 
  pull(pred) 

#Calculate RMSE of Model with Movie and User effects
rmse_movie_user_effects <- RMSE(predicted_ratings_bi_bu,
                                validation$rating)

#Save Results in a Data Frame
rmse_results <- rbind(rmse_results,
                      data_frame(Method = "Movie and User effect",
                                 RMSE = rmse_movie_user_effects)) 


# Clean up memory
rm(predicted_ratings_bi_bu, rmse_movie_user_effects)
gc()

######################################################################
# Model with Both Movie,User and Time Effects
######################################################################
edx_time <- edx %>%
  mutate(timestamp = round_date(as_datetime(timestamp), unit = "week"))
head(edx_time)
validation_time <- validation %>%
  mutate(timestamp = round_date(as_datetime(timestamp), unit = "week"))
head(validation_time)

time_avgs <- edx_time %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  group_by(timestamp) %>%
  summarize(b_t = mean(rating - mu - b_i - b_u)) 

predicted_ratings_bi_bu_bt <- validation_time %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(time_avgs, by='timestamp') %>%
  mutate(pred = mu + b_i + b_u + b_t) %>%
  pull(pred) 
rmse_movie_user_time_effects <- RMSE(predicted_ratings_bi_bu_bt,
                              validation_time$rating) 
rmse_movie_user_time_effects
rmse_results <- rbind(rmse_results,
                    data_frame(Method = "Regression model using movie-user-date effect",
                               RMSE = rmse_movie_user_time_effects)) 

# Clean up memory
rm(predicted_ratings_bi_bu_bt, rmse_movie_user_time_effects)
gc()

######################################################################
# Regularization model with Movie and User
######################################################################

#Define lambda values for cross-validation purpose
lambdas <- seq(0, 10, 1)

#Calculate RMSEs for each lambda
rmses <- sapply(lambdas, function(l){
  mu_reg <- mean(edx$rating)
  b_i_reg <- edx %>% 
    group_by(movieId) %>%
    summarize(b_i_reg = sum(rating - mu_reg)/(n()+l))
  b_u_reg <- edx %>% 
    left_join(b_i_reg, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u_reg = sum(rating - b_i_reg - mu_reg)/(n()+l))
  predicted_ratings_b_i_u <- 
    validation %>% 
    left_join(b_i_reg, by = "movieId") %>%
    left_join(b_u_reg, by = "userId") %>%
    mutate(pred = mu_reg + b_i_reg + b_u_reg) %>%
    pull(pred)
  return(RMSE(validation$rating,predicted_ratings_b_i_u))
})

#Plot RMSE vs lambda
ggplot(mapping = aes(x = lambdas, y = rmses)) + theme_light()+
  xlab("lambda") +
  ylab("RMSE") +
  geom_point()

#Extract lambda that minimizes RMSE
lambda_min <- lambdas[which.min(rmses)]
lambda_min

#Extract minimized RMSE and save it into a dataframe
rmse_reg_movie_user_effects <- min(rmses)
rmse_results <- rbind(rmse_results,
                      data_frame(Method = "Regularization model using movie-user effect",
                                 RMSE = rmse_reg_movie_user_effects)) 

# Clean up memory
rm(lambdas,rmses)
gc()

######################################################################
# Regularization model with Movie,User and Time
######################################################################

#Define lambda values for cross-validation purpose
lambdas <- seq(0, 10, 1)

#Calculate RMSEs for each lambda
rmses <- sapply(lambdas, function(l){
  mu_reg <- mean(edx_time$rating)
  b_i_reg <- edx_time %>% 
    group_by(movieId) %>%
    summarize(b_i_reg = sum(rating - mu_reg)/(n()+l))
  b_i_reg
  b_u_reg <- edx_time %>% 
    left_join(b_i_reg, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u_reg = sum(rating - b_i_reg - mu_reg)/(n()+l))
  b_u_reg
  b_t_reg <- edx_time %>%
    left_join(b_i_reg, by="movieId") %>%
    left_join(b_u_reg, by="userId") %>%
    group_by(timestamp) %>%
    summarize(b_t_reg = sum(rating - b_i_reg - b_u_reg - mu_reg)/(n()+l))
  b_t_reg
  predicted_ratings <- 
    validation_time %>%
    left_join(b_i_reg, by = "movieId") %>%
    left_join(b_u_reg, by = "userId") %>%
    left_join(b_t_reg, by = "timestamp") %>%
    mutate(pred = mu_reg + b_i_reg + b_u_reg + b_t_reg) %>%
    pull(pred)
  return(RMSE(validation$rating,predicted_ratings))
})

#Plot RMSE vs lambda
ggplot(mapping = aes(x = lambdas, y = rmses)) + theme_light()+
  xlab("Lambda") +
  ylab("RMSE") +
  geom_point()

#Extract lambda that minimizes RMSE
lambda_min <- lambdas[which.min(rmses)]
lambda_min


#Extract minimized RMSE 
rmse_reg_movie_user_effects <- min(rmses)
rmse_results <- rbind(rmse_results,
                      data_frame(Method = "Regularization model using movie,user and time effect",
                                 RMSE = rmse_reg_movie_user_effects)) 

# Clean up memory
rm(lambdas,rmses)
gc()

######################################################################
# Summary of results
######################################################################
rmse_results %>% knitr::kable(caption = "RMSEs")



