# Project 3 Write-up

For this project, I used geographic data from the GADM database and rasters from the Worldpop database to investigate how effecting machine learning techniques can be at predicting population values in Cambodia. From GADM, I obtained spatial data about Cambodia and its subdivisions. Worldpop had many rasters including population counts from 2020, and other rasters from 2015 including topography, slope, and distance to many topographical characteristics such as artificial surfaces, water, cultivated area, etc. 

Using R-studio, I performed both a logistic regression and a random forest analysis to predict values for population and its distribution in Cambodia. In the preprocessing of this data, the rasters were stacked together and cropped to fit within the boundaries of Cambodia and to only include the necessary variables.  Both the logistic regression and random forest models overestimated the population of Cambodia with values of 19,630,861 and 19,630,805 respectively, whereas the actual value was 19,626,880. Part of this difference could have occurred because the raster for Cambodian population was a slightly smaller size than the generated population rasters. I have no explanation as to why there was a change in raster size. 

For validation, I did what was given in the script – finding the sum of the errors for both models, as well as mean squared error. In the Logistic Regression model, the sum of errors is -145.313, the sum of the absolute errors is 13,875,149 and the MSE is 8.594136. In the Random Forest model, the sum of errors is -201.304, the sum of absolute errors is 13,827,214 and the MSE is 8.595975. 

-	Logistic Regression resulting population difference raster:
 ![diff_sum logistic regression](https://user-images.githubusercontent.com/78227378/117354994-02be3980-ae80-11eb-8cfa-131f94988b52.png)
 
-	Random Forest resulting population difference raster:
 ![Diff_sums random forest](https://user-images.githubusercontent.com/78227378/117355003-05209380-ae80-11eb-9bcb-ab568ef8cbf1.png)

-	A representation of the actual population distribution:
 ![pop_sum logisitic regression](https://user-images.githubusercontent.com/78227378/117355018-094cb100-ae80-11eb-8133-653067804ce7.png)
 
Since the estimates of both models were very similar, the resulting rasters showing the distribution of the population sums and difference were also similar and can be found below. Some of the reasons for variation could be the weights of the associated variables. But, there is a lake in the middle of Cambodia, marked by a big empty oval-ish shape towards the middle of the country. There are also many clearly defined paths along the right half of the country, which turned out to be rivers. At the confluence of the three largest rivers is where the major hub is with a majority of the population. The actual city appears to be underestimated because it is a very dense population that isn’t spread out very much. Interestingly, the immediate surrounding area was overestimated which could be do the excess light pollution from the city. A final feature that is worth mentioning is the speckled area in the top-left section of Cambodia. This area is not covered by many trees or greenery, so there are patches of underestimation. Strangely, it appears that the regions of Cambodia that had the most variation were the non-mountainous regions, ie where the rivers are and the top left area of the map. 
When looking at the node purity graph of the random forest model, it can be shown that 4 of the rasters from Worldpop had a very high weight compared to the others. At the highest was the night time lights, followed by distance from cultivated areas, artificial surfaces and woody tree areas respectively. This is important to the above interpretation because it helps to identify which of the variables were deemed important by the model and would be more likely to affect the output.   

-	Associated Variable Weights from Random Forest Model:
![Cambodia variables](https://user-images.githubusercontent.com/78227378/117354979-fd60ef00-ae7f-11eb-8d10-349f9d83c090.png)

In conclusion, both models were fairly accurate in predicting the population and distribution of the population in Cambodia, with the Random Forest model being slightly more accurate. I am curious by how much the output would be different if the variables with the lowest weights were excluded from the model, since it appears that there are only four strong variables. The terrain of Cambodia is variable, containing mountains, rivers, and lakes, which I thought would have provided a larger error from the models. This shows that the variables included in the making of the models were highly appropriate, even though the weights varied a lot. It was interesting to see the discrepancies in predictions right around the major city, where the model was unable to predict the distribution of the population, but still resulted in a very similar overall account of the population.


#### Code:
``` R
rm(list=ls(all=TRUE))

# install.packages("raster", dependencies = TRUE)
# install.packages("sf", dependencies = TRUE)
# install.packages("tidyverse", dependencies = TRUE)
# install.packages("exactextractr")
# install.packages("tidymodels")
# install.packages("vip")
# install.packages("randomForest", dependencies = TRUE)
# install.packages("rgl", dependencies = TRUE)

library(sf)
library(raster)
library(tidyverse)
library(exactextractr)
library(tidymodels)
library(vip)
library(randomForest)

setwd('KHM_data')
### Import Administrative Boundaries ###

khm_adm0  <- read_sf("gadm36_KHM_0.shp")
khm_adm1  <- read_sf("gadm36_KHM_1.shp") 
khm_adm2  <- read_sf("gadm36_KHM_2.shp")
khm_adm3  <- read_sf("gadm36_KHM_3.shp")

### Import Land Use Land Cover, Night Time Lights and Settlements Covariates ###

f <- list.files(pattern="khm_", recursive=TRUE)
lulc <- stack(lapply(f, function(i) raster(i, band=1)))

names(lulc) <- c("water", "dst011", "dst040", "dst130", "dst140","dst150", "dst160","dst190", "dst200", "pop19", "slope", "topo", "ntl")

lulc <- crop(lulc, khm_adm0)
lulc <- mask(lulc, khm_adm0)

lulc_adm2 <- exact_extract(lulc, khm_adm2, fun=c('sum', 'mean'))

#########################
### Linear Regression ###
#########################

data <- lulc_adm2[ , 1:13]

data_split <- initial_split(data, prop = 4/5)

data_train <- training(data_split)
data_test <- testing(data_split)

data_recipe <- 
  recipe(sum.pop19 ~ ., data = data_train)

preprocess <- prep(data_recipe)

lr_model <- 
  linear_reg()%>%
  set_engine("lm") %>%
  set_mode("regression")

lr_workflow <- workflow() %>%
  add_recipe(data_recipe) %>%
  add_model(lr_model)

final_model <- fit(lr_workflow, data)

rstr_to_df <- as.data.frame(lulc, xy = TRUE)

names(rstr_to_df) <- c("x", "y", "sum.water", "sum.dst011", "sum.dst040", "sum.dst130", "sum.dst140", 
                       "sum.dst150", "sum.dst160", "sum.dst190", "sum.dst200", "sum.topo", 
                       "sum.slope", "sum.ntl", "sum.pop20")

preds <- predict(final_model, new_data = rstr_to_df)

coords_preds <- cbind.data.frame(rstr_to_df[ ,1:2], preds)

predicted_values_sums <- rasterFromXYZ(coords_preds)

ttls <- exact_extract(predicted_values_sums, khm_adm2, fun=c('sum'))

khm_adm2 <- khm_adm2 %>%
  add_column(preds_sums = ttls)

predicted_totals_sums <- rasterize(khm_adm2, predicted_values_sums, field = "preds_sums")

gridcell_proportions_sums  <- predicted_values_sums / predicted_totals_sums

cellStats(gridcell_proportions_sums, sum)

khm_pop19 <- raster("khm_ppp_2019.tif")
#khm_pop20 <- raster("khm_ppp_2020.tif")

khm_adm2_pop19 <- exact_extract(khm_pop19, khm_adm2, fun=c('sum'))
khm_adm2 <- khm_adm2 %>%
  add_column(pop19 = khm_adm2_pop19)

population_adm2 <- rasterize(khm_adm2, predicted_values_sums, field = "pop19")

population_sums <- gridcell_proportions_sums * population_adm2

cellStats(population_sums, sum)

sum(khm_adm2$pop19)

diff_sums <- population_sums - khm_pop19

plot(population_sums)
plot(diff_sums)
rasterVis::plot3D(diff_sums)
cellStats(abs(diff_sums), sum)

diff_sq <- diff_sums * diff_sums
sums= cellStats(diff_sq, sum)
mse = (sums / length(diff_sums))
mse
#####################
### Random Forest ###
#####################

model <- randomForest(sum.pop19 ~ ., data = data)

print(model)
plot(model)
varImpPlot(model)

names(lulc) <- c("sum.water", "sum.dst011", "sum.dst040", "sum.dst130", "sum.dst140", 
                 "sum.dst150", "sum.dst160", "sum.dst190", "sum.dst200", "sum.pop20", 
                 "sum.slope", "sum.topo", "sum.ntl")

predicted_values_sums <- raster::predict(lulc, model, type="response", progress="window")

ttls <- exact_extract(predicted_values_sums, khm_adm2, fun=c('sum'))

khm_adm2 <- khm_adm2 %>%
  add_column(rf_preds_sums = ttls)

predicted_totals_sums <- rasterize(khm_adm2, predicted_values_sums, field = "rf_preds_sums")

gridcell_proportions_sums  <- predicted_values_sums / predicted_totals_sums

cellStats(gridcell_proportions_sums, sum)

population_adm2 <- rasterize(khm_adm2, predicted_values_sums, field = "pop20")

population_sums <- gridcell_proportions_sums * population_adm2

cellStats(population_sums, sum)

sum(khm_adm2$pop20)

diff_sums <- population_sums - khm_pop20

plot(population_sums)
plot(diff_sums)
rasterVis::plot3D(diff_sums)
cellStats(abs(diff_sums), sum)

diff_sq <- diff_sums * diff_sums
sums= cellStats(diff_sq, sum)
mse = (sums / length(diff_sums))
mse

```
