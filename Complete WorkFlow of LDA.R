# intall packages
install.packages(c("topicmodels", "tm", "SnowballC", "childesr", "dplyr", "magrittr", "stringr", "quanteda", "udpipe", "LDAvis", "tidytext"))


library(topicmodels)
library(tm)
library(SnowballC)
library(childesr)
library(dplyr)
library(magrittr)
library(stringr)
library(quanteda)
library(udpipe)
library(LDAvis)
library(tidytext)

# UDpipe model to subtract content words (eliminate English)
ud_model <- udpipe_download_model(language = "english")
ud_model <- udpipe_load_model(file = ud_model$file_model)

# get CHILDES corpus
texts1 <- get_utterances(corpus = "Providence")

# chunk into 250 utterance per documents
grouped_data <- texts1 %>%
  mutate(doc_id = rep(1:(n() %/% 250 + 1), each = 250, length.out = n())) %>%
  group_by(doc_id) %>%
  summarise(document = paste(stem, collapse = " ")) %>%
  ungroup()

# create a meal time rep_doc to spike the data
rep_doc <- data.frame(
  doc_id = max(grouped_data$doc_id, na.rm = TRUE) + 1,
  document = "Eat, Drink, Plate, Fork, Spoon, Knife, Cup, Glass, Napkin, Table, Chair, Food, Water, Juice, Milk, Bread, Butter, Jam, Cheese, Meat, Chicken, Beef, Fish, Vegetable, Carrot, Peas, Potato, Salad, Lettuce, Tomato, Cucumber, Rice, Pasta, Soup, Broth, Cereal, Breakfast, Lunch, Dinner, Snack, Dessert, Ice cream, Cake, Cookie, Pie, Candy, Fruit, Apple, Banana, Orange, Grape, Strawberry, Blueberry, Raspberry, Peach, Plum, Kiwi, Mango, Pineapple, Spoonful, Bite, Chew, Swallow, Sip, Pour, Cook, Bake, Fry, Boil, Steam, Roast, Grill, Taste, Delicious, Yummy, Tasty, Flavor, Sweet, Sour, Bitter, Salty, Spicy, Hot, Warm, Cold, Fresh, Frozen, Raw, Cooked, Burnt, Crispy, Soft, Hard, Tender, Juicy, Dry, Fill, Serve, Pass, Share, More, Less, Enough, Full, Hungry, Thirsty, Seconds, Tablecloth, Placemat, Forkful, Crumb, Drip, Spill, Pick, Mix, Stir, Cut, Slice, Chop, Spread, Dip, Squeeze, Mash, Peel, Shell, Whisk, Grate, Taste, Tray, Oven, Microwave, Stove, Pot, Pan, Lid, Bowl, Mixing, Kitchen, Pantry, Refrigerator, Freezer, Spice, Herb, Salt, Pepper, Sauce, Ketchup, Mustard, Mayonnaise, Vinegar, Oil, Butter, Margarine, Honey, Syrup, Season, Dish, Recipe, Menu, Leftover, Grocery, Market, Farm, Fresh, Organic, Natural, Healthy, Nutritious, Balanced, Protein, Carbohydrate, Fiber, Vitamin, Mineral, Energy, Snack, Finger, Bite-size, Nibble, Chewy, Crunchy, Gummy, Sticky, Melt, Cool, Warm, Reheat, Serve, Plate, Dish, Portion, Meal, Course, Appetizer, Main, Dessert, Cleanup, Wipe, Rinse, Scrub, Dry, Fold, Put away, Leftover, Pack, Lunchbox, Cooler, Picnic, Blanket, Basket, Drink, Beverage, Thirsty, Hydrate, Refill, Fresh, Cool, Ice, Smoothie"
)

# append it into exisiting dataset
grouped_data <- rbind(grouped_data, rep_doc)

# ensure ordered by number
grouped_data <- grouped_data %>%
  arrange(doc_id)

#annotate each document
get_content_words <- function(doc) {
  anno <- udpipe_annotate(ud_model, x = doc)
  anno_df <- as.data.frame(anno)
  content_words <- anno_df %>%
    filter(upos %in% c("NOUN", "VERB", "ADJ", "ADV")) %>%
    summarise(document = paste(lemma, collapse = " "))
  return(content_words$document)
}

grouped_content_words <- grouped_data %>%
  rowwise() %>%
  mutate(document = get_content_words(document)) %>%
  ungroup()

grouped_content_words$doc_id <- as.character(grouped_content_words$doc_id)

# build the corpus
corpus <- corpus(grouped_content_words$document, docnames = grouped_content_words$doc_id)

# tokenization and preprocessing
tokens <- tokens(corpus, what = "word",remove_punct = TRUE,remove_numbers = TRUE,remove_url = TRUE) %>%
  tokens_tolower() %>%
  tokens_remove(stopwords("english")) %>%
  tokens_remove(c("", "na")) %>%
  tokens_remove(min_nchar = 3)

# create DFM
dfm_obj <- dfm(tokens)

# try to locate an appropriate pruning parameters in a range based on the topic coherence
min_docfreq_values <- c(0.0005,0.001, 0.005, 0.01)
max_docfreq_values <- c(0.3, 0.4, 0.5, 0.6)

# Fixed number of topics for initial evaluation
initial_k <- 20

# Store results
pruning_results <- list()

# Evaluate combinations of pruning parameters
for (min_df in min_docfreq_values) {
  for (max_df in max_docfreq_values) {
    dfm_trimmed <- dfm_trim(dfm_obj, min_docfreq = min_df, max_docfreq = max_df, docfreq_type = "prop", verbose = TRUE)
    dtm <- convert(dfm_trimmed, to = "topicmodels")
    
    lda_model <- LDA(dtm, k = initial_k, method = "Gibbs", 
                     control = list(seed = 1234, burnin = 1000, iter = 2000, thin = 500))
    
    # Calculate coherence score
    coherence <- FindTopicsNumber(
      dtm,
      topics = initial_k,
      metrics = "CaoJuan2009",
      method = "Gibbs",
      control = list(seed = 1234, burnin = 1000, iter = 2000, thin = 500),
      mc.cores = 1L,
      verbose = TRUE
    )
    
    coherence_score <- coherence$CaoJuan2009
    
    # Store results
    pruning_results[[paste(min_df, max_df, sep = "_")]] <- list(
      dfm = dfm_trimmed, 
      model = lda_model, 
      coherence = coherence_score
    )
  }
}

# Find the best pruning parameters based on coherence
best_pruning <- names(pruning_results)[which.min(sapply(pruning_results, function(x) x$coherence))]
best_min_df <- as.numeric(strsplit(best_pruning, "_")[[1]][1])
best_max_df <- as.numeric(strsplit(best_pruning, "_")[[1]][2])

# set the number of topics
k <- initial_k

# set parameters and prune DFM
dfm_trimmed <- dfm_trim(dfm_obj, min_docfreq = best_min_df , max_docfreq = best_max_df, docfreq_type = "prop", verbose = TRUE)

# convert to DTM
dtm <- convert(dfm_trimmed, to = "topicmodels")

# using Gibbs to train LDA model
lda_model <- LDA(dtm, k = k, method = "Gibbs", control = list(seed = 1234, burnin = 1000, iter = 2000, thin = 500))

#sketch the result
topics <- terms(lda_model, 15)
print(topics)



# interactive visualization using LDAvis
phi <- posterior(lda_model)$terms
theta <- posterior(lda_model)$topics 

# count number of words
doc.length <- slam::row_sums(dtm)

# vocabulary table
vocab <- colnames(dtm)

# frequency of each word
term.frequency <- slam::col_sums(dtm)

# create JSON object for LDAvis
json_lda <- LDAvis::createJSON(
  phi = phi, 
  theta = theta, 
  doc.length = doc.length, 
  vocab = vocab, 
  term.frequency = term.frequency
)

# open the visualization
serVis(json_lda)



#visualization using Faceted Bar Chart
library(ggplot2)
library(dplyr)

ap_top_terms <- ap_topics %>%
  group_by(topic) %>%
  slice_max(beta, n = 10) %>% 
  ungroup() %>%
  arrange(topic, -beta)

ap_top_terms %>%
  mutate(term = reorder_within(term, beta, topic)) %>%
  ggplot(aes(beta, term, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  scale_y_reordered()



#check the spiking effect of rep_doc

# Identify the document ID of the spiked document
spiked_doc_id <- tail(grouped_content_words$doc_id, 1)

# Extract the topic distribution for the spiked document
spiked_doc_topic_distribution <- posterior(lda_model)$topics[spiked_doc_id, ]

# Print the topic distribution for the spiked document
print(spiked_doc_topic_distribution)

# Sort the topic distribution for better readability
sorted_spiked_doc_topic_distribution <- sort(spiked_doc_topic_distribution, decreasing = TRUE)
print(sorted_spiked_doc_topic_distribution)

