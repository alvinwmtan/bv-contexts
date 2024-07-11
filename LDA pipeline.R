install.packages(c("topicmodels", "tm", "SnowballC", "childesr", "dplyr", "magrittr", "stringr", "quanteda", "udpipe", "LDAvis", "tidytext","ldatuning","tidyverse","viridis","Rstne"))


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
library(ldatuning)
library(readr)
library(viridis) #for enough colors
library(tidyverse)
library(Rtsne)
library(plotly)

# UDpipe model to subtract content words (eliminate English)
ud_model <- udpipe_download_model(language = "english")
ud_model <- udpipe_load_model(file = ud_model$file_model)

# get CHILDES corpus
texts1 <- get_utterances(corpus = "Providence")

# chunk into fixed utterance per documents
grouped_data <- texts1 %>%
  mutate(doc_id = rep(1:(n() %/% 300 + 1), each = 300, length.out = n())) %>%
  group_by(doc_id) %>%
  summarise(document = paste(stem, collapse = " ")) %>%
  ungroup()


# Read booksharing documents
booksharing_files <- list.files(path = "C:/Users/Shawn/Desktop/meal time", full.names = TRUE, pattern = "*.txt")
booksharing_docs <- sapply(booksharing_files, read_file)

# Read mealtime documents
mealtime_files <- list.files(path = "C:/Users/Shawn/Desktop/book sharing", full.names = TRUE, pattern = "*.txt")
mealtime_docs <- sapply(mealtime_files, read_file)

# Create dataframes for the new documents
booksharing_df <- data.frame(
  doc_id = (max(grouped_data$doc_id, na.rm = TRUE) + 1):(max(grouped_data$doc_id, na.rm = TRUE) + length(booksharing_docs)),
  document = booksharing_docs,
  stringsAsFactors = FALSE
)
mealtime_df <- data.frame(
  doc_id = (max(grouped_data$doc_id, na.rm = TRUE) + length(booksharing_docs) + 1):(max(grouped_data$doc_id, na.rm = TRUE) + length(booksharing_docs) + length(mealtime_docs)),
  document = mealtime_docs,
  stringsAsFactors = FALSE
)

# Combine with original grouped_data
grouped_data <- bind_rows(grouped_data, booksharing_df, mealtime_df)


#annotate each document (clean name)
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


#find out the best pruning parameter based on topic coherence
# Define ranges for pruning parameters
min_docfreq_values <- seq(0.0005, 0.01, by = 0.001)
max_docfreq_values <- seq(0.3, 0.6, by = 0.1)

# Fixed number of topics for initial evaluation
initial_k <- 20

# Store results
results <- expand.grid(min_df = min_docfreq_values, max_df = max_docfreq_values)
results$coherence <- NA

# Evaluate combinations of pruning parameters
for (i in 1:nrow(results)) {
  min_df <- results$min_df[i]
  max_df <- results$max_df[i]
  
  dfm_trimmed <- dfm_trim(dfm_obj, min_docfreq = min_df, max_docfreq = max_df, docfreq_type = "prop", verbose = TRUE)
  dtm <- convert(dfm_trimmed, to = "topicmodels")
  
  lda_model <- LDA(dtm, k = initial_k, method = "Gibbs", 
                   control = list(seed = 1234, burnin = 1000, iter = 2000, thin = 500))
  
  coherence <- FindTopicsNumber(
    dtm,
    topics = initial_k,
    metrics = "CaoJuan2009",
    method = "Gibbs",
    control = list(seed = 1234, burnin = 1000, iter = 2000, thin = 500),
    mc.cores = 1L,
    verbose = TRUE
  )
  
  results$coherence[i] <- coherence$CaoJuan2009
}

# Plot the coherence scores as a three dimensional scatter plot
plot_ly(results, x = ~min_df, y = ~max_df, z = ~coherence, type = 'scatter3d', mode = 'markers',
        marker = list(size = 5, color = ~coherence, colorscale = 'Viridis', showscale = TRUE)) %>%
  layout(title = "Topic Coherence Across Pruning Parameters",
         scene = list(xaxis = list(title = 'Minimum Document Frequency'),
                      yaxis = list(title = 'Maximum Document Frequency'),
                      zaxis = list(title = 'Topic Coherence')))


#use the best pruning parameter
dfm_trimmed <- dfm_trim(dfm_obj, min_docfreq = 0.0015 , max_docfreq = 0.6, docfreq_type = "prop", verbose = TRUE)

# convert to DTM
dtm <- convert(dfm_trimmed, to = "topicmodels")

# using Gibbs to train LDA model
lda_model <- LDA(dtm, k = 20, method = "Gibbs", control = list(seed = 1234, burnin = 1000, iter = 2000, thin = 500))

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
ap_topics <- tidy(lda_model, matrix = "beta")
ap_topics
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



# visualization using t-SNE
# Get topic distributions
theta <- posterior(lda_model)$topics 

# Perform t-SNE
set.seed(1234)
tsne_model <- Rtsne(theta, dims = 2, perplexity = 30, verbose = TRUE)

# Create a data frame with t-SNE results
tsne_df <- data.frame(tsne_model$Y)
colnames(tsne_df) <- c("Dim1", "Dim2")
tsne_df$doc_id <- as.character(rownames(tsne_df))

# Identify the spiking documents
total_docs <- nrow(tsne_df)
meal_time_docs <- as.character((total_docs - 9):(total_docs - 5))
book_sharing_docs <- as.character((total_docs - 4):total_docs)

# Add markers to t-SNE data frame
tsne_df <- tsne_df %>%
  mutate(
    spiking = case_when(
      doc_id %in% meal_time_docs ~ "Meal Time",
      doc_id %in% book_sharing_docs ~ "Book Sharing",
      TRUE ~ "Normal Documents"
    ),
    shape = case_when(
      doc_id %in% meal_time_docs ~ "triangle",
      doc_id %in% book_sharing_docs ~ "rectangle",
      TRUE ~ "circle"
    )
  )

# Check for missing values in the relevant columns
print("Checking for missing values in relevant columns:")
print(sapply(tsne_df[, c("Dim1", "Dim2", "spiking", "shape")], function(x) sum(is.na(x))))

# Ensure the values in the 'shape' column are valid
print("Unique values in the 'shape' column:")
print(unique(tsne_df$shape))

# Ensure the values in the 'spiking' column are valid
print("Unique values in the 'spiking' column:")
print(unique(tsne_df$spiking))

# Determine the dominant topic for each document
tsne_df$dominant_topic <- apply(theta, 1, which.max)
tsne_df$dominant_topic <- as.factor(tsne_df$dominant_topic)

# Remove rows with missing values
tsne_df <- na.omit(tsne_df)

# Plotting with ggplot2
p <- ggplot(tsne_df, aes(x = Dim1, y = Dim2, color = dominant_topic, shape = shape, size = spiking)) +
  geom_point(alpha = 0.7) +
  scale_shape_manual(values = c("circle" = 16, "triangle" = 17, "rectangle" = 15)) +
  scale_size_manual(values = c("Normal Documents" = 3, "Meal Time" = 6, "Book Sharing" = 6)) +
  labs(title = "t-SNE Visualization of Document Topic Distributions",
       x = "Dimension 1", y = "Dimension 2",
       color = "Dominant Topic",
       shape = "Document Type",
       size = "Document Type") +
  theme_minimal() +
  theme(legend.position = "right") +
  guides(shape = guide_legend(override.aes = list(size = c(3, 6, 6))),
         size = guide_legend(override.aes = list(shape = c(16, 17, 15))))

# Adjust legend titles
p + scale_shape_manual(
  values = c("circle" = 16, "triangle" = 17, "rectangle" = 15),
  labels = c("Normal Documents", "Meal Time", "Book Sharing")) + 
  scale_size_manual(
  values = c("Normal Documents" = 3, "Meal Time" = 6, "Book Sharing" = 6),
  labels = c("Normal Documents", "Meal Time", "Book Sharing"))
  


# dynamics plot
# Assuming `theta` is the matrix with topic distributions for each segment
# Select the first 100 segments/documents
topic_proportions <- as.data.frame(theta)
topic_proportions$document <- 1:nrow(topic_proportions)

# Convert to long format for ggplot2
topic_proportions_long <- topic_proportions %>%
  pivot_longer(cols = -document, names_to = "topic", values_to = "proportion")

# Ensure topics are ordered numerically
topic_proportions_long$topic <- factor(topic_proportions_long$topic, levels = as.character(1:ncol(theta)))

# Generate a color palette with enough colors for all topics using viridis
colors <- viridis_pal(option = "C")(ncol(theta))

# Plotting with ggplot2
ggplot(topic_proportions_long, aes(x = factor(document), y = proportion, fill = topic)) +
  geom_bar(stat = "identity") +
  scale_fill_manual(values = colors) +
  labs(title = "Topic Proportions Over All Document Segments",
       x = "Document",
       y = "Proportion of topics",
       fill = "Topic") +
  theme_minimal() +
  theme(legend.position = "right",
        axis.text.x = element_text(angle = 90, hjust = 1, size = 8)) +  # Rotate x-axis labels and adjust text size
  scale_x_discrete(breaks = seq(1, 1544, by = 100),  # Show label every 100 documents
                   labels = function(x) paste0("Doc ", x))  # Customize label format
