setwd("Google Drive/Marc's Stuff/Current Classes/Lati 318 Readings in Cicero/DeRePublicaAnalysis/")

library(ggplot2)
library("reshape2")
library("plyr")

cic_phil_txts <- c("repub", "inventione", "orator", "optgen", "topica", "oratore", "fato", "paradoxa", "partitione", "brut", "consulatu", "leg", "fin", "tusc", "nd", "acad", "cat", "amic", "divinatione", "off", "compet")
cic_orat_txts <- c("quinc", "rosccom","legagr", "ver", "imp", "caecina","cluentio","rabirio", "cat", "murena", "sulla", "flacco", "arch", "postreditum", "domo", "haruspicum", "plancio", "sestio", "vatin", "cael", "prov", "balbo", "milo", "piso", "scauro", "fonteio", "rabirio", "marc", "lig", "deio", "phil")
cic_epis_txts <- c("att", "fam", "brut", "quinc")

cicero_data <- read.csv("cicero_data.csv")
cicero_phil <- cicero_data[cicero_data$text_name %in% cic_phil_txts,]
cicero_orat <- cicero_data[cicero_data$text_name %in% cic_orat_txts,]
cicero_epis <- cicero_data[cicero_data$text_name %in% cic_epis_txts,]
tacitus_data <- read.csv("tacitus_data.csv")
livy_data <- read.csv("livy_data.csv")

sum <- c("num_words", "num_chars", "num_sentences")
for (s in sum) {
  print(mean(cicero_phil[[s]]))
  print(mean(cicero_orat[[s]]))
  print("----")
}

cicero_data <- subset(cicero_data, select=-c(num_sentences, num_words,num_chars))
cicero_phil <- cicero_data[cicero_data$text_name %in% cic_phil_txts,]
cicero_orat <- cicero_data[cicero_data$text_name %in% cic_orat_txts,]
cicero_epis <- cicero_data[cicero_data$text_name %in% cic_epis_txts,]
tacitus_data <- subset(tacitus_data, select=-c(num_sentences, num_words,num_chars))
livy_data <- subset(livy_data, select=-c(num_sentences, num_words,num_chars))


repub <- cicero_data[cicero_data$text_name == 'repub',]
cic_phil <- data.frame("cic_phil", as.list(apply(cicero_phil[,-1], 2, sum)))
cic_orat <- data.frame("cic_orat", as.list(apply(cicero_orat[,-1], 2, sum)))
liv <- data.frame("liv", as.list(apply(livy_data[,-1], 2, sum)))
tac <- data.frame("tac", as.list(apply(tacitus_data[,-1], 2, sum)))

barplot <- function(row, title="") {
  row2 <- row
  row2[['num_interjection']] <- row2[['num_interjection']] + row2[['num_exclamation']]
  row2[['num_untagged']] <- row2[['num_unknown']] + row2[['num_untagged']] + row2[['num_punctuation']]
  row2[['num_noun']] <- row2[['num_noun']] + row2[['num_.']]
  row2 <- subset(row2, select = -c(num_exclamation, num_punctuation, num_unknown, num_.))
  df <- melt(row2)
  df <- mutate(df, prop = value / sum(value))
  df$pos <- rep(NA, nrow(df))
  for (i in 1:nrow(df)) {
    df[i,5] <- substr(df[i,2], 5, nchar(as.character(df[i,2])))
  }
  df$pos <- factor(df$pos,df[order(-df$prop),][['pos']] )
  ggplot(df, aes(x=pos,y=prop*100)) + geom_bar(stat = "identity", col = 'red', fill = 'red', alpha = 0.5) + ylab('Percentage of Words') + xlab('Part of Speech') + ggtitle(title) +
    theme(text = element_text(size=23),axis.text.x = element_text(angle=70, hjust=1, size = 30))
}

png(filename="presentation/fig1.png", width=900, height = 800)
barplot(repub, title = "Part of Speech Distribution in De Re Publica")
dev.off()

png(filename="presentation/fig2.png", width=900, height = 800)
barplot(cic_phil, title = "Average Part of Speech Distribution in Ciceronian Philosophy")
dev.off()

barplot(cic_orat, title = "Average Part of Speech Distribution in Ciceronian Oratory")
