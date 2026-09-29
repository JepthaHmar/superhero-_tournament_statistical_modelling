#############################################################################
# STATISTICAL MODELLING OF THE ULTIMATE SUPERHERO TOURNAMENT
# A data-driven clash of universes using Principal Component Analysis,
# K-Means Clustering, Logistic Regression and Markov Chain transitions
#
# Group 7 - Final Project
# Course       : Statistical Learning Models using R
# Institution  : Indian Institute of Management Kashipur
# Programme    : MBA (Analytics), Batch 2025-27
# Submitted to : Prof. Jimut Bahan Chakraborty
#
# Group Members:
#   - Abdul Arham Shamim
#   - Anjali
#   - Chahat Kamalni
#   - Jeptha H Lalawmpuia
#   - Nandini Talwar
#
# Data         : superheroes_nlp_dataset.csv
#                (Superheroes NLP Dataset, Kaggle - scraped from SuperHeroDB)
# Requirements : R 4.x. Uses base R plus the 'cluster' package, which ships
#                with every standard R installation (no install needed).
# How to run   : Keep the CSV in the data/ folder (or next to this script),
#                then run from the repository root:
#                  Rscript superhero_tournament.R
#                or source() it from RStudio.
# Outputs      : Console results + plots saved to ./tournament_outputs/
#
# CORE QUESTION
#   Can statistical modelling turn superhero battles from subjective fan
#   debates into a data-driven tournament, and which characteristics give
#   a superhero the greatest probability of winning?
#
# PIPELINE
#   1. Data cleaning      : keep Marvel & DC, remove duplicates and
#                           incomplete records, retain 6 ability predictors
#   2. PCA                : compress 6 correlated abilities into a single
#                           Ability Index (PC1); visualise PC1 vs PC2
#   3. K-Means            : exploratory grouping into combat archetypes
#                           (NOT used to predict matchups)
#   4. Logistic regression: binomial GLM on pairwise ability differences,
#                           with overall-score difference as a proxy winner
#   5. Markov transitions : propagate win probabilities through a 16-hero
#                           knockout bracket (R16 -> QF -> SF -> F -> Champion)
#   6. Publisher view     : aggregate championship odds, Marvel vs DC
#   7. Sensitivity        : insert a perfectly balanced new hero (40-100 on
#                           all six abilities) and find the tipping point
#
# MODELLING NOTE
#   The dataset contains NO real battle results. The winner of each
#   training pair is therefore a PROXY: the hero with the higher
#   overall_score. Coefficients describe which ability gaps line up with
#   that proxy, not verified fight outcomes (see "Future Work" in the deck).
#############################################################################


## ================================================================
## 0. SETUP
## ================================================================
rm(list = ls())
suppressPackageStartupMessages(library(cluster))   # silhouette()
set.seed(2026)                                     # reproducibility

DATA_FILE  <- "superheroes_nlp_dataset.csv"
DATA_PATH  <- if (file.exists(file.path("data", DATA_FILE))) file.path("data", DATA_FILE) else DATA_FILE
OUTPUT_DIR <- "tournament_outputs"
if (!dir.exists(OUTPUT_DIR)) dir.create(OUTPUT_DIR)

MARVEL_COL <- "#C0392B"
DC_COL     <- "#2471A3"

ability_cols <- c("intelligence_score", "strength_score", "speed_score",
                  "durability_score", "power_score", "combat_score")
ability_lbl  <- c("Intelligence", "Strength", "Speed",
                  "Durability", "Power", "Combat")


## ================================================================
## 1. LOAD AND CLEAN THE DATA
## ================================================================
raw <- read.csv(DATA_PATH, stringsAsFactors = FALSE, na.strings = c("", "NA"))
cat("Raw dataset:", nrow(raw), "rows\n")

heroes <- raw[, c("name", "creator", "alignment", "overall_score", ability_cols)]

# Coerce scores to numeric. Rows corrupted by stray quotes/commas in the
# free-text fields (or non-numeric entries such as "Infinity") become NA
# here and are removed by the completeness filter below.
for (cc in c("overall_score", ability_cols)) {
  heroes[[cc]] <- suppressWarnings(as.numeric(heroes[[cc]]))
}

heroes <- heroes[!is.na(heroes$name) & heroes$name != "", ]
heroes <- heroes[heroes$creator %in% c("Marvel Comics", "DC Comics"), ]
heroes <- heroes[!duplicated(heroes$name), ]
heroes <- heroes[complete.cases(heroes[, c("overall_score", ability_cols)]), ]
rownames(heroes) <- NULL

cat("Cleaned profiles (Marvel + DC, unique, complete):", nrow(heroes), "\n")
print(table(heroes$creator))

cat("\n--- Correlation between the six abilities ---\n")
print(round(cor(heroes[, ability_cols]), 2))


## ================================================================
## 2. PCA - BUILDING THE ABILITY INDEX
## ================================================================
# Abilities are standardised before projection so that no single
# dimension dominates because of its scale.
pca <- prcomp(heroes[, ability_cols], center = TRUE, scale. = TRUE)

# Orient PC1 so that a higher value means a stronger character
pc_sign <- ifelse(cor(pca$x[, 1], rowSums(heroes[, ability_cols])) < 0, -1, 1)
heroes$PC1 <- pc_sign * pca$x[, 1]
heroes$PC2 <- pca$x[, 2]

var_exp <- summary(pca)$importance[2, ]
loadings_pc1 <- pc_sign * pca$rotation[, 1]
names(loadings_pc1) <- ability_lbl

cat(sprintf("\nPC1 explains %.1f%% of variance (Ability Index)\n", 100 * var_exp[1]))
cat(sprintf("PC2 explains %.1f%% of variance\n", 100 * var_exp[2]))
cat("\n--- PC1 loadings ---\n")
print(round(sort(loadings_pc1, decreasing = TRUE), 3))

png(file.path(OUTPUT_DIR, "01_pca_scree.png"), width = 700, height = 500)
barplot(100 * var_exp, names.arg = paste0("PC", 1:6), col = "grey40",
        ylab = "% variance explained", main = "PCA Scree Plot - Six Abilities")
invisible(dev.off())

png(file.path(OUTPUT_DIR, "02_pca_pc1_vs_pc2.png"), width = 800, height = 600)
plot(heroes$PC1, heroes$PC2, pch = 19, cex = 0.7,
     col = adjustcolor(ifelse(heroes$creator == "Marvel Comics", MARVEL_COL, DC_COL), 0.6),
     xlab = sprintf("PC1 - Ability Index (%.1f%%)", 100 * var_exp[1]),
     ylab = sprintf("PC2 (%.1f%%)", 100 * var_exp[2]),
     main = "Character Coordinate Space: Marvel vs DC")
legend("topleft", c("Marvel", "DC"), col = c(MARVEL_COL, DC_COL), pch = 19, bty = "n")
invisible(dev.off())


## ================================================================
## 3. K-MEANS - EXPLORATORY COMBAT ARCHETYPES
## ================================================================
# Clustering is purely descriptive; it is not used in the matchup model.
Z <- scale(heroes[, ability_cols])
d <- dist(Z)
k_range <- 2:6
sil <- sapply(k_range, function(k) {
  km <- kmeans(Z, centers = k, nstart = 25)
  mean(silhouette(km$cluster, d)[, 3])
})
names(sil) <- k_range
cat("\n--- Average silhouette width by K ---\n")
print(round(sil, 3))

# Silhouette peaks at K = 2 (grounded vs powerhouse). K = 3 is chosen to
# isolate a richer mid-tier while staying close on silhouette.
K <- 3
km <- kmeans(Z, centers = K, nstart = 25)

# Label clusters from their profiles:
#   highest average Ability Index            -> Cosmic Powerhouses
#   of the rest, highest average Combat      -> Street-Level / Skill-Based
#   remaining cluster                        -> Mid-Tier / Specialists
mean_pc1    <- tapply(heroes$PC1, km$cluster, mean)
mean_combat <- tapply(heroes$combat_score, km$cluster, mean)
cosmic <- names(which.max(mean_pc1))
rest   <- setdiff(names(mean_pc1), cosmic)
street <- rest[which.max(mean_combat[rest])]
labels <- setNames(rep("Mid-Tier / Specialists", K), names(mean_pc1))
labels[cosmic] <- "Cosmic Powerhouses"
labels[street] <- "Street-Level / Skill-Based"
heroes$archetype <- unname(labels[as.character(km$cluster)])

cat("\n--- Archetype sizes ---\n")
print(table(heroes$archetype))
cat("\n--- Archetype profiles (mean ability scores) ---\n")
arch_prof <- aggregate(heroes[, ability_cols], list(Archetype = heroes$archetype), mean)
arch_prof[, -1] <- round(arch_prof[, -1], 1)
print(arch_prof, row.names = FALSE)

png(file.path(OUTPUT_DIR, "03_kmeans_silhouette.png"), width = 700, height = 500)
plot(k_range, sil, type = "b", pch = 19, xlab = "Number of clusters (K)",
     ylab = "Average silhouette width", main = "Choosing K for K-Means")
abline(v = K, lty = 2, col = "grey50")
invisible(dev.off())

arch_cols <- c("Cosmic Powerhouses" = "#8E44AD", "Mid-Tier / Specialists" = "#F39C12",
               "Street-Level / Skill-Based" = "#27AE60")
png(file.path(OUTPUT_DIR, "04_kmeans_archetypes.png"), width = 800, height = 600)
plot(heroes$PC1, heroes$PC2, pch = 19, cex = 0.7,
     col = adjustcolor(arch_cols[heroes$archetype], 0.7),
     xlab = "PC1 - Ability Index", ylab = "PC2",
     main = "K-Means (K = 3) Combat Archetypes")
legend("topleft", names(arch_cols), col = arch_cols, pch = 19, bty = "n")
invisible(dev.off())


## ================================================================
## 4. LOGISTIC REGRESSION - PAIRWISE WIN PROBABILITY
## ================================================================
# Training cases: random pairs of heroes (A, B).
#   Predictors : ability differences  A - B  (six variables)
#   Target     : 1 if A has the higher overall_score (proxy winner), else 0
# Pairs with identical overall scores are dropped (no proxy winner).
N_PAIRS <- 20000
a_idx <- sample(nrow(heroes), N_PAIRS, replace = TRUE)
b_idx <- sample(nrow(heroes), N_PAIRS, replace = TRUE)
ok <- a_idx != b_idx & heroes$overall_score[a_idx] != heroes$overall_score[b_idx]
a_idx <- a_idx[ok]; b_idx <- b_idx[ok]

pairs <- as.data.frame(as.matrix(heroes[a_idx, ability_cols]) -
                       as.matrix(heroes[b_idx, ability_cols]))
names(pairs) <- tolower(ability_lbl)
pairs$A_wins <- as.integer(heroes$overall_score[a_idx] > heroes$overall_score[b_idx])
cat("\nPairwise training cases:", nrow(pairs), "\n")

glm_fit <- glm(A_wins ~ ., data = pairs, family = binomial)
cat("\n--- Binomial GLM on ability differences ---\n")
print(summary(glm_fit)$coefficients, digits = 3)

beta <- coef(glm_fit)[-1]
cat("\nCoefficients ranked (per 1-point ability advantage):\n")
print(round(sort(beta, decreasing = TRUE), 3))
cat("Note: a slightly negative Combat coefficient means extra combat rating\n",
    "does not predict wins once power and intelligence are accounted for.\n", sep = "")

png(file.path(OUTPUT_DIR, "05_logistic_coefficients.png"), width = 700, height = 500)
b_sorted <- sort(beta)
barplot(b_sorted, horiz = TRUE, las = 1, col = ifelse(b_sorted < 0, "#E74C3C", "#34495E"),
        main = "Logistic Regression Coefficients", xlab = "Log-odds per 1-point advantage")
abline(v = 0)
invisible(dev.off())

# Win probability for hero A vs hero B. Only the ability gap is used (no
# intercept), so the model is symmetric and P(B wins) = 1 - P(A wins).
win_prob <- function(abil_a, abil_b) {
  plogis(sum(beta * (as.numeric(abil_a) - as.numeric(abil_b))))
}


## ================================================================
## 5. BUILD THE ROUND-OF-16 BRACKET
## ================================================================
# 16 entrants: the top 8 Marvel and top 8 DC characters by Ability Index.
# Seeds 1-16 are assigned by PC1; exact ties are broken by publisher block
# (Marvel first) and then alphabetically, so the bracket is reproducible.
# Standard seeding keeps top seeds apart: 1v16, 8v9, 5v12, 4v13, 6v11,
# 3v14, 7v10, 2v15.
SEED_ORDER <- c(1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15)

top_by_pub <- function(pub, n = 8) {
  x <- heroes[heroes$creator == pub, ]
  x[order(-x$PC1, x$name), ][1:n, ]
}

seed_roster <- function(pool) {
  pool <- pool[order(-pool$PC1, pool$tiebreak, pool$name), ][1:16, ]
  pool$seed <- 1:16
  pool
}

entrants <- rbind(top_by_pub("Marvel Comics"), top_by_pub("DC Comics"))
entrants$tiebreak <- ifelse(entrants$creator == "Marvel Comics", 1, 2)
roster <- seed_roster(entrants)

cat("\n--- Round-of-16 roster ---\n")
print(roster[, c("seed", "name", "creator", "PC1")], row.names = FALSE, digits = 4)


## ================================================================
## 6. MARKOV TRANSITIONS THROUGH THE BRACKET
## ================================================================
# State = the round a hero has reached. The probability of advancing
# depends only on the current round and the possible opponents in it
# (the Markov property). For each round:
#   P(hero advances) = P(hero reached this round)
#                      x SUM over opponents [ P(opponent reached) x P(hero beats opponent) ]
#
# PROBABILITY AXIOM (QA rule): the pairwise matrix is filled so that
# P(B beats A) = 1 - P(A beats B). This guarantees the championship
# probabilities across all 16 heroes sum to exactly 1.
solve_tournament <- function(roster) {
  br <- roster[match(SEED_ORDER, roster$seed), ]
  n  <- nrow(br)
  abil <- as.matrix(br[, ability_cols])
  P <- matrix(0, n, n)
  for (i in 1:(n - 1)) for (j in (i + 1):n) {
    P[i, j] <- win_prob(abil[i, ], abil[j, ])
    P[j, i] <- 1 - P[i, j]
  }

  S <- matrix(0, n, 5)      # columns: R16, QF, SF, Final, Champion
  S[, 1] <- 1
  for (r in 1:4) {
    block <- 2^r
    for (start in seq(1, n, by = block)) {
      idx <- start:(start + block - 1)
      left  <- idx[1:(block / 2)]
      right <- idx[(block / 2 + 1):block]
      for (i in left)  S[i, r + 1] <- S[i, r] * sum(S[right, r] * P[i, right])
      for (j in right) S[j, r + 1] <- S[j, r] * sum(S[left, r]  * P[j, left])
    }
  }
  stopifnot(abs(sum(S[, 5]) - 1) < 1e-9)     # probability axiom check

  out <- data.frame(Seed = br$seed, Hero = br$name, Creator = br$creator,
                    Quarterfinal = S[, 2], Semifinal = S[, 3],
                    Final = S[, 4], Champion = S[, 5])
  out[order(-out$Champion), ]
}

results <- solve_tournament(roster)
rownames(results) <- NULL

cat("\n--- Survival probabilities by stage (top 8 contenders) ---\n")
show <- results[1:8, ]
show[, 4:7] <- round(100 * show[, 4:7], 1)
print(show, row.names = FALSE)

champ <- results[1, ]
cat(sprintf("\nModelled champion: %s (%s) with %.1f%% championship odds (seed %d)\n",
            champ$Hero, champ$Creator, 100 * champ$Champion, champ$Seed))
cat("Raw ability rank and tournament odds diverge: bracket position matters.\n")

png(file.path(OUTPUT_DIR, "06_champion_probabilities.png"), width = 850, height = 600)
par(mar = c(5, 14, 4, 2))
top_plot <- results[16:1, ]
barplot(100 * top_plot$Champion, names.arg = top_plot$Hero, horiz = TRUE, las = 1,
        col = ifelse(top_plot$Creator == "Marvel Comics", MARVEL_COL, DC_COL),
        xlab = "Championship probability (%)", main = "Who Wins the Tournament?")
legend("bottomright", c("Marvel", "DC"), fill = c(MARVEL_COL, DC_COL), bty = "n")
invisible(dev.off())


## ================================================================
## 7. MARVEL VS DC - AGGREGATED CHAMPIONSHIP PROBABILITY
## ================================================================
pub_odds <- tapply(results$Champion, results$Creator, sum)
cat("\n--- Aggregate championship probability by publisher ---\n")
print(round(100 * pub_odds, 1))
cat("This reflects bracket composition and seeding, not a claim that one\n",
    "publisher's characters are canonically stronger.\n", sep = "")

png(file.path(OUTPUT_DIR, "07_marvel_vs_dc.png"), width = 550, height = 550)
pie(pub_odds, labels = sprintf("%s\n%.1f%%", sub(" Comics", "", names(pub_odds)), 100 * pub_odds),
    col = c(DC_COL, MARVEL_COL)[match(names(pub_odds), c("DC Comics", "Marvel Comics"))],
    main = "Share of Championship Probability")
invisible(dev.off())


## ================================================================
## 8. SENSITIVITY ANALYSIS - A BALANCED NEW ENTRANT
## ================================================================
# A new hero with the SAME rating on all six abilities (40 to 100) is
# projected into the existing PCA space, added to the 16 entrants, and the
# roster is re-sorted so the weakest entrant is bumped. A new hero that
# ties the top Ability Index is seeded ahead of existing ties.
project_pc1 <- function(abil) {
  nd <- as.data.frame(t(setNames(abil, ability_cols)))
  pc_sign * predict(pca, newdata = nd)[1, 1]
}

levels_tested <- seq(40, 100, by = 10)
cutoff_pc1 <- min(roster$PC1)
sens <- data.frame(Rating = levels_tested, PC1 = NA, Makes_Top16 = NA,
                   Champion_Prob = NA, Marvel_Share = NA)

for (k in seq_along(levels_tested)) {
  lvl <- levels_tested[k]
  nh <- data.frame(name = "New Balanced Hero", creator = "New Entrant",
                   alignment = NA, overall_score = NA,
                   t(setNames(rep(lvl, 6), ability_cols)))
  nh$PC1 <- project_pc1(rep(lvl, 6)); nh$PC2 <- NA
  nh$archetype <- NA; nh$tiebreak <- 0; nh$seed <- NA

  pool <- rbind(roster, nh)
  new_roster <- seed_roster(pool)
  res <- solve_tournament(new_roster)

  sens$PC1[k]           <- nh$PC1
  sens$Makes_Top16[k]   <- "New Balanced Hero" %in% new_roster$name
  sens$Champion_Prob[k] <- sum(res$Champion[res$Hero == "New Balanced Hero"])
  sens$Marvel_Share[k]  <- sum(res$Champion[res$Creator == "Marvel Comics"])
}

cat(sprintf("\nRoster cut-off (16th entrant's Ability Index): %.3f\n", cutoff_pc1))
cat("\n--- Sensitivity: balanced new hero ---\n")
sens_show <- sens
sens_show$PC1 <- round(sens_show$PC1, 3)
sens_show$Champion_Prob <- round(100 * sens_show$Champion_Prob, 1)
sens_show$Marvel_Share  <- round(100 * sens_show$Marvel_Share, 1)
print(sens_show, row.names = FALSE)

png(file.path(OUTPUT_DIR, "08_sensitivity_tipping_point.png"), width = 800, height = 550)
plot(sens$Rating, 100 * sens$Champion_Prob, type = "b", pch = 19, lwd = 2,
     xlab = "Rating on all six abilities", ylab = "Championship probability (%)",
     main = "Tipping Point for a Balanced New Entrant")
invisible(dev.off())


## ================================================================
## 9. SUMMARY
## ================================================================
cat("\n================ SUMMARY ================\n")
cat(sprintf("Profiles analysed          : %d Marvel & DC characters\n", nrow(heroes)))
cat(sprintf("Ability Index (PC1)        : %.1f%% of variance\n", 100 * var_exp[1]))
cat(sprintf("K-Means                    : K = %d archetypes (silhouette %.2f)\n", K, sil[as.character(K)]))
cat(sprintf("Strongest predictor        : %s (coef %.3f)\n",
            names(which.max(beta)), max(beta)))
cat(sprintf("Modelled champion          : %s, %.1f%%\n", champ$Hero, 100 * champ$Champion))
cat(sprintf("Marvel vs DC               : %.1f%% vs %.1f%%\n",
            100 * pub_odds["Marvel Comics"], 100 * pub_odds["DC Comics"]))
cat(sprintf("Balanced entrant at 100    : %.1f%% championship odds\n",
            100 * tail(sens$Champion_Prob, 1)))
cat("Plots saved to:", normalizePath(OUTPUT_DIR), "\n")
