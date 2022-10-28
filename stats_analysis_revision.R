#library(plyr)
library(dplyr)
library(lmerTest)
library(brms)
library(ggplot2)
library(zplyr)
library(MuMIn)
library(simr)
library("rstan")
library(multcomp)
library(Rmisc)
library(emmeans)
library(lsmeans)

setwd('/Users/lijiaxuan/Desktop/model_material/NoisyChannel Supplementary Materials/revision_output/')

data <- read.csv('output_revision_master.csv')
data$P600 <- as.numeric(data$P600)
data$N400 <- as.numeric(data$N400)
data$Condition <- as.factor(data$Condition)
levels(data$Condition)

options(warn=-1)
for (x in c('reversal-1','reversal-2','animacy-1','animacy-2','animacy-3','substitution-1','substitution-2','substitution-3','syntax')){
  mydata <- subset(data,data$Label == x)
  N400_m0 <- lmer(N400 ~ Condition + (1|Item),data = mydata)
  p_n4 <- summary(N400_m0)
  cat(x, p_n4$coefficients[2,4], p_n4$coefficients[2,5],'\n')
  P600_m0 <- lmer(P600 ~ Condition + (1|Item),data = mydata)
  p_p6 <- summary(P600_m0)
  cat(x, p_p6$coefficients[2,4],p_p6$coefficients[2,5],'\n\n')
}
#############reversal1

mydata <- subset(data,data$Label == 'reversal-1')
N400_m0 <- lmer(N400 ~ Condition + (1|Item),data = mydata)
summary(N400_m0)
N400_m1 <- lmer(N400 ~  (1|Item),data = mydata)
anova(N400_m0,N400_m1)

P600_m0 <- lmer(P600 ~ Condition + (1|Item),data = mydata)
summary(P600_m0)
P600_m1 <- lmer(P600 ~  (1|Item),data = mydata)
anova(P600_m0,P600_m1)

#############reversal2
mydata <- subset(data,data$Label == 'reversal-2')
N400_m0 <- lmer(N400 ~ Condition + (1|Item),data = mydata)
summary(N400_m0)
N400_m1 <- lmer(N400 ~  (1|Item),data = mydata)
anova(N400_m0,N400_m1)

P600_m0 <- lmer(P600 ~ Condition + (1|Item),data = mydata)
summary(P600_m0)
P600_m1 <- lmer(P600 ~  (1|Item),data = mydata)
anova(P600_m0,P600_m1)

#############animacy_1_attr
mydata <- subset(data,data$Label == 'animacy-1')
N400_m0 <- lmer(N400 ~ Condition + (1|Item),data = mydata)
summary(N400_m0)
N400_m1 <- lmer(N400 ~  (1|Item),data = mydata)
anova(N400_m0,N400_m1)

P600_m0 <- lmer(P600 ~ Condition + (1|Item),data = mydata)
summary(P600_m0)
P600_m1 <- lmer(P600 ~  (1|Item),data = mydata)
anova(P600_m0,P600_m1)

#############animacy_2_nonattr
mydata <- subset(data,data$Label == 'animacy-2')
N400_m0 <- lmer(N400 ~ Condition + (1|Item),data = mydata)
summary(N400_m0)
N400_m1 <- lmer(N400 ~  (1|Item),data = mydata)
anova(N400_m0,N400_m1)

P600_m0 <- lmer(P600 ~ Condition + (1|Item),data = mydata)
summary(P600_m0)
P600_m1 <- lmer(P600 ~  (1|Item),data = mydata)
anova(P600_m0,P600_m1)

#############animacy_3
mydata <- subset(data,data$Label == 'animacy-3')
N400_m0 <- lmer(N400 ~ Condition + (1|Item),data = mydata)
summary(N400_m0)
N400_m1 <- lmer(N400 ~  (1|Item),data = mydata)
anova(N400_m0,N400_m1)

P600_m0 <- lmer(P600 ~ Condition + (1|Item),data = mydata)
summary(P600_m0)
P600_m1 <- lmer(P600 ~  (1|Item),data = mydata)
anova(P600_m0,P600_m1)

############sub1
mydata <- subset(data,data$Label == 'substitution-1')
N400_m0 <- lmer(N400 ~ Condition + (1|Item),data = mydata)
summary(N400_m0)
N400_m1 <- lmer(N400 ~  (1|Item),data = mydata)
anova(N400_m0,N400_m1)

P600_m0 <- lmer(P600 ~ Condition + (1|Item),data = mydata)
summary(P600_m0)
P600_m1 <- lmer(P600 ~  (1|Item),data = mydata)
anova(P600_m0,P600_m1)


############sub2
mydata <- subset(data,data$Label == 'substitution-2')
N400_m0 <- lmer(N400 ~ Condition + (1|Item),data = mydata)
summary(N400_m0)
N400_m1 <- lmer(N400 ~  (1|Item),data = mydata)
anova(N400_m0,N400_m1)

P600_m0 <- lmer(P600 ~ Condition + (1|Item),data = mydata)
summary(P600_m0)
P600_m1 <- lmer(P600 ~  (1|Item),data = mydata)
anova(P600_m0,P600_m1)

############sub3
mydata <- subset(data,data$Label == 'substitution-3')
N400_m0 <- lmer(N400 ~ Condition + (1|Item),data = mydata)
summary(N400_m0)
N400_m1 <- lmer(N400 ~  (1|Item),data = mydata)
anova(N400_m0,N400_m1)

P600_m0 <- lmer(P600 ~ Condition + (1|Item),data = mydata)
summary(P600_m0)
P600_m1 <- lmer(P600 ~  (1|Item),data = mydata)
anova(P600_m0,P600_m1)

####syntax
mydata <- subset(data,data$Label == 'syntax')
N400_m0 <- lmer(N400 ~ Condition + (1|Item),data = mydata)
summary(N400_m0)
N400_m1 <- lmer(N400 ~  (1|Item), data = mydata)
anova(N400_m0,N400_m1)

P600_m0 <- lmer(P600 ~ Condition + (1|Item),data = mydata)
summary(P600_m0)
P600_m1 <- lmer(P600 ~  (1|Item),data = mydata)
anova(P600_m0,P600_m1)
