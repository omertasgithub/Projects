setwd("/Users/omertas/Documents/BU_classes/cs555/final project")

student <-read.csv("StudentsPerformance.csv")
set.seed(12929)

#We are only interested the students had no prep
#Among these students we will investigate prental level education
#and lunch
#Also we will investigate parental level education in two groups high school and college
#some high school and highschool will be considered as only high school
#associate's degree, bachelor's degree, master's degree and some college
#will be onder college
#for this purpose we will run the following code
student$parental.level.of.education <- 
  ifelse(student$parental.level.of.education== "high school" |
           student$parental.level.of.education == "some high school", 
         "high school", "college")

#As I mentioned above we are only intersted none test prep students
#and we will investigate how their lunch and parental education (high vs college)
#affect their math score
student <- student[student$test.preparation.course=="none", 
                   c("parental.level.of.education", "lunch", "math.score")]
student <- 
  student[sample(1:nrow(student), 500), ]

#It is always good to use graph to have some ideas
boxplot(student$math.score~student$lunch, data = student, 
        main = "none prep Students Test Score by Lunch", xlab = "lunch", ylab = "Math Score")
boxplot(student$math.score~student$parental.level.of.education, data = student, 
        main = "none prep Student Math Score by Parental education", 
        xlab = "Parental Level of Education", ylab = "math score")
#Sometimes graph may not give you the whole stroies
outlier_iqr <- function(x){
  iqr <- IQR(x,na.rm = T,type = 7)
  q <- quantile(x)
  upper_bound = q[4]+(iqr*1.5)
  lower_bound = q[2]-(iqr*1.5)
  outliers <- which ((x > upper_bound) | (x < lower_bound))
  return(outliers)
}


outliers <- outlier_iqr(student$math.score)
#Exclude otuliers 
student <- student[-c(outliers), ]


#Use the given data (among the students none test prep) to 
#test whether mean math score of standard lunch different 
#than mean math score of free/reduced lunch. (Formally Test). a = 0.05

#we can either run t or f-test (k=2)
#I will do f test

#Step1 Set up the hypotheses and and select alpha level
#H0: M1=M2
#H1:M1!=M2
#Step2 Select the approptiate test stat
#F = MSB/MSW
#Step 3 State decison rule
n <- nrow(student)
k <- 2
critical <- qf(0.95, k-1, n-k)
#Step3 decision rule
#Reject H0 if F >= critical
#otherwise do not reject H0

#Step4 Compute test statistic
m <- aov(student$math.score~student$lunch)
smmry <- summary(m)
F <- smmry[[1]][1,4]

#Step5 Conclusion
if (F >= critical) {
  print("Reject H0")
} else {print("Fail to reject H0)")}

#We have siginificant evidence at the a = 0.05 level there is a difference
#in math score between lunc(free reduced, standard)


#Now we are going to do to two way Anova to see parental level education
#effects. And we will tall whether they have significant interections
library(car)

attach(student)
my_model <- 
  lm(math.score~lunch+parental.level.of.education+lunch*parental.level.of.education)
  
summary(my_model)  
Anova(my_model, type = 3)
  

interaction.plot(lunch, parental.level.of.education, math.score, col=1:2)  
#They do not cross. Therefore they do not cancel out
#lunch                             2.457e-10 ***
#parental.level.of.education        0.009955 ** 
#lunch:parental.level.of.education  0.603818    
#Therefore, lunch and parental level of education are significant
#and they are not significant interection (do not cancel out)
#No need to test differently. We can test both at the same time
#in two way anova
