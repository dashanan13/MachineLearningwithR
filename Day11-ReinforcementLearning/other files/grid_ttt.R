install.packages("devtools")
# install_github("nproellochs/ReinforcementLearning")
library(ReinforcementLearning)

states <- c("s1", "s2", "s3", "s4")
actions <- c("up", "down", "left", "right")

env <- gridworldEnvironment
print(env)

data <- sampleExperience(N = 1000, 
                         env = env, 
                         states = states, 
                         actions = actions)
head(data)

control <- list(alpha = 0.1, gamma = 0.5, epsilon = 0.1)

model <- ReinforcementLearning(data, 
                               s = "State", 
                               a = "Action", 
                               r = "Reward", 
                               s_new = "NextState", 
                               control = control)

computePolicy(model)
# summary(model)

data_unseen <- data.frame(State = c("s1", "s2", "s1"), 
                          stringsAsFactors = FALSE)

data_unseen$OptimalAction <- predict(model, data_unseen$State)

data_unseen

data_new <- sampleExperience(N = 1000, 
                             env = env, 
                             states = states, 
                             actions = actions, 
                             actionSelection = "epsilon-greedy",
                             model = model, 
                             control = control)


model_new <- ReinforcementLearning(data_new, 
                                   s = "State", 
                                   a = "Action", 
                                   r = "Reward", 
                                   s_new = "NextState", 
                                   control = control,
                                   model = model)

print(model_new)
plot(model_new)

#Tic Tac Toe

# Load dataset
data("tictactoe")

# Define reinforcement learning parameters
control <- list(alpha = 0.2, gamma = 0.4, epsilon = 0.1)

# Perform reinforcement learning
model <- ReinforcementLearning(tictactoe, s = "State", a = "Action", r = "Reward", 
                               s_new = "NextState", iter = 1, control = control)

# Calculate optimal policy
pol <- computePolicy(model)

# Print policy
head(pol)